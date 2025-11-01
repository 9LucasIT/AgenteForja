# app.py  —  WhatsApp Lead Agent (Veglienzone)
# FastAPI + LLM (OpenAI) con salida JSON estandarizada para n8n

import os
import json
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ===== 1) Prompt del asesor comercial (sistema) =====

AGENT_PROMPT = """
Sos el ASESOR COMERCIAL de Inmobiliaria Veglienzone (Rosario). Actuás como humano: cordial, cercano y resolutivo.
OBJETIVO: entender la necesidad del cliente y CALIFICAR el lead para derivarlo a un vendedor humano.

ESTILO:
- Una sola pregunta por turno.
- Agradecé y validá cada dato recibido (“Perfecto, venta en zona Centro y 2 dorm.”).
- Si ya hay datos suficientes, ofrecé 2–3 opciones sintéticas o pedí SOLO el dato faltante clave.
- Si te dan código de propiedad (ej. “A101”) o dirección exacta, priorizalo.
- Nunca listados eternos ni respuestas de FAQ. Nada de “soy IA”.

CAMPOS a reunir (pedilos solo si faltan y con naturalidad):
- Operación: alquiler/venta
- Ubicación: zona/barrio/dirección
- Tipo: depto/casa/ph/local/cochera/terreno
- Ambientes/dormitorios
- Presupuesto (rango o tope)
- Cochera: sí/no/indiferente
- Mascotas: sí/no
- Urgencia/plazo
- Nombre y teléfono de contacto (si no coincide con el WhatsApp)

FORMATO DE SALIDA (siempre DEVOLVÉ SOLO un JSON con estas claves):
{
  "reply_text": "texto para el cliente (obligatorio)",
  "closing_text": "texto de cierre si ya se califica (opcional, puede ser \"\")",
  "vendor_push": true/false,
  "vendor_message": "resumen para el vendedor si vendor_push=true"
}

CRITERIOS para vendor_push=true:
- Tiene al menos 3 de estos 4: operación, zona/dirección, tipo, (ambientes o presupuesto); O
- Pidió visita / agendar / hablar con asesor; O
- Dio un código de propiedad válido.

PLANTILLAS (podés adaptar):
- Acuse y repregunta: “¡Genial, {{operación}} en {{zona}}! ¿Tenés un presupuesto aproximado o un rango?”
- Ofrecer opciones: “Tengo estas opciones en {{zona}}: • A101 – 2 dorm, con cochera, USD 120.000 • C244 – 1 dorm, a estrenar, USD 85.000. ¿Te envío la ficha de alguna o ajustamos datos?”
- Cierre (closing_text): “Gracias, con esto ya puedo pasarte con un asesor para coordinar visita u opciones filtradas. Te escriben por este WhatsApp.”
- vendor_message (resumen): “LEAD CALIFICADO – Veglienzone | Cliente: {{nombre o ‘sin nombre’}} – WhatsApp: +{{telefono}} | Operación: {{operación}} | Zona: {{zona}} | Tipo: {{tipo}} | Ambientes: {{ambientes}} | Presupuesto: {{presupuesto}} | Notas: {{mascotas/cochera/urgencia/códigos}}”

EDGE CASES:
- Si dicen “hola/consulta/precio?”, preguntá qué buscan y dónde.
- Si es dirección exacta, devolvé ficha sintética de ESA propiedad y ofrecé coordinar visita o ver similares.
- Si piden algo que no cubrís, sé honesto y ofrecé alternativa.

RECORDATORIO: Devolvé SOLO el JSON con las 4 claves. Nada de texto fuera del JSON.
"""

# ===== 2) Utilidades =====

def parse_agent_json(text: str):
    """
    Intenta parsear el JSON del modelo y devuelve un dict con claves seguras.
    Si no es JSON válido, arma un fallback amable.
    """
    base = {
        "reply_text": "¡Hola! Soy el asesor de Veglienzone. ¿La búsqueda es para alquiler o para venta, y en qué zona?",
        "closing_text": "",
        "vendor_push": False,
        "vendor_message": ""
    }
    if not text:
        return base

    # recorte defensivo por si el modelo habla antes/después
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end+1]
        try:
            data = json.loads(snippet)
            base.update({
                "reply_text": (data.get("reply_text") or base["reply_text"]).strip(),
                "closing_text": (data.get("closing_text") or "").strip(),
                "vendor_push": bool(data.get("vendor_push", False)),
                "vendor_message": (data.get("vendor_message") or "").strip(),
            })
            return base
        except Exception:
            pass

    # si no se pudo parsear, devolvé al menos “reply_text”
    base["reply_text"] = text.strip() or base["reply_text"]
    return base


def build_messages(user_text: str):
    return [
        {"role": "system", "content": AGENT_PROMPT},
        {"role": "user", "content": (user_text or "").strip()},
    ]


# ===== 3) LLM (OpenAI); si no hay clave, usa mock para test =====

_OPENAI_OK = False
try:
    from openai import OpenAI  # pip install openai
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False

_client = None
if _OPENAI_OK and os.getenv("OPENAI_API_KEY"):
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "400"))

def call_llm(messages):
    # Modo real (si hay API Key)
    if _client is not None:
        resp = _client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return resp.choices[0].message.content

    # Modo mock (sin API Key) — te permite testear el flujo n8n/Green
    user = ""
    for m in messages:
        if m.get("role") == "user":
            user = (m.get("content") or "").lower()
            break

    # Respuesta mínima en JSON
    reply = "¡Genial! ¿La búsqueda es para alquiler o para venta y en qué zona?"
    closing = ""
    vpush = False
    vmsg = ""
    return json.dumps({
        "reply_text": reply,
        "closing_text": closing,
        "vendor_push": vpush,
        "vendor_message": vmsg
    })


# ===== 4) FastAPI =====

app = FastAPI(title="Veglienzone Lead Agent")

# CORS para pruebas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Inbound(BaseModel):
    user_phone: Optional[str] = ""
    text: str
    source: Optional[str] = "green"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/qualify")
def qualify(payload: Inbound):
    """
    Entrada desde n8n:
      { "user_phone": "549...", "text": "hola ...", "source": "green" }
    Salida (siempre):
      { reply_text, closing_text, vendor_push, vendor_message }
    """
    messages = build_messages(payload.text)
    raw = call_llm(messages)
    out = parse_agent_json(raw)

    # saneo defensivo final (evita nulls/arrays)
    result = {
        "reply_text": str(out.get("reply_text") or "").strip()[:3000],
        "closing_text": str(out.get("closing_text") or "").strip()[:2000],
        "vendor_push": bool(out.get("vendor_push", False)),
        "vendor_message": str(out.get("vendor_message") or "").strip()[:3000],
    }
    return result

# opcional para correr local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
