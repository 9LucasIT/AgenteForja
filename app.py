# app.py — Veglienzone WhatsApp Lead Agent (FastAPI)
import os, re, json
from typing import Optional, List, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import defaultdict

# =========================
# 1) PROMPT DEL ASESOR
# =========================
AGENT_PROMPT = """
Sos el ASESOR COMERCIAL de Inmobiliaria Veglienzone (Rosario). Hablás como humano:
cálido, claro y profesional. Una (1) pregunta por turno. No sos un bot, no uses frases robóticas ni tecnicismos.

OBJETIVO:
- Resolver la consulta y calificar al cliente para derivarlo a un asesor humano por WhatsApp.
- NO coordinás horarios ni agendas visitas.

SALUDO INICIAL (solo en el primer turno del chat o tras “reset”):
"Gracias por contactarte con el área comercial de Veglienzone Gestión Inmobiliaria. ¿Cómo podemos ayudarte hoy?
1- Alquileres
2- Ventas
3- Tasaciones"

ESCENARIO A — PROPIEDAD ESPECÍFICA (dirección, código o link):
- Confirmá la unidad y respondé preguntas típicas de ficha (zona, tipo, precio, dormitorios, cochera, amenities, expensas si aplica).
- Si el cliente está interesado (“me sirve”, “¿podemos avanzar?”, “pasame contacto”, “quiero ver”), ofrecé derivarlo a un asesor humano. Si acepta → vendor_push=true.

ESCENARIO B — BÚSQUEDA GENERAL:
- Calificá con UNA pregunta por turno, solo lo que falte:
  • Operación: alquiler o venta
  • Zona/barrio o dirección aproximada
  • Tipo: departamento, casa, PH, local, terreno…
  • Dormitorios/ambientes
  • Presupuesto (rango o tope)
  • Cochera y mascotas (si corresponde)
- Con datos suficientes, ofrecé 2–3 opciones sintéticas o pedí el dato faltante clave.

REGLAS DEL NEGOCIO:
- ALQUILER: preguntar (si falta) si tiene ingresos demostrables que TRIPLIQUEN el costo del alquiler, qué tipo de garantía usaría (seguro de caución Finaer u otra garantía propietaria), cuántos habitantes y si tiene mascotas.
- VENTA: no se aceptan m² como parte de pago ni vehículos (no considerar canjes de ese tipo). Aclaralo amablemente si lo proponen.
- Derivación: si el cliente confirma que quiere hablar con un asesor, seteá vendor_push=true y prepará vendor_message.
- Canal único: WhatsApp. No coordines disponibilidad horaria ni visites.

FORMATO DE SALIDA — devolvé SOLO este JSON:
{
  "reply_text": "texto humano para el cliente (una pregunta por turno, o respuesta a su consulta)",
  "closing_text": "mensaje final si ya se deriva a asesor (vacío si no aplica)",
  "vendor_push": true/false,
  "vendor_message": "resumen para el asesor humano: cliente (si lo dio), WhatsApp, operación, zona/dirección, tipo, dormitorios, presupuesto, cochera/mascotas; si es ALQUILER incluir ingresos x3 y tipo de garantía; en VENTA recordar que no se aceptan m² ni vehículos."
}

CRITERIOS vendor_push=true:
- El cliente pide contacto/visita/avanzar, o
- Ya aportó datos suficientes (≥3 de: operación, zona/dirección, tipo y [dormitorios o presupuesto]).
- Cuando vendor_push=true, pedile confirmación suave: “¿Querés que te contacte un asesor por este WhatsApp?” Si responde afirmativo, incluí closing_text con el aviso de derivación.

RECORDATORIO:
- Jamás escribir fuera del JSON. No menciones que sos IA.
- Si es el primer turno (o tras “reset”), usá el saludo inicial.
"""

# =========================
# 2) MEMORIA POR chatId
# =========================
memory = defaultdict(list)

def add_to_memory(chat_id: str, role: str, content: str):
    if not content:
        return
    memory[chat_id].append({"role": role, "content": content})
    if len(memory[chat_id]) > 8:
        memory[chat_id] = memory[chat_id][-8:]

def get_memory(chat_id: str) -> List[Dict]:
    return memory.get(chat_id, [])

def clear_memory(chat_id: str):
    memory.pop(chat_id, None)

# =========================
# 3) EXTRACCIÓN LIGERA DE CONTEXTO
# =========================
URL_RE = re.compile(r'https?://\S+', re.IGNORECASE)
CODE_RE = re.compile(r'\b([A-Za-z]\d{2,5})\b')                   # ej: A101, C244
ADDR_RE = re.compile(r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑa-záéíóúñ]+)*\s+\d{2,5})\b')

def extract_specific_refs(text: str) -> Dict[str,str]:
    out = {}
    urls = URL_RE.findall(text or "")
    if urls:
        out["link"] = urls[0]
    m_code = CODE_RE.search(text or "")
    if m_code:
        out["code"] = m_code.group(1)
    m_addr = ADDR_RE.search(text or "")
    if m_addr:
        out["address"] = m_addr.group(1)
    return out

def build_messages(chat_id: str, user_text: str, is_first_turn: bool) -> List[Dict]:
    msgs = []
    # memoria previa (sin duplicar prompts)
    for m in get_memory(chat_id):
        msgs.append(m)

    # prompt del asesor
    msgs.append({"role": "system", "content": AGENT_PROMPT})

    # hint si detectamos propiedad específica
    refs = extract_specific_refs(user_text)
    if refs:
        hint = "HINT: El cliente menciona una PROPIEDAD ESPECÍFICA. "
        if "link" in refs:    hint += f"Link: {refs['link']}. "
        if "code" in refs:    hint += f"Código: {refs['code']}. "
        if "address" in refs: hint += f"Dirección: {refs['address']}. "
        hint += "Respondé consultas de FICHA y, si se interesa, ofrecé derivación a un asesor humano."
        msgs.append({"role": "system", "content": hint})

    # marca de primer turno
    if is_first_turn:
        msgs.append({"role": "system", "content": "PRIMER_TURNO: usá el saludo inicial antes de calificar."})

    # entrada de usuario
    msgs.append({"role": "user", "content": user_text})
    return msgs

# =========================
# 4) LLM (OpenAI) con fallback mock
# =========================
_OPENAI_OK = False
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False

client = None
if _OPENAI_OK and os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "400"))

def call_llm(messages: List[Dict]) -> str:
    # real
    if client is not None:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return resp.choices[0].message.content or ""
    # mock (sin API key)
    return json.dumps({
        "reply_text": "Entendido. ¿La búsqueda es para alquiler o para venta, y en qué zona?",
        "closing_text": "",
        "vendor_push": False,
        "vendor_message": ""
    })

# =========================
# 5) FASTAPI
# =========================
class Inbound(BaseModel):
    user_phone: Optional[str] = ""
    text: str
    source: Optional[str] = "green"

app = FastAPI(title="Veglienzone Lead Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

def parse_agent_json(text: str) -> Dict:
    base = {"reply_text": "", "closing_text": "", "vendor_push": False, "vendor_message": ""}
    if not text:
        base["reply_text"] = "¿La búsqueda es para alquiler o para venta, y en qué zona?"
        return base
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start:end+1])
            base["reply_text"]    = (data.get("reply_text") or "").strip()
            base["closing_text"]  = (data.get("closing_text") or "").strip()
            base["vendor_push"]   = bool(data.get("vendor_push", False))
            base["vendor_message"]= (data.get("vendor_message") or "").strip()
            return base
        except Exception:
            pass
    base["reply_text"] = text.strip()
    return base

@app.post("/qualify")
def qualify(payload: Inbound):
    chat_id = payload.user_phone or "unknown"
    text_in = (payload.text or "").strip()

    # reset conversacional
    if text_in.lower() in {"reset", "reiniciar", "nuevo"}:
        clear_memory(chat_id)
        return {
            "reply_text": "Gracias por contactarte con el área comercial de Veglienzone Gestión Inmobiliaria. ¿Cómo podemos ayudarte hoy?\n1- Alquileres\n2- Ventas\n3- Tasaciones",
            "closing_text": "",
            "vendor_push": False,
            "vendor_message": ""
        }

    # ¿es primer turno?
    is_first = len(get_memory(chat_id)) == 0

    # construir mensajes (memoria + prompt + hint + user)
    messages = build_messages(chat_id, text_in, is_first)

    # llamar LLM y parsear
    raw = call_llm(messages)
    out = parse_agent_json(raw)

    # si el modelo decide vendor_push pero NO deja vendor_message, fabricamos uno claro
    if out.get("vendor_push") and not (out.get("vendor_message") or "").strip():
        out["vendor_message"] = f"LEAD CALIFICADO – Veglienzone | WhatsApp: +{payload.user_phone} | Contexto: {text_in[:200]}"

    # guardar memoria de turno
    add_to_memory(chat_id, "user", text_in)
    add_to_memory(chat_id, "assistant", out.get("reply_text", ""))

    # saneo final
    return {
        "reply_text": (out.get("reply_text") or "").strip()[:3000],
        "closing_text": (out.get("closing_text") or "").strip()[:2000],
        "vendor_push": bool(out.get("vendor_push", False)),
        "vendor_message": (out.get("vendor_message") or "").strip()[:3000],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
