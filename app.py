# app.py — Veglienzone WhatsApp Lead Agent (FastAPI + OpenAI + Green-API inbound)
import os, re, json, time
from typing import Optional, List, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import defaultdict

# =========================
# PROMPT DEL ASESOR
# =========================
AGENT_PROMPT = """
Sos el ASESOR COMERCIAL de Inmobiliaria Veglienzone (Rosario). Hablás como humano:
cálido, claro y profesional. Una (1) pregunta por turno. No uses frases robóticas.

OBJETIVO:
- Resolver la consulta y calificar al cliente para derivarlo a un asesor humano por WhatsApp.
- NO coordinás horarios ni agendas visitas.

SALUDO INICIAL (solo primer turno o tras “reset”):
"Gracias por contactarte con el área comercial de Veglienzone Gestión Inmobiliaria. ¿Cómo podemos ayudarte hoy?
1- Alquileres
2- Ventas
3- Tasaciones"

ESCENARIO A — PROPIEDAD ESPECÍFICA (dirección, código o link):
- Confirmá la unidad y respondé preguntas de ficha (zona, tipo, precio, dormitorios, cochera, amenities, expensas).
- Si el cliente se interesa, ofrecé derivarlo a un asesor humano. Si acepta → vendor_push=true.

ESCENARIO B — BÚSQUEDA GENERAL:
- Calificá con UNA pregunta por turno, solo lo que falte:
  • Operación: alquiler o venta
  • Zona/barrio o dirección aproximada
  • Tipo: departamento, casa, PH, local, terreno…
  • Dormitorios/ambientes
  • Presupuesto (rango o tope)
  • Cochera y mascotas (si corresponde)
- Con datos suficientes, ofrecé 2–3 opciones o pedí el dato faltante clave.

REGLAS DEL NEGOCIO:
- ALQUILER: preguntar (si falta) si tiene ingresos demostrables que TRIPLIQUEN el costo del alquiler, qué tipo de garantía usaría
  (seguro de caución Finaer u otra garantía propietaria), cuántos habitantes y si tiene mascotas.
- VENTA: no se aceptan m² ni vehículos como parte de pago (aclaralo amablemente si lo proponen).
- Derivación: si confirma que quiere que lo contacten, seteá vendor_push=true y prepará vendor_message.
- Canal único: WhatsApp. No coordines disponibilidad ni visitas.

POLÍTICA DE SALUDO:
- Solo mostrás el saludo inicial en el PRIMER TURNO o tras “reset”.
- En turnos siguientes NUNCA repitas el saludo inicial.

FORMATO (devolvé SOLO este JSON):
{
  "reply_text": "texto para el cliente",
  "closing_text": "mensaje final si se deriva (vacío si no aplica)",
  "vendor_push": true/false,
  "vendor_message": "resumen claro para el asesor: operación, zona/dirección, tipo, dormitorios, presupuesto, cochera/mascotas; en ALQUILER incluir ingresos x3 y tipo de garantía; en VENTA recordar que no se aceptan m² ni vehículos."
}

CRITERIOS vendor_push=true:
- El cliente pide contacto/visita/avanzar, o
- Aporta datos suficientes (≥3 de: operación, zona/dirección, tipo y [dormitorios o presupuesto]).
- Cuando vendor_push=true, preguntá: “¿Querés que te contacte un asesor por este WhatsApp?” Si responde que sí, incluí closing_text.

RECORDATORIO:
- No escribas nada fuera del JSON.
"""

# =========================
# MEMORIA Y GUARDAS
# =========================
memory = defaultdict(list)            # chatId -> [{"role":...,"content":...}]
last_msg_by_chat = {}                 # chatId -> last idMessage (evita duplicados de entrada)
last_ts_by_chat = {}                  # chatId -> last timestamp (rate-limit simple)
last_sent_by_chat = {}                # chatId -> {"text": str, "ts": float} (evita doble envío de salida)

def add_to_memory(chat_id: str, role: str, content: str):
    if not content:
        return
    memory[chat_id].append({"role": role, "content": content})
    if len(memory[chat_id]) > 10:
        memory[chat_id] = memory[chat_id][-10:]

def get_memory(chat_id: str) -> List[Dict]:
    return memory.get(chat_id, [])

def clear_memory(chat_id: str):
    memory.pop(chat_id, None)

# =========================
# DETECCIÓN DE REFERENCIAS
# =========================
URL_RE  = re.compile(r'https?://\S+', re.IGNORECASE)
CODE_RE = re.compile(r'\b([A-Za-z]\d{2,5})\b')  # ej: A101, C244
ADDR_RE = re.compile(r'\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑa-záéíóúñ]+)*\s+\d{2,5})\b')

def extract_specific_refs(text: str) -> Dict[str, str]:
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

def mentions_specific_without_data(text: str) -> bool:
    t = (text or "").lower()
    has_hint = any(kw in t for kw in [
        "tengo la dirección", "tengo direccion", "tengo la dire",
        "tengo el link", "tengo el enlace", "tengo ubicación", "tengo la ubicación",
        "tengo el codigo", "tengo el código"
    ])
    has_ref = bool(URL_RE.search(text or "") or CODE_RE.search(text or "") or ADDR_RE.search(text or ""))
    return has_hint and not has_ref

# =========================
# CONTROL DE SALUDO REPETIDO + ANTI DOBLE ENVÍO
# =========================
GREETING_SNIPPET = "Gracias por contactarte con el área comercial de Veglienzone"

def clean_greeting(reply_text: str, is_first_turn: bool) -> str:
    if is_first_turn or not reply_text:
        return reply_text
    text = reply_text.strip()
    if GREETING_SNIPPET.lower() in text.lower():
        lines = [l for l in text.splitlines()]
        i = 0
        while i < len(lines) and lines[i].strip():
            i += 1
        rest = "\n".join(lines[i:]).strip()
        return rest or "Perfecto, contame un poco más así te ayudo."
    return reply_text

import time as _time
def _should_send(chat_id: str, text: str, window_sec: float = 6.0) -> bool:
    """
    Evita mandar dos veces el mismo texto al mismo chat en una ventana corta.
    Retorna True si se debe enviar; False si se considera duplicado inmediato.
    """
    if not chat_id or not text:
        return False
    rec = last_sent_by_chat.get(chat_id, {"text": None, "ts": 0.0})
    if rec["text"] == text and (_time.time() - rec["ts"]) < window_sec:
        return False
    last_sent_by_chat[chat_id] = {"text": text, "ts": _time.time()}
    return True

# =========================
# ARMADO DE MENSAJES PARA IA
# =========================
def build_messages(chat_id: str, user_text: str, is_first_turn: bool) -> List[Dict]:
    msgs: List[Dict] = []
    for m in get_memory(chat_id):
        msgs.append(m)
    msgs.append({"role": "system", "content": AGENT_PROMPT})

    refs = extract_specific_refs(user_text)
    if refs:
        hint = "HINT: El cliente menciona una PROPIEDAD ESPECÍFICA. "
        if "link" in refs:    hint += f"Link: {refs['link']}. "
        if "code" in refs:    hint += f"Código: {refs['code']}. "
        if "address" in refs: hint += f"Dirección: {refs['address']}. "
        hint += "Respondé como ficha y, si hay interés, ofrecé derivación a un asesor humano."
        msgs.append({"role": "system", "content": hint})

    if is_first_turn:
        msgs.append({"role": "system", "content": "PRIMER_TURNO: usá el saludo inicial."})
    else:
        msgs.append({"role": "system", "content": "No repitas el saludo inicial. Continuá la conversación directamente."})

    msgs.append({"role": "user", "content": user_text})
    return msgs

# =========================
# LLM (OpenAI) con fallback
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

MODEL        = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE", "0.4"))
MAX_TOKENS   = int(os.getenv("LLM_MAX_TOKENS", "400"))

def call_llm(messages: List[Dict]) -> str:
    if client is not None:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return resp.choices[0].message.content or ""
    # fallback (si falta API key)
    return json.dumps({
        "reply_text": "¿La búsqueda es para alquiler o para venta, y en qué zona?",
        "closing_text": "",
        "vendor_push": False,
        "vendor_message": ""
    })

def parse_agent_json(text: str) -> Dict:
    base = {"reply_text": "", "closing_text": "", "vendor_push": False, "vendor_message": ""}
    if not text:
        base["reply_text"] = "¿La búsqueda es para alquiler o para venta, y en qué zona?"
        return base
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start:end+1])
            base["reply_text"]     = (data.get("reply_text") or "").strip()
            base["closing_text"]   = (data.get("closing_text") or "").strip()
            base["vendor_push"]    = bool(data.get("vendor_push", False))
            base["vendor_message"] = (data.get("vendor_message") or "").strip()
            return base
        except Exception:
            pass
    base["reply_text"] = text.strip()
    return base

# =========================
# FASTAPI
# =========================
class Inbound(BaseModel):
    user_phone: Optional[str] = ""
    text: str
    source: Optional[str] = "green"

class GreenInbound(BaseModel):
    typeWebhook: Optional[str] = ""
    instanceData: Optional[dict] = None
    timestamp: Optional[int] = None
    idMessage: Optional[str] = ""
    senderData: Optional[dict] = None
    messageData: Optional[dict] = None

app = FastAPI(title="Veglienzone Lead Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug")
def debug():
    mode = "openai" if os.getenv("OPENAI_API_KEY") else "mock"
    return {
        "llm_mode": mode,
        "memory_sessions": len(memory),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", ""),
        "GREEN_ID": os.getenv("GREEN_ID", ""),
        "GREEN_TOKEN": "***" if os.getenv("GREEN_TOKEN") else "",
        "N8N_VENDOR_WEBHOOK": "***" if os.getenv("N8N_VENDOR_WEBHOOK") else "",
    }

# -------------------------
# /qualify (core conversacional)
# -------------------------
@app.post("/qualify")
def qualify(payload: Inbound):
    chat_id  = payload.user_phone or "unknown"
    text_in  = (payload.text or "").strip()

    # reset conversacional
    if text_in.lower() in {"reset", "reiniciar", "nuevo"}:
        clear_memory(chat_id)
        return {
            "reply_text": (
                "Gracias por contactarte con el área comercial de Veglienzone Gestión Inmobiliaria. "
                "¿Cómo podemos ayudarte hoy?\n"
                "1- Alquileres\n"
                "2- Ventas\n"
                "3- Tasaciones\n\n"
                "Nota: si en cualquier momento escribís *reset*, la conversación se reinicia desde cero."
            ),
            "closing_text": "",
            "vendor_push": False,
            "vendor_message": ""
        }

    # atajo: “tengo la dirección/link/código” pero aún no lo pasó
    if mentions_specific_without_data(text_in):
        reply = "¡Genial! Pasame la dirección exacta, el link de la publicación o el código, así te confirmo los detalles."
        add_to_memory(chat_id, "user", text_in)
        add_to_memory(chat_id, "assistant", reply)
        return {
            "reply_text": reply,
            "closing_text": "",
            "vendor_push": False,
            "vendor_message": ""
        }

    is_first = len(get_memory(chat_id)) == 0
    messages = build_messages(chat_id, text_in, is_first)
    raw = call_llm(messages)
    out = parse_agent_json(raw)

    # anti-saludo repetido
    out["reply_text"] = clean_greeting(out.get("reply_text", ""), is_first)
    if is_first and GREETING_SNIPPET.lower() in out["reply_text"].lower():
        out["reply_text"] += "\n\nNota: si en cualquier momento escribís *reset*, la conversación se reinicia desde cero."

    if out.get("vendor_push") and not (out.get("vendor_message") or "").strip():
        out["vendor_message"] = f"LEAD CALIFICADO – Veglienzone | WhatsApp +{payload.user_phone} | Contexto: {text_in[:200]}"

    add_to_memory(chat_id, "user", text_in)
    add_to_memory(chat_id, "assistant", out.get("reply_text", ""))

    return {
        "reply_text": (out.get("reply_text") or "").strip()[:3000],
        "closing_text": (out.get("closing_text") or "").strip()[:2000],
        "vendor_push": bool(out.get("vendor_push", False)),
        "vendor_message": (out.get("vendor_message") or "").strip()[:3000],
    }

# -------------------------
# Webhook directo desde Green-API
# -------------------------
@app.post("/api/green/inbound")
def green_inbound(ev: GreenInbound):
    # Solo procesamos mensajes ENTRANTES del cliente
    if (ev.typeWebhook or "").lower() != "incomingmessagereceived":
        return {"ok": True}

    chat_id   = (ev.senderData or {}).get("chatId", "")          # ej: 549xxx@c.us
    user_phone = chat_id.replace("@c.us", "") if chat_id else ""
    text      = ""
    md = ev.messageData or {}
    if (md.get("typeMessage") == "textMessage") and md.get("textMessageData"):
        text = (md["textMessageData"].get("textMessage") or "").strip()

    # --------- guardas anti duplicado / rate limit (de entrada) ---------
    if ev.idMessage and chat_id:
        if last_msg_by_chat.get(chat_id) == ev.idMessage:
            return {"ok": True}
        last_msg_by_chat[chat_id] = ev.idMessage
    now = time.time()
    last_ts = last_ts_by_chat.get(chat_id, 0)
    if now - last_ts < 2.0:
        return {"ok": True}
    last_ts_by_chat[chat_id] = now
    # -------------------------------------------------------------------

    result = qualify(Inbound(user_phone=user_phone, text=text))

    # Responder al cliente por Green (con anti-doble-envío de salida)
    import requests
    idInstance = os.getenv("GREEN_ID")
    apiToken   = os.getenv("GREEN_TOKEN")
    if idInstance and apiToken and chat_id:
        url = f"https://api.green-api.com/waInstance{idInstance}/sendMessage/{apiToken}"

        reply_text = (result.get("reply_text") or "").strip()
        if reply_text and _should_send(chat_id, reply_text):
            try:
                requests.post(url, json={"chatId": chat_id, "message": reply_text}, timeout=10)
            except Exception:
                pass

        closing_text = (result.get("closing_text") or "").strip()
        if closing_text and _should_send(chat_id, closing_text):
            try:
                requests.post(url, json={"chatId": chat_id, "message": closing_text}, timeout=10)
            except Exception:
                pass

    # Si hay handoff al vendedor, avisamos a n8n (solo aquí)
    if result.get("vendor_push"):
        n8n_url = os.getenv("N8N_VENDOR_WEBHOOK")
        if n8n_url:
            try:
                import requests
                requests.post(n8n_url, json={
                    "vendor_message": result["vendor_message"],
                    "client_phone": user_phone
                }, timeout=10)
            except Exception:
                pass

    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
