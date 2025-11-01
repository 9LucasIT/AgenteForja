# app.py — Veglienzone WhatsApp Lead Agent (FastAPI + OpenAI + Green-API + DB Railway)
import os, re, json, time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import defaultdict

# ======= IA =======
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

# ======= BD (MySQL) =======
# Si usas Postgres, ver instrucciones al final.
import pymysql

DB_CFG = dict(
    host=os.getenv("DB_HOST", ""),
    port=int(os.getenv("DB_PORT", "3306")),
    user=os.getenv("DB_USER", ""),
    password=os.getenv("DB_PASSWORD", ""),
    database=os.getenv("DB_NAME", ""),
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True,
)

def db_conn():
    return pymysql.connect(**DB_CFG)

# --- Suposición de vista/tabla ---
# Usamos una vista (o SELECT) que devuelva SIEMPRE estas columnas:
#   code (str), address (str), zone (str), operation (alquiler|venta),
#   price (numeric), bedrooms (int), parking (bool/int), pets (bool/int), summary (str)
# Si tu esquema es distinto, podés crear una vista:
#   CREATE OR REPLACE VIEW view_properties AS
#   SELECT codigo AS code, direccion AS address, zona AS zone, operacion AS operation,
#          precio AS price, dormitorios AS bedrooms, cochera AS parking,
#          mascotas AS pets,
#          CONCAT(tipo,' – ',descripcion) AS summary
#   FROM propiedades;
#
# Reemplazá "view_properties" por tu tabla/vista real abajo si fuera necesario.
PROPERTIES_TABLE = "view_properties"

# ======= Memoria / Guardia =======
memory = defaultdict(list)            # chatId -> [{"role":...,"content":...}]
last_msg_by_chat = {}                 # evita duplicados de entrada
last_ts_by_chat = {}                  # rate limit entrada
last_sent_by_chat = {}                # evita doble envío de salida
state = defaultdict(dict)             # chatId -> {"operation":..., "zone":..., "selected_code":...}

GREETING_SNIPPET = "Gracias por contactarte con el área comercial de Veglienzone"

URL_RE  = re.compile(r'https?://\S+', re.IGNORECASE)
CODE_RE = re.compile(r'\b([A-Za-z]\d{2,5})\b')                      # ej: A101
ADDR_RE = re.compile(r'\b([A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ\.]+(?:\s+[A-ZÁÉÍÓÚÑ\wáéíóúñ\.]+)*\s+\d{2,6})\b')

# ======= Prompts =======
AGENT_PROMPT = """
Sos el ASESOR COMERCIAL de Inmobiliaria Veglienzone (Rosario). Hablás humano, cálido y profesional. 
UNA (1) pregunta por turno. No uses frases robóticas.

FLUJO:
1) Saludo inicial (solo primer turno o tras "reset"):
"Gracias por contactarte con el área comercial de Veglienzone Gestión Inmobiliaria. ¿Cómo podemos ayudarte hoy?
1- Alquileres
2- Ventas
3- Tasaciones
Nota: si en cualquier momento escribís reset, la conversación se reinicia desde cero."

2) Inmediatamente después del saludo, preguntá:
"¿Tenés una dirección o link exacto, o estás buscando por alguna zona en particular?"

3) Si hay PROPIEDAD ESPECÍFICA (dirección, código o link):
   - Respondé dudas de ficha: zona, tipo, precio, dormitorios, cochera, amenities, expensas.
   - Si muestra interés, ofrecé derivación a un asesor humano.

4) Si es BÚSQUEDA GENERAL:
   - ALQUILER: Calificá con UNA pregunta por turno (solo lo que falte):
     • zona/barrio, tipo (depto, casa, PH...), dormitorios, presupuesto, cochera y mascotas
     • Requisitos: ingresos demostrables que TRIPLIQUEN el alquiler y tipo de garantía (Finaer u otra propietaria)
     • Aclaración: no coordines horarios/visitas, solo preparás el pase
   - VENTA: Calificá y aclarás que NO se aceptan m² ni vehículos como parte de pago.

5) Cuando haya datos suficientes o pida avanzar: ofrecé derivación por WhatsApp al asesor humano.
   Si acepta → vendor_push=true y prepará vendor_message (resumen claro).

POLÍTICAS:
- No repitas el saludo fuera del primer turno.
- Si recibís un listado de opciones del sistema (LISTINGS), mostralas prolijo y preguntá si quiere más datos o ajustar.
- Formato de respuesta SOLO en JSON:
{
  "reply_text": "...",
  "closing_text": "...",
  "vendor_push": true/false,
  "vendor_message": "..."
}
"""

# ======= Utilidades Memoria/Estado =======
def add_to_memory(chat_id: str, role: str, content: str):
    if not content:
        return
    memory[chat_id].append({"role": role, "content": content})
    if len(memory[chat_id]) > 12:
        memory[chat_id] = memory[chat_id][-12:]

def get_memory(chat_id: str) -> List[Dict]:
    return memory.get(chat_id, [])

def clear_memory(chat_id: str):
    memory.pop(chat_id, None)
    state.pop(chat_id, None)

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
    if not chat_id or not text:
        return False
    rec = last_sent_by_chat.get(chat_id, {"text": None, "ts": 0.0})
    if rec["text"] == text and (_time.time() - rec["ts"]) < window_sec:
        return False
    last_sent_by_chat[chat_id] = {"text": text, "ts": _time.time()}
    return True

# ======= Detección simple (intenciones / zona / referencias) =======
ALQ_WORDS  = ["alquiler", "alquilar", "rentar", "alquilo", "busco alquiler", "para alquilar"]
VENTA_WORDS= ["venta", "comprar", "compro", "para comprar", "quiero comprar", "vendo"]
TASA_WORDS = ["tasacion", "tasación", "tasar"]

def detect_operation(text: str) -> Optional[str]:
    t = (text or "").lower()
    if any(w in t for w in ALQ_WORDS):  return "alquiler"
    if any(w in t for w in VENTA_WORDS): return "venta"
    if any(w in t for w in TASA_WORDS):  return "tasaciones"
    return None

def extract_zone_hint(text: str) -> Optional[str]:
    # heurística muy simple: "zona X", "en X", "por X"
    t = (text or "").lower()
    m = re.search(r"zona\s+([a-záéíóúñ\s]+)", t)
    if m: return m.group(1).strip().title()[:40]
    m = re.search(r"en\s+([a-záéíóúñ]{3,}(?:\s+[a-záéíóúñ]{3,})?)", t)
    if m: return m.group(1).strip().title()[:40]
    return None

def extract_specific_refs(text: str) -> Dict[str, str]:
    out = {}
    urls = URL_RE.findall(text or "")
    if urls: out["link"] = urls[0]
    m_code = CODE_RE.search(text or "")
    if m_code: out["code"] = m_code.group(1)
    m_addr = ADDR_RE.search(text or "")
    if m_addr: out["address"] = m_addr.group(1)
    return out

def mentions_specific_without_data(text: str) -> bool:
    t = (text or "").lower()
    has_hint = any(kw in t for kw in [
        "tengo la dirección", "tengo direccion", "tengo el link", "tengo enlace",
        "tengo el codigo", "tengo el código"
    ])
    has_ref = bool(URL_RE.search(text or "") or CODE_RE.search(text or "") or ADDR_RE.search(text or ""))
    return has_hint and not has_ref

# ======= BD: búsquedas =======
def listings_for(operation: str, zone: str, limit: int = 3) -> List[Dict[str, Any]]:
    if not operation or not zone:
        return []
    q = f"""
        SELECT code, address, zone, operation, price, bedrooms, parking, pets, summary
        FROM {PROPERTIES_TABLE}
        WHERE operation = %s
          AND zone LIKE %s
        ORDER BY price ASC
        LIMIT %s
    """
    try:
        with db_conn() as cnn, cnn.cursor() as cur:
            cur.execute(q, (operation.lower(), f"%{zone}%", limit))
            rows = cur.fetchall() or []
            return rows
    except Exception:
        return []

def find_property_by_ref(code: Optional[str]=None, address_like: Optional[str]=None) -> Optional[Dict[str, Any]]:
    if not code and not address_like:
        return None
    if code:
        q = f"SELECT * FROM {PROPERTIES_TABLE} WHERE code=%s LIMIT 1"
        args = (code,)
    else:
        q = f"SELECT * FROM {PROPERTIES_TABLE} WHERE address LIKE %s LIMIT 1"
        args = (f"%{address_like}%",)
    try:
        with db_conn() as cnn, cnn.cursor() as cur:
            cur.execute(q, args)
            row = cur.fetchone()
            return row
    except Exception:
        return None

def listings_to_system_hint(listings: List[Dict[str, Any]]) -> str:
    if not listings:
        return ""
    lines = ["LISTINGS: Estas son opciones para mostrar al usuario (no inventes otras):"]
    for r in listings:
        line = f"• {r.get('code','')} – {r.get('address','')} – ${r.get('price','')} – {r.get('bedrooms','?')} dorm"
        if r.get("parking"):
            line += " – cochera"
        lines.append(line)
    lines.append("Preguntá si quiere más info de alguna, o si desea ajustar la zona o el presupuesto.")
    return "\n".join(lines)

# ======= LLM calls =======
def call_llm(messages: List[Dict]) -> str:
    if client is not None:
        resp = client.chat.completions.create(
            model=MODEL, messages=messages,
            temperature=TEMPERATURE, max_tokens=MAX_TOKENS
        )
        return resp.choices[0].message.content or ""
    # fallback si no hay API
    return json.dumps({
        "reply_text": "¿La búsqueda es para alquiler o venta y en qué zona?",
        "closing_text": "",
        "vendor_push": False,
        "vendor_message": ""
    })

def parse_agent_json(text: str) -> Dict:
    base = {"reply_text": "", "closing_text": "", "vendor_push": False, "vendor_message": ""}
    if not text:
        base["reply_text"] = "¿La búsqueda es para alquiler o venta y en qué zona?"
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

# ======= Mensajes a la IA =======
def build_messages(chat_id: str, user_text: str, is_first_turn: bool) -> List[Dict]:
    msgs: List[Dict] = []
    for m in get_memory(chat_id):
        msgs.append(m)
    msgs.append({"role": "system", "content": AGENT_PROMPT})

    # Pistas de estado recogidas
    st = state.get(chat_id, {})
    op = st.get("operation")
    zone = st.get("zone")

    # Si detectamos propiedad exacta, informar a la IA
    refs = extract_specific_refs(user_text)
    if refs:
        hint = "HINT: El usuario menciona PROPIEDAD ESPECÍFICA. "
        if "link" in refs:    hint += f"Link: {refs['link']}. "
        if "code" in refs:    hint += f"Código: {refs['code']}. "
        if "address" in refs: hint += f"Dirección: {refs['address']}. "
        msgs.append({"role": "system", "content": hint})

    # Si tenemos ALQUILER + ZONA (sin dirección ni link) → cargar listados desde BD
    if op == "alquiler" and zone and not refs:
        lst = listings_for("alquiler", zone, limit=3)
        sys_hint = listings_to_system_hint(lst)
        if sys_hint:
            msgs.append({"role": "system", "content": sys_hint})

    # primer turno o no
    if is_first_turn:
        msgs.append({"role": "system", "content": "PRIMER_TURNO: usá el saludo inicial y preguntá dirección/link o zona."})
    else:
        msgs.append({"role": "system", "content": "No repitas el saludo inicial. Continuá la conversación."})

    msgs.append({"role": "user", "content": user_text})
    return msgs

# ======= FastAPI =======
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
        "DB_HOST": DB_CFG["host"],
        "DB_NAME": DB_CFG["database"],
        "table": PROPERTIES_TABLE
    }

# ---------- Core conversacional ----------
@app.post("/qualify")
def qualify(payload: Inbound):
    chat_id  = payload.user_phone or "unknown"
    text_in  = (payload.text or "").strip()

    # reset
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

    # actualizar estado básico (operación / zona) por heurística
    op = detect_operation(text_in)
    if op:
        state[chat_id]["operation"] = op
    zn = extract_zone_hint(text_in)
    if zn:
        state[chat_id]["zone"] = zn

    # caso "tengo la dirección/link/código" sin datos
    if mentions_specific_without_data(text_in):
        reply = "¡Genial! Pasame la dirección exacta, el link de la publicación o el código, así te confirmo los detalles."
        add_to_memory(chat_id, "user", text_in)
        add_to_memory(chat_id, "assistant", reply)
        return {"reply_text": reply, "closing_text": "", "vendor_push": False, "vendor_message": ""}

    # property exacta: guardo ref en estado (por si la IA la usa)
    refs = extract_specific_refs(text_in)
    if refs.get("code"):
        state[chat_id]["selected_code"] = refs["code"]

    # primer turno?
    is_first = len(get_memory(chat_id)) == 0
    messages = build_messages(chat_id, text_in, is_first)
    raw = call_llm(messages)
    out = parse_agent_json(raw)

    # anti-saludo repetido
    out["reply_text"] = clean_greeting(out.get("reply_text", ""), is_first)
    if is_first and GREETING_SNIPPET.lower() in out["reply_text"].lower():
        out["reply_text"] += "\n\nNota: si en cualquier momento escribís *reset*, la conversación se reinicia desde cero."

    # vendor msg por defecto si hace push
    if out.get("vendor_push") and not (out.get("vendor_message") or "").strip():
        resum = []
        st = state.get(chat_id, {})
        if st.get("operation"): resum.append(f"Operación: {st['operation']}")
        if st.get("zone"):      resum.append(f"Zona: {st['zone']}")
        if refs.get("code"):    resum.append(f"Código: {refs['code']}")
        vendor_message = " | ".join(resum) or "LEAD CALIFICADO – Veglienzone"
        out["vendor_message"] = vendor_message

    add_to_memory(chat_id, "user", text_in)
    add_to_memory(chat_id, "assistant", out.get("reply_text", ""))

    return {
        "reply_text": (out.get("reply_text") or "").strip()[:3000],
        "closing_text": (out.get("closing_text") or "").strip()[:2000],
        "vendor_push": bool(out.get("vendor_push", False)),
        "vendor_message": (out.get("vendor_message") or "").strip()[:3000],
    }

# ---------- Webhook Green-API ----------
@app.post("/api/green/inbound")
def green_inbound(ev: GreenInbound):
    if (ev.typeWebhook or "").lower() != "incomingmessagereceived":
        return {"ok": True}

    chat_id   = (ev.senderData or {}).get("chatId", "")
    user_phone = chat_id.replace("@c.us", "") if chat_id else ""
    text      = ""
    md = ev.messageData or {}
    if (md.get("typeMessage") == "textMessage") and md.get("textMessageData"):
        text = (md["textMessageData"].get("textMessage") or "").strip()

    # anti duplicado de entrada
    if ev.idMessage and chat_id:
        if last_msg_by_chat.get(chat_id) == ev.idMessage:
            return {"ok": True}
        last_msg_by_chat[chat_id] = ev.idMessage
    now = time.time()
    if now - last_ts_by_chat.get(chat_id, 0) < 2.0:
        return {"ok": True}
    last_ts_by_chat[chat_id] = now

    result = qualify(Inbound(user_phone=user_phone, text=text))

    # responder por Green (anti doble salida)
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

    # push al vendedor (n8n)
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
