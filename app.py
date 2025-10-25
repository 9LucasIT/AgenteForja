import os
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# ====== ENV ======
DATABASE_URL = os.getenv("DATABASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL no está configurada.")

# Acepta tanto mysql:// como mysql+pymysql://; forzamos el driver pymysql
if DATABASE_URL.startswith("mysql://"):
    DATABASE_URL = DATABASE_URL.replace("mysql://", "mysql+pymysql://", 1)

# ====== DB ENGINE ======
engine: Engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    isolation_level="AUTOCOMMIT",
)

# ====== OPENAI ======
from openai import OpenAI
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY no está configurada.")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ====== FASTAPI ======
app = FastAPI(title="RealEstate AI Agent", version="1.0.0")


# ---------- Utils ----------
def normalize_phone(p: str) -> str:
    digits = re.sub(r"\D", "", str(p))
    return digits

def to_ars_number(txt: Optional[str]) -> Optional[int]:
    if not txt:
        return None
    t = str(txt).lower()
    t = t.replace("ars", "").replace("$", "").replace(" ", "")
    t = t.replace(".", "").replace(",", "")
    # 80k / 80K
    if re.search(r"\d+k$", t):
        base = re.sub(r"[^\d]", "", t[:-1]) or "0"
        return int(base) * 1000
    # 120mil, 120k
    if "mil" in t:
        base = re.sub(r"[^\d]", "", t) or "0"
        return int(base) * 1000
    digits = re.sub(r"[^\d]", "", t)
    return int(digits) if digits else None

def safe_bool_yesno(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("si", "sí", "true", "con", "yes", "y", "1"):
        return "sí"
    if s in ("no", "false", "sin", "n", "0"):
        return "no"
    return None


# ---------- Prompts ----------
AGENT_SYSTEM_PROMPT = """
Sos una asesora inmobiliaria argentina, cálida y profesional. Tratá de “vos”.
Tu objetivo es calificar al contacto con trato humano y natural.

Slots a completar:
- zona (barrio/dirección aproximada) → slots.zona
- presupuesto_min (ARS) → slots.presupuesto_min
- presupuesto_max (ARS) → slots.presupuesto_max
- dormitorios (entero) → slots.dormitorios
- cochera (sí/no) → slots.cochera

Reglas:
- UNA sola pregunta por mensaje.
- Antes de preguntar, confirmá en 1 línea lo que entendiste (“Perfecto, zona Abasto. …”).
- No repreguntes lo ya dicho; si dicen “ya te dije”, revisá lo que hay y seguí con lo que falte.
- Si ya hay mínimo, pedí máximo; si hay máximo, pedí mínimo.
- Aceptá formatos: 80k, 100.000, $120 mil → números.
- Tono cercano/humano, sin listas, sin paréntesis, máx. 2 líneas.

Cierre (vendor_push=true):
- Cuando tengas: zona + presupuesto_min + presupuesto_max.
- Si hay dormitorios/cochera, mejor; si no, igual podés cerrar.

Devolvé SIEMPRE JSON:
{
  "text": "<mensaje de 1–2 líneas>",
  "next_question": "<pregunta próxima o null>",
  "vendor_push": <true|false>,
  "updates": {
    "slots": {
      "zona": <string|null>,
      "presupuesto_min": <number|string|null>,
      "presupuesto_max": <number|string|null>,
      "dormitorios": <number|string|null>,
      "cochera": <"sí"|"no"|string|null>,
      "conversation": <array de turnos {role, content}>
    }
  }
}
Nada fuera de ese JSON.
"""

FEWSHOT = [
    {
        "role": "user",
        "content": "Hola"
    },
    {
        "role": "assistant",
        "content": '{"text":"¡Hola! ¿Cómo estás? Contame en qué zona te gustaría buscar.","next_question":"¿En qué zona o barrio te gustaría?","vendor_push":false,"updates":{"slots":{"zona":null,"presupuesto_min":null,"presupuesto_max":null,"dormitorios":null,"cochera":null,"conversation":[{"role":"user","content":"Hola"},{"role":"assistant","content":"¡Hola! ¿Cómo estás? Contame en qué zona te gustaría buscar."}]}}}'
    },
    {
        "role": "user",
        "content": "En Abasto"
    },
    {
        "role": "assistant",
        "content": '{"text":"Perfecto, zona Abasto. Para ubicarnos mejor necesito un rango de presupuesto.","next_question":"¿Cuál sería tu presupuesto mínimo aproximado (en ARS)?","vendor_push":false,"updates":{"slots":{"zona":"Abasto","presupuesto_min":null,"presupuesto_max":null,"dormitorios":null,"cochera":null,"conversation":[{"role":"user","content":"En Abasto"}]}}}'
    },
    {
        "role": "user",
        "content": "80k"
    },
    {
        "role": "assistant",
        "content": '{"text":"Genial, tomo $80.000 como mínimo. Para cerrar el rango necesito el máximo.","next_question":"¿Y el presupuesto máximo (en ARS)?","vendor_push":false,"updates":{"slots":{"zona":"Abasto","presupuesto_min":80000,"presupuesto_max":null,"dormitorios":null,"cochera":null,"conversation":[{"role":"user","content":"80k"}]}}}'
    },
    {
        "role": "user",
        "content": "100000"
    },
    {
        "role": "assistant",
        "content": '{"text":"Perfecto, rango $80.000–$100.000 en Abasto. Con eso ya puedo pasarte con el equipo.","next_question":null,"vendor_push":true,"updates":{"slots":{"zona":"Abasto","presupuesto_min":80000,"presupuesto_max":100000,"dormitorios":null,"cochera":null,"conversation":[{"role":"user","content":"100000"}]}}}'
    }
]


# ---------- DB helpers ----------
def db_fetchone(sql: str, params: dict) -> Optional[Dict[str, Any]]:
    with engine.connect() as conn:
        res = conn.execute(text(sql), params)
        row = res.mappings().first()
        return dict(row) if row else None

def db_fetchall(sql: str, params: dict) -> List[Dict[str, Any]]:
    with engine.connect() as conn:
        res = conn.execute(text(sql), params)
        rows = res.mappings().all()
        return [dict(r) for r in rows]

def db_execute(sql: str, params: dict) -> None:
    with engine.connect() as conn:
        conn.execute(text(sql), params)


# ---------- Session & Leads ----------
def get_or_create_session(user_phone: str) -> Dict[str, Any]:
    s = db_fetchone(
        "SELECT * FROM chat_session WHERE user_phone=:p LIMIT 1",
        {"p": user_phone}
    )
    if s:
        return s
    slots = {}
    db_execute(
        """
        INSERT INTO chat_session (user_phone, last_message_id, last_welcome_at, cooldown_until, status, slots_json, created_at, updated_at)
        VALUES (:p, NULL, NULL, NULL, 'active', :slots, NOW(), NOW())
        """,
        {"p": user_phone, "slots": json.dumps(slots)},
    )
    return db_fetchone(
        "SELECT * FROM chat_session WHERE user_phone=:p LIMIT 1",
        {"p": user_phone}
    )

def update_session(user_phone: str, slots: Dict[str, Any]) -> None:
    db_execute(
        """
        UPDATE chat_session
        SET slots_json=:slots, updated_at=NOW()
        WHERE user_phone=:p
        """,
        {"p": user_phone, "slots": json.dumps(slots)},
    )

def ensure_lead_row(user_phone: str, vendor_phone: str) -> None:
    row = db_fetchone(
        "SELECT id FROM leads WHERE user_phone=:p LIMIT 1",
        {"p": user_phone}
    )
    if row:
        return
    db_execute(
        """
        INSERT INTO leads (user_phone, status, vendor_phone, created_at, updated_at)
        VALUES (:p, 'pendiente', :v, NOW(), NOW())
        """,
        {"p": user_phone, "v": vendor_phone},
    )

def update_lead_from_slots(user_phone: str, slots: Dict[str, Any]) -> None:
    # Campos compatibles con tu tabla leads
    sql = """
    UPDATE leads SET
      inmueble_interes = COALESCE(:zona, inmueble_interes),
      dormitorios = COALESCE(:dormitorios, dormitorios),
      cochera = COALESCE(:cochera, cochera),
      presupuesto_min = COALESCE(:pmin, presupuesto_min),
      presupuesto_max = COALESCE(:pmax, presupuesto_max),
      updated_at=NOW()
    WHERE user_phone=:p
    """
    db_execute(sql, {
        "zona": slots.get("zona"),
        "dormitorios": slots.get("dormitorios"),
        "cochera": slots.get("cochera"),
        "pmin": slots.get("presupuesto_min"),
        "pmax": slots.get("presupuesto_max"),
        "p": user_phone
    })


# ---------- Matching de propiedades ----------
def suggest_properties(slots: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    zona = (slots.get("zona") or "").strip()
    pmin = slots.get("presupuesto_min") or 0
    pmax = slots.get("presupuesto_max") or 10**12
    dorm = slots.get("dormitorios")
    coch = slots.get("cochera")

    where = ["precio BETWEEN :pmin AND :pmax"]
    params = {"pmin": int(pmin), "pmax": int(pmax)}

    if zona:
        where.append("zona LIKE :zona")
        params["zona"] = f"%{zona}%"
    if dorm is not None:
        where.append("dormitorios = :dorm")
        params["dorm"] = int(dorm)
    if coch in ("sí", "no"):
        # en tu tabla: 1/0
        where.append("cochera = :coch")
        params["coch"] = 1 if coch == "sí" else 0

    sql = f"""
    SELECT codigo, direccion, zona, precio, dormitorios, cochera
    FROM propiedades
    WHERE {" AND ".join(where)}
    ORDER BY precio ASC
    LIMIT {limit}
    """
    return db_fetchall(sql, params)


# ---------- LLM ----------
def build_messages(slots: dict, conversation: list, user_text: str):
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    messages += FEWSHOT

    state_summary = {
        "slots": {
            "zona": slots.get("zona"),
            "presupuesto_min": slots.get("presupuesto_min"),
            "presupuesto_max": slots.get("presupuesto_max"),
            "dormitorios": slots.get("dormitorios"),
            "cochera": slots.get("cochera"),
        },
        "conversation": conversation[-10:],
    }
    messages.append({"role": "user", "content": f"ESTADO_ACTUAL_JSON={state_summary}"})
    messages.append({"role": "user", "content": user_text})
    return messages

def ask_llm(messages: list) -> dict:
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.6,
        messages=messages
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        # En caso de formato raro, pedimos un fallback ultra estricto
        fallback = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "Devuelve exclusivamente JSON válido. Sin texto adicional."},
                {"role": "user", "content": f"Reformatea este contenido a JSON válido: {content}"}
            ]
        )
        data = json.loads(fallback.choices[0].message.content)
    return data


# ---------- Schemas ----------
class QualifyIn(BaseModel):
    user_phone: str
    message_id: Optional[str] = None
    text: str

class QualifyOut(BaseModel):
    text: str
    next_question: Optional[str]
    vendor_push: bool
    updates: Dict[str, Any]
    suggestions: Optional[List[Dict[str, Any]]] = None


# ---------- API ----------
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/qualify", response_model=QualifyOut)
def qualify(payload: QualifyIn = Body(...)):
    user_phone = normalize_phone(payload.user_phone)
    if not user_phone:
        raise HTTPException(status_code=400, detail="user_phone inválido")

    # 1) Traemos/creamos sesión
    session = get_or_create_session(user_phone)
    slots: Dict[str, Any] = {}
    try:
        slots = json.loads(session.get("slots_json") or "{}")
    except Exception:
        slots = {}

    # Historial de conversación básico
    conversation = slots.get("conversation") or []
    conversation.append({"role": "user", "content": payload.text})

    # 2) Preguntamos al LLM
    messages = build_messages(slots, conversation, payload.text)
    reply = ask_llm(messages)

    # 3) Normalizamos slots que devuelve el LLM
    new_slots = reply.get("updates", {}).get("slots", {}) if isinstance(reply.get("updates"), dict) else {}
    # merge de estado
    merged = {
        "zona": new_slots.get("zona", slots.get("zona")),
        "presupuesto_min": new_slots.get("presupuesto_min", slots.get("presupuesto_min")),
        "presupuesto_max": new_slots.get("presupuesto_max", slots.get("presupuesto_max")),
        "dormitorios": new_slots.get("dormitorios", slots.get("dormitorios")),
        "cochera": new_slots.get("cochera", slots.get("cochera")),
    }

    # Limpieza de tipos
    if merged["presupuesto_min"] is not None:
        merged["presupuesto_min"] = to_ars_number(str(merged["presupuesto_min"]))
    if merged["presupuesto_max"] is not None:
        merged["presupuesto_max"] = to_ars_number(str(merged["presupuesto_max"]))
    if merged["dormitorios"] is not None:
        try:
            merged["dormitorios"] = int(str(merged["dormitorios"]))
        except Exception:
            merged["dormitorios"] = None
    if merged["cochera"] is not None:
        merged["cochera"] = safe_bool_yesno(merged["cochera"])

    # 4) Adjuntamos conversación al estado
    assistant_text = str(reply.get("text") or "").strip()
    next_q = reply.get("next_question")
    vendor_push = bool(reply.get("vendor_push") is True)

    conversation.append({"role": "assistant", "content": assistant_text})
    merged["conversation"] = conversation

    # 5) Guardamos sesión y actualizamos lead
    update_session(user_phone, merged)
    vendor_phone = os.getenv("VENDOR_PHONE", "5493412654593")  # tu default
    ensure_lead_row(user_phone, vendor_phone)
    update_lead_from_slots(user_phone, merged)

    # 6) Si ya está calificado, sugerimos propiedades (hasta 3)
    suggestions = None
    if vendor_push and merged.get("zona") and merged.get("presupuesto_min") and merged.get("presupuesto_max"):
        props = suggest_properties(merged, limit=3)
        if props:
            # Formateo corto, no invasivo
            lines = []
            for p in props:
                cochera_txt = "sí" if p.get("cochera") == 1 else "no"
                lines.append(f"- {p['codigo']} • {p['direccion']} ({p['zona']}) • ${p['precio']} • {p['dormitorios']} dorm • cochera {cochera_txt}")
            # agregamos una línea final amable
            extra = "\n\nTe comparto algunas opciones que encajan con lo que buscás:\n" + "\n".join(lines)
            assistant_text = (assistant_text + extra)[:1500]  # por las dudas
            suggestions = props

    response = {
        "text": assistant_text,
        "next_question": next_q if next_q else None,
        "vendor_push": vendor_push,
        "updates": {"slots": merged},
        "suggestions": suggestions
    }
    return response
