import os
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text

# =========================
# Config & Conexión a MySQL
# =========================

DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL env var is required")

# Forzar charset y un pool sano para Railway
if "?" in DATABASE_URL:
    DATABASE_URL += "&charset=utf8mb4"
else:
    DATABASE_URL += "?charset=utf8mb4"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # reconecta si el pool queda “muerto”
    pool_recycle=300,     # recicla conexiones inactivas (Railway)
    echo=False,
    future=True,
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # opcional
VENDOR_PHONE = os.getenv("VENDOR_PHONE", "")  # se usa en n8n (mensaje al vendedor)

# ==============
# FastAPI & I/O
# ==============

app = FastAPI(title="Real-Estate Qualifier API")

class QualifyIn(BaseModel):
    message_id: Optional[str] = None
    user_phone: str
    text: str

class QualifyOut(BaseModel):
    text: str
    next_question: Optional[str] = None
    vendor_push: bool = False
    updates: Dict[str, Any]


# =========================
# Utilidades de persistencia
# =========================

def _load_session(user_phone: str) -> Dict[str, Any]:
    """
    Lee la última sesión desde chat_session. Si no hay, devuelve slots vacíos.
    """
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text("""
                    SELECT id, slots_json, last_message_id, status
                    FROM chat_session
                    WHERE user_phone = :p
                    ORDER BY id DESC
                    LIMIT 1
                """),
                {"p": user_phone},
            ).first()

            if row:
                slots = row.slots_json or {}
                # slots_json puede venir como str (según cómo esté creada la tabla)
                if isinstance(slots, str):
                    try:
                        slots = json.loads(slots)
                    except Exception:
                        slots = {}
                return dict(slots)
            return {}
    except Exception as e:
        print("DB load error:", e)
        # lanzamos 500 para que n8n no repita la misma pregunta en bucle
        raise HTTPException(status_code=500, detail="DB unavailable")


def _save_session(user_phone: str, slots: Dict[str, Any], last_message_id: Optional[str] = None):
    """
    Inserta o actualiza la fila de chat_session para el user_phone.
    Si tu tabla NO tiene unique(user_phone), hacemos select+insert/update.
    """
    try:
        payload = json.dumps(slots, ensure_ascii=False)
        with engine.begin() as conn:
            existing = conn.execute(
                text("SELECT id FROM chat_session WHERE user_phone=:p ORDER BY id DESC LIMIT 1"),
                {"p": user_phone},
            ).first()

            if existing:
                conn.execute(
                    text("""
                        UPDATE chat_session
                           SET slots_json=:s,
                               last_message_id=:mid,
                               updated_at=NOW()
                         WHERE id=:id
                    """),
                    {"s": payload, "mid": last_message_id, "id": existing.id},
                )
            else:
                conn.execute(
                    text("""
                        INSERT INTO chat_session (user_phone, slots_json, last_message_id, status, created_at, updated_at)
                        VALUES (:p, :s, :mid, 'active', NOW(), NOW())
                    """),
                    {"p": user_phone, "s": payload, "mid": last_message_id},
                )
    except Exception as e:
        print("DB save error:", e)
        raise HTTPException(status_code=500, detail="DB unavailable")


# =========================
# Extracción/Progreso de slots
# =========================

REQUIRED_SLOTS_ORDER = ["zona", "presupuesto_min", "presupuesto_max", "dormitorios", "cochera"]

def _simple_extract(slots: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    """
    Heurísticas rápidas (sin LLM) para capturar info de forma robusta.
    Se complementa con LLM si tenés OPENAI_API_KEY.
    """
    t = user_text.lower()

    # zona: si hay una palabra tipo barrio conocida o el usuario menciona “barrio/zona/avenida/calle”
    if "zona" in t or "barrio" in t or "avenida" in t or "calle" in t or "dirección" in t or "direccion" in t:
        # toma algo simple: la última palabra significativa
        tokens = re.findall(r"[a-záéíóúñ]+", t)
        if tokens:
            # preferimos la última palabra distinta de “zona/barrio”
            for w in reversed(tokens):
                if w not in {"zona", "barrio", "la", "el"} and len(w) > 2:
                    slots.setdefault("zona", w.capitalize())
                    break

    # números → presupuesto o dormitorios
    # captura todos los enteros (ej: 120000, 3)
    nums = re.findall(r"\d{2,}|\b\d\b", t.replace(".", "").replace(",", ""))
    if nums:
        # si menciona “presupuesto” y hay números, úsalo
        if "presu" in t or "$" in t:
            val = int(nums[0])
            if "presupuesto_min" not in slots:
                slots["presupuesto_min"] = val
            elif "presupuesto_max" not in slots and val >= slots.get("presupuesto_min", 0):
                slots["presupuesto_max"] = val

        # dormitorios
        if "dorm" in t or "habit" in t or "cuarto" in t:
            # toma el número de 1 dígito si existe
            d = [int(n) for n in nums if len(n) == 1]
            if d and "dormitorios" not in slots:
                slots["dormitorios"] = d[0]

    # cochera
    if "cocher" in t:
        if "no" in t or "sin" in t:
            slots.setdefault("cochera", 0)
        else:
            slots.setdefault("cochera", 1)

    return slots


def _decide_next(slots: Dict[str, Any]) -> Dict[str, Any]:
    """
    Devuelve la próxima pregunta o vendor_push=True si ya está para el vendedor.
    """
    for s in REQUIRED_SLOTS_ORDER:
        if not slots.get(s):
            if s == "zona":
                return {"slot": s, "question": "Perfecto, ¿en qué zona o dirección estás interesado?"}
            if s == "presupuesto_min":
                return {"slot": s, "question": "¿Cuál sería tu presupuesto mínimo aproximado (en ARS)?"}
            if s == "presupuesto_max":
                return {"slot": s, "question": "¿Y el presupuesto máximo?"}
            if s == "dormitorios":
                return {"slot": s, "question": "¿Cuántos dormitorios te gustaría?"}
            if s == "cochera":
                return {"slot": s, "question": "¿Necesitás cochera? (sí/no)"}

    # Si todo lo básico está, empujamos al vendedor
    return {"vendor_push": True}


# (Opcional) Extractor con LLM – se usa solo si tenés OPENAI_API_KEY
def _llm_enrich(slots: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return slots
    try:
        # OpenAI client nuevo (SDK v1.x)
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = f"""
Actuás como asistente inmobiliario. A partir del mensaje del cliente,
intentá inferir SOLO si aparecen: zona, presupuesto_min, presupuesto_max, dormitorios, cochera.
Devolvé un JSON con esas claves (si no podés inferir una, déjala nula).
Mensaje del cliente: {user_text}
"""
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Devolvé solo JSON válido. Sin texto adicional."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = rsp.choices[0].message.content.strip()
        inferred = json.loads(content)

        # merge sin pisar valores ya confirmados
        for k in ["zona", "presupuesto_min", "presupuesto_max", "dormitorios", "cochera"]:
            v = inferred.get(k)
            if v and slots.get(k) in (None, "", 0):
                slots[k] = v
    except Exception as e:
        print("LLM enrich error:", e)
    return slots


# =========================
# Endpoints
# =========================

@app.get("/healthz")
def healthz():
    with engine.connect() as c:
        c.execute(text("SELECT 1"))
    return {"ok": True}


@app.post("/qualify", response_model=QualifyOut)
def qualify(payload: QualifyIn):
    """
    Entrada esperada (desde n8n):
    {
      "message_id": "...",            // opcional
      "user_phone": "5493412....",
      "text": "texto del cliente"
    }
    """
    user_phone = payload.user_phone
    user_text = (payload.text or "").strip()

    if not user_phone:
        raise HTTPException(status_code=400, detail="user_phone required")
    if not user_text:
        # si no hay texto, no hacemos nada
        return QualifyOut(
            text="",
            next_question=None,
            vendor_push=False,
            updates={"slots": {}},
        )

    # 1) cargar slots previos
    slots = _load_session(user_phone)

    # 2) heurísticas rápidas
    slots = _simple_extract(slots, user_text)

    # 3) (opcional) enriquecer con LLM
    slots = _llm_enrich(slots, user_text)

    # 4) decidir siguiente paso
    nxt = _decide_next(slots)

    # 5) armar respuesta natural (agente)
    if nxt.get("vendor_push"):
        # resumen corto para el cliente
        zona = slots.get("zona", "N/D")
        pmin = slots.get("presupuesto_min", "N/D")
        pmax = slots.get("presupuesto_max", "N/D")
        dorm = slots.get("dormitorios", "N/D")
        coch = "sí" if slots.get("cochera") == 1 else ("no" if slots.get("cochera") == 0 else "N/D")

        reply = (
            "¡Genial! Ya tengo lo principal.\n"
            f"- Zona: {zona}\n"
            f"- Presupuesto: {pmin} a {pmax}\n"
            f"- Dormitorios: {dorm}\n"
            f"- Cochera: {coch}\n\n"
            "Le paso tu consulta al asesor y te escribe en breve 🧑‍💼"
        )
        # 6) persistir y responder
        _save_session(user_phone, slots, payload.message_id)
        return QualifyOut(
            text=reply,
            next_question=None,
            vendor_push=True,  # <- clave para n8n
            updates={"slots": slots},
        )
    else:
        question = nxt["question"]
        # si faltan varias cosas, guiamos suave
        if "zona" in nxt["slot"]:
            lead = "Gracias. Para avanzar, contame la zona/dirección y un presupuesto estimado; después vemos dormitorios y cochera. 🙂\n"
            reply = lead + question
        else:
            reply = question

        _save_session(user_phone, slots, payload.message_id)
        return QualifyOut(
            text=reply,
            next_question=question,
            vendor_push=False,
            updates={"slots": slots},
        )
