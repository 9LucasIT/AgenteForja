import os
import json
import re
from datetime import datetime, timedelta

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, Boolean, select, update, delete
)
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Config & DB
# -----------------------------------------------------------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

# Usamos PyMySQL (mysql+pymysql:// ...)
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

# -----------------------------------------------------------------------------
# Modelos ORM mínimos (solo lo que usamos)
# -----------------------------------------------------------------------------
class ChatSession(Base):
    __tablename__ = "chat_session"

    id = Column(Integer, primary_key=True)
    user_phone = Column(String(30), index=True, nullable=False)
    last_message_id = Column(String(64), nullable=True)
    last_welcome_at = Column(DateTime, nullable=True)
    cooldown_until = Column(DateTime, nullable=True)
    status = Column(String(20), default="active")
    slots_json = Column(Text, default="{}")              # JSON serializado
    guard_already_sent = Column(Boolean, default=False)  # evita duplicar push

# Crea la tabla si no existe (no toca otras)
Base.metadata.create_all(engine)

# -----------------------------------------------------------------------------
# Schemas de entrada
# -----------------------------------------------------------------------------
class LeadMessage(BaseModel):
    message_id: str | None = None
    user_phone: str
    text: str = Field(default="")
    reset: bool = Field(default=False)  # opcional desde el webhook/normalizador


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="Real Estate Agent API", version="1.0.0")


@app.get("/healthz")
def healthz():
    return {"ok": True, "time": datetime.utcnow().isoformat()}


# -----------------------------------------------------------------------------
# Utilidades de sesión
# -----------------------------------------------------------------------------
WELCOME_REMINDER = (
    "Tip: cuando quieras *reiniciar* la conversación, escribí *reset* y empezamos de cero. 😉"
)

def _load_slots(slots_text: str) -> dict:
    if not slots_text:
        return {}
    try:
        return json.loads(slots_text)
    except Exception:
        return {}

def _save_slots(slots: dict) -> str:
    return json.dumps(slots, ensure_ascii=False)

def get_or_create_session(db, phone: str) -> ChatSession:
    sess = db.execute(select(ChatSession).where(ChatSession.user_phone == phone)).scalar_one_or_none()
    if not sess:
        sess = ChatSession(
            user_phone=phone,
            status="active",
            slots_json=_save_slots({
                "operation": None,      # alquiler / venta
                "zona": None,
                "direccion": None,
                "presupuesto_min": None,
                "presupuesto_max": None,
                "dormitorios": None,
                "cochera": None,
                "mascotas": None,
                "conversation": []
            }),
            guard_already_sent=False
        )
        db.add(sess)
        db.commit()
        db.refresh(sess)
    return sess

def should_push_vendor(slots: dict) -> bool:
    """
    Condición mínima: tener operación, zona o dirección,
    min & max de presupuesto y dormitorios — ajustá a gusto.
    """
    return all([
        bool(slots.get("operation")),
        bool(slots.get("zona") or slots.get("direccion")),
        isinstance(slots.get("presupuesto_min"), (int, float)),
        isinstance(slots.get("presupuesto_max"), (int, float)),
        isinstance(slots.get("dormitorios"), int)
    ])

def human_summary(slots: dict) -> str:
    op = slots.get("operation") or "a definir"
    zona = slots.get("zona") or (slots.get("direccion") or "sin zona/dirección")
    pmin = slots.get("presupuesto_min") or "N/D"
    pmax = slots.get("presupuesto_max") or "N/D"
    dorm = slots.get("dormitorios")
    dorm_s = f"{dorm} dorm" if isinstance(dorm, int) else "sin datos de dorm."
    coch = slots.get("cochera")
    coch_s = "con cochera" if coch is True else ("sin cochera" if coch is False else "sin info de cochera")
    masc = slots.get("mascotas")
    masc_s = masc if isinstance(masc, str) and masc.strip() else "sin info de mascotas"

    return f"{op.capitalize()} en {zona}. Presupuesto: ${pmin}–${pmax}. {dorm_s}, {coch_s}, {masc_s}."


def first_time_welcome_needed(sess: ChatSession) -> bool:
    # Mostramos el tip de "reset" 1 sola vez por sesión
    return not bool(sess.last_welcome_at)


# -----------------------------------------------------------------------------
# NLU básica (regex muy simples)
# -----------------------------------------------------------------------------
RE_NUM = re.compile(r"\d+")
RE_PRESU = re.compile(r"\$?\s*(\d[\d\.]*)")

def extract_numbers(text: str) -> list[int]:
    nums = [int("".join(n.split("."))) for n in RE_NUM.findall(text)]
    return nums

def parse_yes(text: str) -> bool | None:
    t = text.lower()
    if any(x in t for x in ["sí", "si", "claro", "ok", "dale", "afirmativo"]):
        return True
    if any(x in t for x in ["no", "nop", "negativo"]):
        return False
    return None

def parse_operation(text: str) -> str | None:
    t = text.lower()
    if "alquiler" in t or "alquilar" in t or "alquilo" in t:
        return "alquiler"
    if "venta" in t or "vender" in t or "compro" in t:
        return "venta"
    return None


# -----------------------------------------------------------------------------
# Core de diálogo
# -----------------------------------------------------------------------------
def next_question_for(slots: dict) -> str | None:
    """
    Decide qué preguntar luego, en orden lógico.
    """
    if not slots.get("operation"):
        return "¿La búsqueda es para *alquiler* o para *venta*?"

    if not (slots.get("zona") or slots.get("direccion")):
        return "¿En qué *zona* o *dirección exacta* estás interesado? (calle y número si lo tenés)"

    if not slots.get("presupuesto_min"):
        return "¿Cuál sería tu *presupuesto mínimo* aproximado (en ARS)?"

    if not slots.get("presupuesto_max"):
        return "¿Y el *presupuesto máximo*?"

    if not slots.get("dormitorios"):
        return "¿Cuántos *dormitorios* te gustaría?"

    if slots.get("cochera") is None:
        return "¿Necesitás *cochera*?"

    if slots.get("mascotas") in (None, ""):
        return "Para afinar la búsqueda: ¿Tenés *mascotas* que debamos contemplar?"

    # Todo colectado → no hay próxima pregunta
    return None


def apply_user_message(slots: dict, text: str) -> None:
    """
    Actualiza slots a partir del mensaje del usuario con heurísticas simples.
    """
    # operation
    op = parse_operation(text)
    if op:
        slots["operation"] = op

    # direccion / zona
    if any(k in text.lower() for k in ["calle", "av ", "avenida", "altura", "n°", "numero", "número"]):
        slots["direccion"] = text.strip()
    elif "zona" in text.lower() or "barrio" in text.lower():
        slots["zona"] = text.strip()

    # presupuestos (extraemos todos los números, tomamos el menor como min y el mayor como max si corresponde)
    if any(k in text.lower() for k in ["presupuesto", "$", "precio", "maximo", "mínimo", "minimo", "rango"]):
        nums = extract_numbers(text)
        if nums:
            lo, hi = min(nums), max(nums)
            # si falta min
            if not slots.get("presupuesto_min"):
                slots["presupuesto_min"] = lo
            # si falta max
            if not slots.get("presupuesto_max"):
                slots["presupuesto_max"] = hi if hi >= lo else lo

    # dormitorios
    if "dorm" in text.lower():
        nums = extract_numbers(text)
        if nums:
            slots["dormitorios"] = nums[0]

    # cochera
    if "cochera" in text.lower():
        yn = parse_yes(text)
        if yn is not None:
            slots["cochera"] = yn

    # mascotas
    if "mascota" in text.lower() or "perro" in text.lower() or "gato" in text.lower():
        slots["mascotas"] = text.strip()


def build_agent_reply(slots: dict, intro_ok: bool = False) -> dict:
    """
    Construye la respuesta del agente con tono humano y 1 sola pregunta.
    """
    header = ""
    if intro_ok:
        header = f"Perfecto 👍\n{human_summary(slots)}\n"

    q = next_question_for(slots)

    if not q:
        # calificado → se genera push al vendedor
        text = header or (human_summary(slots) + "\n")
        text += "¡Listo! Con eso ya puedo pasarle el resumen al asesor para que te contacte enseguida. 🙌"
        return {
            "text": text,
            "next_question": None,
            "vendor_push": True,
            "updates": {"slots": slots}
        }

    # hay próxima pregunta
    text = header + q
    return {
        "text": text,
        "next_question": q,
        "vendor_push": False,
        "updates": {"slots": slots}
    }


# -----------------------------------------------------------------------------
# Endpoint principal
# -----------------------------------------------------------------------------
@app.post("/qualify")
def qualify(payload: LeadMessage, background_tasks: BackgroundTasks):
    """
    Endpoint principal que procesa los mensajes entrantes del usuario.
    Maneja 'reset' y construye la siguiente pregunta/acción.
    """
    incoming_text = (payload.text or "").strip()

    with SessionLocal() as db:
        # Obtener o crear sesión
        sess = get_or_create_session(db, payload.user_phone)
        slots = _load_slots(sess.slots_json)

        # Guardamos mini conversación (para contexto, opcional)
        slots.setdefault("conversation", [])
        if incoming_text:
            slots["conversation"].append({"role": "user", "content": incoming_text})
            # mantenemos como máx. 15 turnos
            slots["conversation"] = slots["conversation"][-15:]

        # ---------------------------------------------------------------------
        # 1) RESET de sesión (si esto ocurre, se borra todo y volvemos a empezar)
        # ---------------------------------------------------------------------
        lower_t = incoming_text.lower()
        if payload.reset or lower_t in {"reset", "reiniciar", "nuevo chat", "empezar de nuevo"}:
            # borramos cualquier sesión y creamos una limpia
            db.execute(delete(ChatSession).where(ChatSession.user_phone == payload.user_phone))
            db.commit()

            new = get_or_create_session(db, payload.user_phone)
            new.last_welcome_at = datetime.utcnow()  # marcamos que ya mostramos el tip
            new.slots_json = _save_slots({
                "operation": None,
                "zona": None,
                "direccion": None,
                "presupuesto_min": None,
                "presupuesto_max": None,
                "dormitorios": None,
                "cochera": None,
                "mascotas": None,
                "conversation": []
            })
            db.commit()

            warm = (
                "¡Arranquemos de nuevo! 😊\n"
                "Contame: ¿la búsqueda es para *alquiler* o para *venta*?\n"
                f"{WELCOME_REMINDER}"
            )
            return {
                "text": warm,
                "next_question": "¿Es para *alquiler* o *venta*?",
                "vendor_push": False,
                "updates": {"slots": _load_slots(new.slots_json)}
            }

        # ---------------------------------------------------------------------
        # 2) NLU simple → actualizar slots con el mensaje del usuario
        # ---------------------------------------------------------------------
        apply_user_message(slots, incoming_text)

        # ---------------------------------------------------------------------
        # 3) Primera vez: mostramos el TIP de reset solo una vez por sesión
        # ---------------------------------------------------------------------
        intro = False
        if first_time_welcome_needed(sess):
            intro = True
            sess.last_welcome_at = datetime.utcnow()

        # ---------------------------------------------------------------------
        # 4) Construir respuesta del agente
        # ---------------------------------------------------------------------
        result = build_agent_reply(slots, intro_ok=intro)

        # ---------------------------------------------------------------------
        # 5) Guard de notificación al vendedor (no duplica)
        # ---------------------------------------------------------------------
        if result.get("vendor_push"):
            if not sess.guard_already_sent and should_push_vendor(slots):
                # Señal para tu workflow (n8n enviará al vendedor)
                # marcamos banderita local para que el IF de n8n pueda leerla si lo necesitas
                result["vendor_push_guard"] = True
                # y marcamos en DB para no volver a empujar
                sess.guard_already_sent = True
            else:
                # ya enviado → no empujar otra vez
                result["vendor_push_guard"] = False

        # ---------------------------------------------------------------------
        # 6) Persistir cambios de sesión
        # ---------------------------------------------------------------------
        sess.slots_json = _save_slots(result["updates"]["slots"])
        sess.last_message_id = payload.message_id or sess.last_message_id
        db.commit()

        # ---------------------------------------------------------------------
        # 7) Agregar contexto del asistente (opcional, histórico)
        # ---------------------------------------------------------------------
        slots_after = _load_slots(sess.slots_json)
        slots_after["conversation"].append({"role": "assistant", "content": result["text"]})
        sess.slots_json = _save_slots(slots_after)
        db.commit()

        # Responder
        # Si es bienvenida (intro) agregamos el TIP de reset UNA sola vez
        if intro and result.get("text") and WELCOME_REMINDER not in result["text"]:
            result["text"] = result["text"] + "\n\n" + WELCOME_REMINDER

        return result
