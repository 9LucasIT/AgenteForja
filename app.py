# app.py
import os
import json
import re
import unicodedata
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# -------------------------------------------------------------------
# Config DB
# -------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("MYSQL_URL") or os.getenv("MYSQL_DATABASE_URL")
# Aseguramos driver pymysql
if DATABASE_URL and DATABASE_URL.startswith("mysql://"):
    DATABASE_URL = DATABASE_URL.replace("mysql://", "mysql+pymysql://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

# -------------------------------------------------------------------
# Modelos (solo lo que EXISTE en tu DB)
# chat_session: id, user_phone, slots_json, created_at, updated_at, ... (otras columnas ignoradas)
# -------------------------------------------------------------------
class ChatSession(Base):
    __tablename__ = "chat_session"

    id = Column(Integer, primary_key=True)
    user_phone = Column(String(32), index=True, nullable=False)
    slots_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

# -------------------------------------------------------------------
# FastAPI
# -------------------------------------------------------------------
app = FastAPI()


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _norm(txt: str) -> str:
    if not txt:
        return ""
    txt = unicodedata.normalize("NFD", txt).encode("ascii", "ignore").decode("utf-8")
    return re.sub(r"\s+", " ", txt.lower()).strip()

def detect_operacion(txt: str) -> Optional[str]:
    t = _norm(txt)
    rent_keys = ("alquiler", "alquilo", "alquilar", "renta", "rent", "en alquiler")
    sell_keys = ("venta", "vendo", "vender", "comprar", "compro", "en venta")
    if any(k in t for k in rent_keys):
        return "alquiler"
    if any(k in t for k in sell_keys):
        return "venta"
    return None

def parse_money(txt: str) -> Optional[int]:
    t = _norm(txt)
    nums = re.findall(r"\d{1,3}(?:[\.\,]?\d{3})*|\d+", t)
    if not nums:
        return None
    raw = nums[0].replace(".", "").replace(",", "")
    try:
        return int(raw)
    except:
        return None

def parse_int(txt: str) -> Optional[int]:
    t = _norm(txt)
    m = re.search(r"\b(\d+)\b", t)
    if not m:
        return None
    try:
        return int(m.group(1))
    except:
        return None

def yes_no(txt: str) -> Optional[bool]:
    t = _norm(txt)
    yes = ("si", "sí", "sii", "claro", "ok", "dale", "obvio", "si,", "si.")
    no = ("no", "nop", "nunca", "no,", "no.")
    if any(t == y or t.startswith(y + " ") for y in yes) or "tengo" in t or "quiero" in t:
        return True
    if any(t == n or t.startswith(n + " ") for n in no):
        return False
    return None

def has_address_number(txt: str) -> bool:
    # si hay número (para diferenciar "dirección exacta" de solo zona)
    return bool(re.search(r"\b\d{1,5}\b", txt or ""))

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

# -------------------------------------------------------------------
# Persistencia de sesión / slots
# -------------------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_or_create_session(db: Session, phone: str) -> ChatSession:
    sess = db.query(ChatSession).filter(ChatSession.user_phone == phone).first()
    if not sess:
        sess = ChatSession(user_phone=phone, slots_json=json.dumps({}))
        db.add(sess)
        db.commit()
        db.refresh(sess)
    return sess

def read_slots(sess: ChatSession) -> Dict[str, Any]:
    try:
        return json.loads(sess.slots_json or "{}") or {}
    except:
        return {}

def write_slots(db: Session, sess: ChatSession, slots: Dict[str, Any]) -> None:
    sess.slots_json = json.dumps(slots, ensure_ascii=False)
    sess.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(sess)

# -------------------------------------------------------------------
# Req / Res
# -------------------------------------------------------------------
class QualifyPayload(BaseModel):
    user_phone: str = Field(..., description="Número del cliente (solo dígitos, con código país)")
    text: str = Field("", description="Mensaje del cliente")
    message_id: Optional[str] = Field(None, description="ID del mensaje (opcional)")

class BotResponse(BaseModel):
    text: str
    next_question: str
    updates: Dict[str, Any] = {}
    vendor_push_guard: Optional[bool] = None

# -------------------------------------------------------------------
# Lógica de conversación por etapas (slots._stage)
# Etapas: op -> zona -> pmin -> pmax -> dorm -> cochera -> mascotas -> direccion -> resumen
# -------------------------------------------------------------------
def have_minimum_for_vendor(slots: Dict[str, Any]) -> bool:
    # Condición de "suficientemente calificado" para avisar al vendedor
    return bool(
        slots.get("operacion")
        and (slots.get("zona") or slots.get("direccion"))
        and slots.get("presupuesto_min")
        and slots.get("presupuesto_max")
        and slots.get("dormitorios") is not None
    )

def welcome_reset_message() -> str:
    return (
        "Arranquemos de nuevo! 😊\n"
        "Contame: ¿la búsqueda es para alquiler o para venta?\n"
        "Tip: cuando quieras reiniciar la conversación, escribí *reset* y empezamos de cero."
    )

# -------------------------------------------------------------------
# Endpoint
# -------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True, "ts": now_iso()}

@app.post("/qualify", response_model=BotResponse)
def qualify(payload: QualifyPayload, db: Session = Depends(get_db)):
    phone = (payload.user_phone or "").strip()
    text = (payload.text or "").strip()

    # Validaciones mínimas
    if not phone:
        # n8n: si ves 422 es porque no llegó user_phone
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="user_phone is required")

    sess = get_or_create_session(db, phone)
    slots = read_slots(sess)
    stage = slots.get("_stage") or "op"

    # RESET
    if _norm(text) == "reset":
        slots = {"_stage": "op"}
        write_slots(db, sess, slots)
        msg = welcome_reset_message()
        return BotResponse(text=msg, next_question=msg, updates={"slots": slots, "stage": "op"}, vendor_push_guard=False)

    # ------------------------
    # ETAPA: OPERACION
    # ------------------------
    if stage in ("op", "operacion"):
        # si ya la tenemos, avanzamos
        if slots.get("operacion"):
            slots["_stage"] = "zona"
            write_slots(db, sess, slots)
            q = "Perfecto. ¿En qué *zona* o *dirección exacta* estás interesado/a? (calle y número si lo tenés)"
            return BotResponse(text=q, next_question=q, updates={"slots": slots, "stage": "zona"}, vendor_push_guard=have_minimum_for_vendor(slots))

        op = detect_operacion(text)
        if op:
            slots["operacion"] = op
            slots["_stage"] = "zona"
            write_slots(db, sess, slots)
            q = f"Perfecto, {op}. Para ayudarte mejor, ¿en qué *zona* o *dirección exacta* estás buscando? (calle y número si lo tenés)"
            return BotResponse(text=q, next_question=q, updates={"slots": slots, "stage": "zona"}, vendor_push_guard=have_minimum_for_vendor(slots))

        ask = "¿La búsqueda es para *alquiler* o para *venta*?"
        return BotResponse(text=ask, next_question=ask, updates={"stage": "op"}, vendor_push_guard=False)

    # ------------------------
    # ETAPA: ZONA / DIRECCION
    # ------------------------
    if stage == "zona":
        if has_address_number(text):
            slots["direccion"] = text
        else:
            slots["zona"] = text

        slots["_stage"] = "pmin"
        write_slots(db, sess, slots)

        q = "¿Cuál sería tu *presupuesto mínimo* aproximado (en ARS)?"
        return BotResponse(text=q, next_question=q, updates={"slots": slots, "stage": "pmin"}, vendor_push_guard=have_minimum_for_vendor(slots))

    # ------------------------
    # ETAPA: PRESUPUESTO MIN
    # ------------------------
    if stage == "pmin":
        val = parse_money(text)
        if val is None:
            q = "No me quedó claro. Decime un número aproximado para el *presupuesto mínimo* (en ARS)."
            return BotResponse(text=q, next_question=q, updates={"stage": "pmin"}, vendor_push_guard=have_minimum_for_vendor(slots))
        slots["presupuesto_min"] = val
        slots["_stage"] = "pmax"
        write_slots(db, sess, slots)

        q = "Genial. ¿Y el *presupuesto máximo* aproximado (en ARS)?"
        return BotResponse(text=q, next_question=q, updates={"slots": slots, "stage": "pmax"}, vendor_push_guard=have_minimum_for_vendor(slots))

    # ------------------------
    # ETAPA: PRESUPUESTO MAX
    # ------------------------
    if stage == "pmax":
        val = parse_money(text)
        if val is None:
            q = "Entendido. ¿Podés indicarme un número para el *presupuesto máximo* (en ARS)?"
            return BotResponse(text=q, next_question=q, updates={"stage": "pmax"}, vendor_push_guard=have_minimum_for_vendor(slots))
        slots["presupuesto_max"] = val
        slots["_stage"] = "dorm"
        write_slots(db, sess, slots)

        q = "Para afinar la búsqueda: ¿Cuántos *dormitorios* te gustaría?"
        return BotResponse(text=q, next_question=q, updates={"slots": slots, "stage": "dorm"}, vendor_push_guard=have_minimum_for_vendor(slots))

    # ------------------------
    # ETAPA: DORMITORIOS
    # ------------------------
    if stage == "dorm":
        n = parse_int(text)
        if n is None:
            q = "¿Cuántos *dormitorios* querés? (ej.: 2)"
            return BotResponse(text=q, next_question=q, updates={"stage": "dorm"}, vendor_push_guard=have_minimum_for_vendor(slots))
        slots["dormitorios"] = n
        slots["_stage"] = "cochera"
        write_slots(db, sess, slots)

        q = "¿Necesitás *cochera*? (sí/no)"
        return BotResponse(text=q, next_question=q, updates={"slots": slots, "stage": "cochera"}, vendor_push_guard=have_minimum_for_vendor(slots))

    # ------------------------
    # ETAPA: COCHERA
    # ------------------------
    if stage == "cochera":
        yn = yes_no(text)
        if yn is None:
            q = "¿Te sirve con *cochera*? (sí/no)"
            return BotResponse(text=q, next_question=q, updates={"stage": "cochera"}, vendor_push_guard=have_minimum_for_vendor(slots))
        slots["cochera"] = bool(yn)
        slots["_stage"] = "mascotas"
        write_slots(db, sess, slots)

        q = "¿Tenés *mascotas* que debamos contemplar?"
        return BotResponse(text=q, next_question=q, updates={"slots": slots, "stage": "mascotas"}, vendor_push_guard=have_minimum_for_vendor(slots))

    # ------------------------
    # ETAPA: MASCOTAS
    # ------------------------
    if stage == "mascotas":
        t = _norm(text)
        tiene = None
        desc = None
        if any(w in t for w in ("si", "sí", "perro", "gato", "mascota", "perros", "gatos")):
            tiene = True
            desc = text
        elif t.startswith("no") or t == "no":
            tiene = False

        if tiene is None:
            q = "¿Tenés *mascotas*? (Podés decirme *no* o contame: perros, gatos, etc.)"
            return BotResponse(text=q, next_question=q, updates={"stage": "mascotas"}, vendor_push_guard=have_minimum_for_vendor(slots))

        slots["mascotas"] = desc if tiene else "no"
        # Si aún no tenemos dirección exacta, la pedimos
        if not slots.get("direccion"):
            slots["_stage"] = "direccion"
            write_slots(db, sess, slots)
            q = "¿Tenés una *dirección exacta*? (calle y número) Si no, decime *no tengo* y sigo."
            return BotResponse(text=q, next_question=q, updates={"slots": slots, "stage": "direccion"}, vendor_push_guard=have_minimum_for_vendor(slots))

        # Si ya hay dirección, vamos a resumen
        slots["_stage"] = "resumen"
        write_slots(db, sess, slots)

    # ------------------------
    # ETAPA: DIRECCION (opcional)
    # ------------------------
    if stage == "direccion":
        if has_address_number(text):
            slots["direccion"] = text
        # si dice "no tengo" o algo así, seguimos igual
        slots["_stage"] = "resumen"
        write_slots(db, sess, slots)

    # ------------------------
    # ETAPA: RESUMEN (calificado)
    # ------------------------
    if stage == "resumen":
        op = slots.get("operacion", "operación a definir")
        zona = slots.get("zona", "zona a definir")
        d = slots.get("direccion")
        rango = ""
        if slots.get("presupuesto_min") and slots.get("presupuesto_max"):
            rango = f"${slots['presupuesto_min']:,}–${slots['presupuesto_max']:,}".replace(",", ".")
        dorm = slots.get("dormitorios", "N/D")
        coch = "con cochera" if slots.get("cochera") else "sin cochera"
        masc = slots.get("mascotas", "sin info de mascotas")

        header = f"Perfecto 👍\n{op.capitalize()} en {('' if zona=='zona a definir' else zona)}"
        if d:
            header += f" (dirección: {d})"
        if rango:
            header += f". Presupuesto: {rango}."
        header += f" {dorm} dorm, {coch}, {masc}."

        # Señal para tu nodo IF en n8n
        push_guard = have_minimum_for_vendor(slots)

        # Dejamos la etapa en resumen para no repreguntar;
        # si el usuario escribe algo nuevo, podrías derivar a búsqueda de propiedades.
        write_slots(db, sess, slots)

        follow = "¿Querés que te envíe opciones que coincidan, o querés ajustar algún dato?"
        text_out = f"{header}\n\n{follow}"
        return BotResponse(
            text=text_out,
            next_question=follow,
            updates={"slots": slots, "stage": "resumen"},
            vendor_push_guard=push_guard,
        )

    # Fallback (no debería pasar)
    slots["_stage"] = "op"
    write_slots(db, sess, slots)
    ask = "¿La búsqueda es para *alquiler* o para *venta*?"
    return BotResponse(text=ask, next_question=ask, updates={"stage": "op"}, vendor_push_guard=False)
