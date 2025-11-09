# app.py
import os
import re
import json
import unicodedata
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, text as sql_text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# ===========================
# CONFIG
# ===========================
SITE_URL = os.getenv("SITE_URL", "https://www.veglienzone.com.ar").rstrip("/")

DATABASE_URL = (
    os.getenv("DATABASE_URL")
    or os.getenv("MYSQL_URL")
    or os.getenv("MYSQL_DATABASE_URL")
    or ""
)
if DATABASE_URL.startswith("mysql://"):
    DATABASE_URL = DATABASE_URL.replace("mysql://", "mysql+pymysql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

# ===========================
# MODELOS (existente)
# ===========================
class ChatSession(Base):
    __tablename__ = "chat_session"
    id = Column(Integer, primary_key=True)
    user_phone = Column(String(32), index=True, nullable=False)
    slots_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

# ===========================
# FASTAPI
# ===========================
app = FastAPI()

# ===========================
# HELPERS GENERALES
# ===========================
def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _norm(txt: str) -> str:
    if not txt:
        return ""
    txt = unicodedata.normalize("NFD", txt).encode("ascii", "ignore").decode("utf-8")
    return re.sub(r"\s+", " ", txt.lower()).strip()

def out(reply_text: str, *, vendor_push: bool=False, vendor_message: Optional[str]=None,
        closing_text: Optional[str]=None, slots: Optional[dict]=None, stage: Optional[str]=None):
    return {
        "reply_text": reply_text or "Perdón, estamos con un inconveniente técnico. ¿Podemos intentar de nuevo?",
        "vendor_push": bool(vendor_push),
        "vendor_message": vendor_message,
        "closing_text": closing_text,
        # inocuos para n8n (por si querés ver slots en logs)
        "slots": slots,
        "stage": stage
    }

def yes_no(txt: str) -> Optional[bool]:
    t = _norm(txt)
    yes = ("si", "sí", "claro", "ok", "dale", "obvio")
    no = ("no", "nop", "nunca")
    if any(t == y or t.startswith(y + " ") for y in yes): return True
    if any(t == n or t.startswith(n + " ") for n in no): return False
    return None

def parse_int(txt: str) -> Optional[int]:
    t = _norm(txt)
    m = re.search(r"\b(\d+)\b", t)
    if not m: return None
    try: return int(m.group(1))
    except: return None

def parse_money(txt: str) -> Optional[int]:
    t = _norm(txt)
    nums = re.findall(r"\d{1,3}(?:[.,]?\d{3})*|\d+", t)
    if not nums: return None
    raw = nums[0].replace(".", "").replace(",", "")
    try: return int(raw)
    except: return None

def detect_operacion(txt: str) -> Optional[str]:
    t = _norm(txt)
    if any(k in t for k in ("alquiler", "alquilo", "alquilar", "renta", "rent", "en alquiler")):
        return "alquiler"
    if any(k in t for k in ("venta", "vendo", "vender", "comprar", "compro", "en venta")):
        return "venta"
    return None

def has_address_number(txt: str) -> bool:
    return bool(re.search(r"[a-zA-Záéíóúñü\.]{3,}\s+\d{1,6}", txt or ""))

# ===========================
# BD: búsqueda por código / dirección / zona (como tenías)
# ===========================
STREET_HINTS = [
    "calle", "c/", "av", "avenida", "pasaje", "pas", "pje", "ruta", "rn", "rp",
    "boulevard", "bvard", "bv", "diagonal", "diag", "esquina", "entre", "altura"
]
CODIGO_RE = re.compile(r"\b([A-Z]\d{3})\b", re.IGNORECASE)

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()

def _looks_like_address(text: str) -> bool:
    t = _normalize(text)
    if any(h in t for h in STREET_HINTS):
        return True
    return bool(re.search(r"[a-zA-Záéíóúñü\.]{3,}\s+\d{1,6}", t))

def _address_tokens(text: str):
    t = _normalize(text)
    m = re.search(r"([a-zA-Záéíóúñü\. ]{3,})\s+(\d{1,6})", t)
    street, number = (None, None)
    if m:
        street = _normalize(m.group(1)).replace(".", "").strip()
        number = m.group(2)
    words = [w for w in re.split(r"[^\wáéíóúñü]+", t) if len(w) >= 4]
    return street, number, words

def _ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()

def find_property_by_user_text(db: Session, text_in: str) -> Optional[Dict[str, Any]]:
    t = _normalize(text_in or "")

    # 1) Código tipo A101/B202
    m = CODIGO_RE.search(text_in or "")
    if m:
        codigo = m.group(1).upper()
        row = db.execute(
            sql_text("SELECT id, codigo, direccion, zona, precio, dormitorios, cochera "
                     "FROM propiedades WHERE codigo=:codigo LIMIT 1"),
            {"codigo": codigo}
        ).mappings().first()
        if row: return dict(row)

    # 2) Dirección aparente
    if _looks_like_address(t):
        street, number, words = _address_tokens(t)
        if street and number:
            like = f"%{street.split()[0]}%{number}%"
            row = db.execute(
                sql_text("SELECT id, codigo, direccion, zona, precio, dormitorios, cochera "
                         "FROM propiedades WHERE direccion LIKE :like LIMIT 1"),
                {"like": like}
            ).mappings().first()
            if row: return dict(row)

        like_parts = [w for w in words if w not in {"calle", "avenida"}][:2]
        if like_parts:
            where = " AND ".join([f"direccion LIKE :w{i}" for i in range(len(like_parts))])
            params = {f"w{i}": f"%{w}%" for i, w in enumerate(like_parts)}
            rows = db.execute(
                sql_text(f"SELECT id, codigo, direccion, zona, precio, dormitorios, cochera "
                         f"FROM propiedades WHERE {where} LIMIT 3"),
                params
            ).mappings().all()
            if rows:
                best = max(rows, key=lambda r: _ratio(" ".join(words), r["direccion"]))
                return dict(best)

    # 3) Zona exacta
    zonas = db.execute(sql_text("SELECT DISTINCT zona FROM propiedades")).scalars().all()
    for z in zonas:
        if _normalize(z) in t:
            row = db.execute(
                sql_text("SELECT id, codigo, direccion, zona, precio, dormitorios, cochera "
                         "FROM propiedades WHERE zona=:z ORDER BY precio ASC LIMIT 1"),
                {"z": z}
            ).mappings().first()
            if row: return dict(row)

    return None

def build_humane_property_reply(p: Dict[str, Any]) -> str:
    cochera_txt = "con cochera" if (p.get("cochera") in (1, True)) else "sin cochera"
    precio = int(p["precio"]) if p.get("precio") is not None else 0
    precio_txt = f"${precio:,}".replace(",", ".") if precio else "a consultar"
    return (
        "¡Genial! Sobre esa propiedad:\n"
        f"• Código: {p.get('codigo', 'N/D')}\n"
        f"• Dirección: {p.get('direccion','N/D')} ({p.get('zona','N/D')})\n"
        f"• Precio: {precio_txt}\n"
        f"• Dormitorios: {p.get('dormitorios','N/D')} – {cochera_txt}\n\n"
        "¿Querés que coordinemos una visita o te envío opciones parecidas en la zona?"
    )

def build_vendor_summary(user_phone: str, p: Dict[str, Any], slots: Dict[str, Any]) -> str:
    return (
        f"Lead +{user_phone} consultó por COD {p.get('codigo','N/D')} – {p.get('direccion','N/D')} ({p.get('zona','N/D')}).\n"
        f"Operación: {slots.get('operacion','N/D')} | Presup.: min {slots.get('presupuesto_min','N/D')} / max {slots.get('presupuesto_max','N/D')} | "
        f"Dorms: {slots.get('dormitorios','N/D')} | Cochera: {slots.get('cochera','N/D')}."
    )

# ===========================
# SESIONES / SLOTS
# ===========================
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
        db.add(sess); db.commit(); db.refresh(sess)
    return sess

def read_slots(sess: ChatSession) -> Dict[str, Any]:
    try: return json.loads(sess.slots_json or "{}") or {}
    except: return {}

def write_slots(db: Session, sess: ChatSession, slots: Dict[str, Any]) -> None:
    sess.slots_json = json.dumps(slots, ensure_ascii=False)
    sess.updated_at = datetime.utcnow()
    db.commit(); db.refresh(sess)

def welcome_reset_message() -> str:
    return (
        "¡Arranquemos de nuevo! 😊\n"
        "Contame: ¿la búsqueda es para *alquiler* o para *venta*?\n"
        "Tip: cuando quieras reiniciar la conversación, escribí *reset* y empezamos de cero."
    )

# ===========================
# TASACIÓN (lo que aprobaste)
# ===========================
_TAS_INICIO_KEYWORDS = (
    "tasacion", "tasación", "tasar", "quiero tasar",
    "hacer tasacion", "necesito tasacion", "valuar", "valuación", "valoracion"
)

def _looks_like_tasacion_start(txt: str) -> bool:
    t = _norm(txt)
    return any(k in t for k in _TAS_INICIO_KEYWORDS)

def _join_features(lst):
    if not lst: return "no informado"
    s = [x for x in lst if x]
    return ", ".join(s) if s else "no informado"

def _extract_features(txt: str):
    t = _norm(txt)
    feats = []
    if any(w in t for w in ("balcon", "balcón", "balkon")): feats.append("balcón")
    if "patio" in t: feats.append("patio")
    if "amenities" in t: feats.append("amenities")
    if "estudio" in t or "factibilidad" in t: feats.append("estudio factibilidad")
    if t in ("no", "ninguno", "ninguna", "ningunos"): return []
    return feats

def _tas_questions(step: str) -> str:
    prompts = {
        "op":   "¡Genial! Para avanzar con la tasación, decime: ¿*tipo de operación*? (venta o alquiler)",
        "prop": "Perfecto. ¿Cuál es el *tipo de propiedad*? (ej.: departamento, casa, local, oficina)",
        "m2":   "Gracias. ¿Cuántos *metros cuadrados* aproximados tiene la propiedad?",
        "dir":  "Anotado. ¿Cuál es la *dirección exacta* del inmueble? (calle y número; si podés, piso/depto)",
        "exp":  "¿La propiedad tiene *expensas*? Si tiene, ¿de cuánto es el *costo mensual* (ARS)? Si no, podés decir *no tiene*.",
        "feat": "¿La propiedad dispone de *balcón, patio, amenities o estudio de factibilidad*? Podés responder con una lista (ej.: “balcón y amenities”) o “no”.",
        "disp": "¡Último dato! ¿Cuál es tu *disponibilidad horaria* aproximada para que te contacte un asesor?",
        "fin":  "Perfecto, con todos estos datos ya cuento con lo suficiente para derivarte con un asesor, muchisimas gracias por tu tiempo!",
    }
    return prompts.get(step, prompts["op"])

def _ensure_tas_slots(slots: dict) -> dict:
    slots.setdefault("_flow", "tasacion")
    slots.setdefault("_tas_step", "op")
    for k in ("tas_operacion","tas_propiedad","tas_m2","tas_direccion","tas_expensas","tas_features","tas_disponibilidad"):
        slots.setdefault(k, None)
    return slots

class BotResponse(BaseModel):
    text: str
    next_question: Optional[str] = None
    updates: Dict[str, Any] = {}
    vendor_push: Optional[bool] = None
    vendor_message: Optional[str] = None

def handle_tasacion(slots: dict, incoming_text: str, user_phone: str) -> Dict[str, Any]:
    slots = _ensure_tas_slots(slots)
    step = slots.get("_tas_step", "op")
    text = incoming_text or ""

    if step == "op":
        op = _norm(text)
        if "venta" in op: slots["tas_operacion"] = "venta"; slots["_tas_step"] = "prop"
        elif "alquiler" in op or "renta" in op: slots["tas_operacion"] = "alquiler"; slots["_tas_step"] = "prop"
        q = _tas_questions(slots["_tas_step"])
        return out(q, slots=slots, stage=slots["_tas_step"])

    if step == "prop":
        slots["tas_propiedad"] = text.strip() or "no informado"
        slots["_tas_step"] = "m2"
        q = _tas_questions("m2")
        return out(q, slots=slots, stage="m2")

    if step == "m2":
        m2 = parse_int(text)
        if m2 is None:
            q = "¿Me pasás un número aproximado de *metros cuadrados*? (ej.: 65)"
            return out(q, slots=slots, stage="m2")
        slots["tas_m2"] = m2
        slots["_tas_step"] = "dir"
        q = _tas_questions("dir")
        return out(q, slots=slots, stage="dir")

    if step == "dir":
        slots["tas_direccion"] = text.strip() or "no informado"
        slots["_tas_step"] = "exp"
        q = _tas_questions("exp")
        return out(q, slots=slots, stage="exp")

    if step == "exp":
        t = _norm(text)
        if any(w in t for w in ("no tiene", "no", "sin expensas", "sin")):
            slots["tas_expensas"] = "no tiene"
        else:
            val = parse_money(text)
            slots["tas_expensas"] = f"${val:,}".replace(",", ".") if val else (text.strip() or "no informado")
        slots["_tas_step"] = "feat"
        q = _tas_questions("feat")
        return out(q, slots=slots, stage="feat")

    if step == "feat":
        feats = _extract_features(text)
        slots["tas_features"] = _join_features(feats) if feats is not None else "no informado"
        slots["_tas_step"] = "disp"
        q = _tas_questions("disp")
        return out(q, slots=slots, stage="disp")

    if step == "disp":
        slots["tas_disponibilidad"] = text.strip() or "no informado"
        slots["_tas_step"] = "fin"
        resumen = (
            "Tasación solicitada ✅\n"
            f"• Operación: {slots.get('tas_operacion','N/D')}\n"
            f"• Propiedad: {slots.get('tas_propiedad','N/D')}\n"
            f"• Metros²: {slots.get('tas_m2','N/D')}\n"
            f"• Dirección: {slots.get('tas_direccion','N/D')}\n"
            f"• Expensas: {slots.get('tas_expensas','N/D')}\n"
            f"• Extras: {slots.get('tas_features','N/D')}\n"
            f"• Disponibilidad: {slots.get('tas_disponibilidad','N/D')}\n"
            f"• Tel cliente: +{user_phone}"
        )
        closing = _tas_questions("fin")
        return out(closing, vendor_push=True, vendor_message=resumen, closing_text=closing, slots=slots, stage="fin")

    slots["_tas_step"] = "op"
    q = _tas_questions("op")
    return out(q, slots=slots, stage="op")

# ===========================
# REQUEST / RESPONSE MODELS (entrada flexible)
# ===========================
class QualifyPayload(BaseModel):
    # puede venir user_phone o no; si no, lo deducimos de chatId/etc.
    user_phone: Optional[str] = Field(None)
    text: Optional[str] = Field("", description="Mensaje del cliente")

    # alternativos que puede mandar n8n
    chatId: Optional[str] = None
    message: Optional[str] = None
    From: Optional[str] = None
    waId: Optional[str] = None
    Body: Optional[str] = None
    senderData: Optional[dict] = None
    messageData: Optional[dict] = None
    message_id: Optional[str] = None

# ===========================
# ENDPOINTS
# ===========================
@app.get("/healthz")
def healthz():
    return {"ok": True, "ts": now_iso()}

@app.post("/qualify")
def qualify(payload: QualifyPayload, db: Session = Depends(get_db)):
    # ---- compat entrada ----
    phone = (payload.user_phone or "").strip()
    if not phone:
        chat_id = (
            payload.chatId or payload.From or payload.waId or
            (payload.senderData or {}).get("sender") or (payload.senderData or {}).get("chatId") or ""
        )
        chat_id = str(chat_id)
        if chat_id:
            phone = chat_id.replace("@c.us", "").replace("whatsapp:", "").replace("+", "")
            phone = re.sub(r"\D", "", phone or "")

    text_in = (payload.text or "").strip()
    if not text_in:
        text_in = (
            payload.message or payload.Body or
            ((payload.messageData or {}).get("textMessageData", {}) or {}).get("textMessage") or
            ""
        ).strip()

    if not phone:
        raise HTTPException(status_code=422, detail="user_phone is required")

    # ---- sesión ----
    sess = get_or_create_session(db, phone)
    slots = read_slots(sess)

    # ---- reset ----
    if _norm(text_in) == "reset":
        slots = {"_stage": "op"}
        write_slots(db, sess, slots)
        return out(welcome_reset_message(), slots=slots, stage="op")

    # ---- TASACIÓN (solo recopila y deriva) ----
    if (slots.get("_flow") == "tasacion") or _looks_like_tasacion_start(text_in):
        slots["_flow"] = "tasacion"
        resp = handle_tasacion(slots, text_in, phone)
        new_slots = resp.get("slots", slots)
        if isinstance(new_slots, dict):
            write_slots(db, sess, new_slots)
        return resp

    # ---- ATAJO por código/dirección/zona (igual que antes) ----
    try:
        prop = find_property_by_user_text(db, text_in)
    except Exception:
        prop = None
    if prop:
        # construimos respuesta tipo asesor + push a vendedor
        slots.setdefault("operacion", detect_operacion(text_in))  # por si vino implícito
        slots.setdefault("zona", prop.get("zona"))
        slots.setdefault("direccion", prop.get("direccion"))
        slots["inmueble_interes"] = prop.get("codigo")
        slots["_stage"] = "resumen"
        write_slots(db, sess, slots)
        humane = build_humane_property_reply(prop)
        vendor_text = build_vendor_summary(phone, prop, slots)
        return out(humane, vendor_push=True, vendor_message=vendor_text, slots=slots, stage="resumen")

    # ---- LOGICA SIMPLE (dos caminos) que pediste ----
    op = detect_operacion(text_in)
    if op:
        # Camino 1: si trae dirección exacta -> buscar ya
        if has_address_number(text_in):
            try:
                prop2 = find_property_by_user_text(db, text_in)
            except Exception:
                prop2 = None
            if prop2:
                slots["operacion"] = op
                slots["direccion"] = prop2.get("direccion")
                slots["zona"] = prop2.get("zona")
                slots["inmueble_interes"] = prop2.get("codigo")
                slots["_stage"] = "resumen"
                write_slots(db, sess, slots)
                humane = build_humane_property_reply(prop2)
                vendor_text = build_vendor_summary(phone, prop2, slots)
                return out(humane, vendor_push=True, vendor_message=vendor_text, slots=slots, stage="resumen")

        # Camino 2: no hay dirección -> enviar link y cerrar
        slots["operacion"] = op
        write_slots(db, sess, slots)
        link = f"{SITE_URL}/{'alquiler' if op=='alquiler' else 'venta'}"
        msg = (f"¡Perfecto! 👌 Te dejo nuestro catálogo de *{op}*: {link}\n"
               f"Si más tarde tenés una *dirección exacta* o *código* de propiedad, decímelo y te paso la info al instante.")
        return out(msg, slots=slots, stage="link")

    # Si no detectamos nada, inicio estándar
    ask = "¿La búsqueda es para *alquiler* o para *venta*?"
    slots["_stage"] = "op"
    write_slots(db, sess, slots)
    return out(ask, slots=slots, stage="op")
