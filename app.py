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
# DB CONFIG
# ===========================
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
# MODELOS
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
# HELPERS TEXTO / PARSING
# ===========================
def _norm(txt: str) -> str:
    if not txt:
        return ""
    txt = unicodedata.normalize("NFD", txt).encode("ascii", "ignore").decode("utf-8")
    return re.sub(r"\s+", " ", txt.lower()).strip()

def detect_operacion(txt: str) -> Optional[str]:
    t = _norm(txt)
    if any(k in t for k in ("alquiler", "alquilo", "alquilar", "renta", "rent", "en alquiler")):
        return "alquiler"
    if any(k in t for k in ("venta", "vendo", "vender", "comprar", "compro", "en venta")):
        return "venta"
    return None

def parse_money(txt: str) -> Optional[int]:
    t = _norm(txt)
    nums = re.findall(r"\d{1,3}(?:[.,]?\d{3})*|\d+", t)
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
    yes = ("si", "sí", "claro", "ok", "dale", "obvio")
    no = ("no", "nop", "nunca")
    if any(t == y or t.startswith(y + " ") for y in yes):
        return True
    if any(t == n or t.startswith(n + " ") for n in no):
        return False
    return None

def has_address_number(txt: str) -> bool:
    return bool(re.search(r"\b\d{1,5}\b", txt or ""))

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

# ===========================
# TASACIÓN – Helpers y Handler (solo recopila y deriva, sin DB)
# ===========================
_TAS_INICIO_KEYWORDS = (
    "tasacion", "tasación", "tasar", "quiero tasar",
    "hacer tasacion", "necesito tasacion", "valuar", "valuación", "valoracion"
)

def _looks_like_tasacion_start(txt: str) -> bool:
    t = _norm(txt)
    return any(k in t for k in _TAS_INICIO_KEYWORDS)

def _join_features(lst):
    if not lst:
        return "no informado"
    s = [x for x in lst if x]
    return ", ".join(s) if s else "no informado"

def _extract_features(txt: str):
    t = _norm(txt)
    feats = []
    if any(w in t for w in ("balcon", "balcón", "balkon")): feats.append("balcón")
    if "patio" in t: feats.append("patio")
    if "amenities" in t: feats.append("amenities")
    if "estudio" in t or "factibilidad" in t: feats.append("estudio factibilidad")
    if t in ("no", "ninguno", "ninguna", "ningunos"):
        return []
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

# ——— Salida que espera n8n ———
def out(reply_text: str, vendor_push: bool = False, vendor_message: Optional[str] = None, closing_text: Optional[str] = None, slots: Optional[dict] = None, stage: Optional[str] = None):
    # n8n solo lee estas 4 keys; slots/stage lo guardás en DB como ya hacías
    return {
        "reply_text": reply_text or "Perdón, estamos con un inconveniente técnico. ¿Podemos intentar de nuevo?",
        "vendor_push": bool(vendor_push),
        "vendor_message": vendor_message,
        "closing_text": closing_text,
        # extra (inocuo para n8n)
        "slots": slots,
        "stage": stage
    }

def handle_tasacion(slots: dict, incoming_text: str, user_phone: str):
    slots = _ensure_tas_slots(slots)
    step = slots.get("_tas_step", "op")
    text = incoming_text or ""

    if step == "op":
        op = _norm(text)
        if "venta" in op:
            slots["tas_operacion"] = "venta"; slots["_tas_step"] = "prop"
        elif "alquiler" in op or "renta" in op:
            slots["tas_operacion"] = "alquiler"; slots["_tas_step"] = "prop"
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
# FIND PROPERTY / REPLIES — (igual)
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

    m = CODIGO_RE.search(text_in or "")
    if m:
        codigo = m.group(1).upper()
        res = db.execute(
            sql_text(
                "SELECT id, codigo, direccion, zona, precio, dormitorios, cochera "
                "FROM propiedades WHERE codigo=:codigo LIMIT 1"
            ),
            {"codigo": codigo}
        )
        row = res.mappings().first()
        if row:
            return dict(row)

    if _looks_like_address(t):
        street, number, words = _address_tokens(t)
        if street and number:
            like = f"%{street.split()[0]}%{number}%"
            res = db.execute(
                sql_text(
                    "SELECT id, codigo, direccion, zona, precio, dormitorios, cochera "
                    "FROM propiedades WHERE direccion LIKE :like LIMIT 1"
                ),
                {"like": like}
            )
            row = res.mappings().first()
            if row:
                return dict(row)

        like_parts = [w for w in words if w not in {"calle", "avenida"}][:2]
        if like_parts:
            where = " AND ".join([f"direccion LIKE :w{i}" for i in range(len(like_parts))])
            params = {f"w{i}": f"%{w}%" for i, w in enumerate(like_parts)}
            res = db.execute(
                sql_text(
                    f"SELECT id, codigo, direccion, zona, precio, dormitorios, cochera "
                    f"FROM propiedades WHERE {where} LIMIT 3"
                ),
                params
            )
            rows = res.mappings().all()
            if rows:
                best = max(rows, key=lambda r: _ratio(" ".join(words), r["direccion"]))
                return dict(best)

    zonas = db.execute(sql_text("SELECT DISTINCT zona FROM propiedades")).scalars().all()
    for z in zonas:
        if _normalize(z) in t:
            res = db.execute(
                sql_text(
                    "SELECT id, codigo, direccion, zona, precio, dormitorios, cochera "
                    "FROM propiedades WHERE zona=:z ORDER BY precio ASC LIMIT 1"
                ),
                {"z": z}
            )
            row = res.mappings().first()
            if row:
                return dict(row)

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
        f"Dorms: {slots.get('dormitorios','N/D')} | Cochera: {slots.get('cochera','N/D')}.\n"
        "Pedir confirmación para visita o enviar comparables."
    )

# ===========================
# SLOTS / SESIONES
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

def have_minimum_for_vendor(slots: Dict[str, Any]) -> bool:
    return bool(
        slots.get("operacion")
        and (slots.get("zona") or slots.get("direccion"))
        and slots.get("presupuesto_min")
        and slots.get("presupuesto_max")
        and (slots.get("dormitorios") is not None)
    )

def welcome_reset_message() -> str:
    return (
        "¡Arranquemos de nuevo! 😊\n"
        "Contame: ¿la búsqueda es para *alquiler* o para *venta*?\n"
        "Tip: cuando quieras reiniciar la conversación, escribí *reset* y empezamos de cero."
    )

# ===========================
# MODELOS DE ENTRADA (sin romper tu flujo)
# ===========================
class QualifyPayload(BaseModel):
    # user_phone puede venir o no (si no, lo deducimos de chatId)
    user_phone: Optional[str] = None
    text: Optional[str] = ""
    # alternativos que tu n8n puede mandar
    chatId: Optional[str] = None
    message: Optional[str] = None
    From: Optional[str] = None
    waId: Optional[str] = None
    Body: Optional[str] = None
    senderData: Optional[dict] = None
    messageData: Optional[dict] = None

# ===========================
# ENDPOINTS
# ===========================
@app.get("/healthz")
def healthz():
    return {"ok": True, "ts": now_iso()}

@app.post("/qualify")
def qualify(payload: QualifyPayload, db: Session = Depends(get_db)):
    # ——— Entrada flexible (no rompe n8n) ———
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
        # único caso que devuelve 422 (igual que antes)
        raise HTTPException(status_code=422, detail="user_phone is required")

    # ——— Sesión / slots ———
    sess = get_or_create_session(db, phone)
    slots = read_slots(sess)
    stage = slots.get("_stage") or "op"

    # ——— RESET ———
    if _norm(text_in) == "reset":
        slots = {"_stage": "op"}
        write_slots(db, sess, slots)
        msg = welcome_reset_message()
        return out(msg, slots=slots, stage="op")

    # ——— TASACIÓN (solo recopila y deriva) ———
    if (slots.get("_flow") == "tasacion") or _looks_like_tasacion_start(text_in):
        slots["_flow"] = "tasacion"
        resp = handle_tasacion(slots, text_in, phone)  # ya devuelve reply_text/… via out()
        # persistimos avances si vinieron
        new_slots = resp.get("slots", slots)
        if isinstance(new_slots, dict):
            write_slots(db, sess, new_slots)
        return resp

    # ——— ATAJO asesor: código/dirección/zona ———
    try:
        prop = find_property_by_user_text(db, text_in)
    except Exception:
        prop = None

    if prop:
        slots.setdefault("zona", prop.get("zona"))
        slots.setdefault("direccion", prop.get("direccion"))
        slots["inmueble_interes"] = prop.get("codigo")
        slots["_stage"] = "resumen"
        write_slots(db, sess, slots)

        humane = build_humane_property_reply(prop)
        vendor_text = build_vendor_summary(phone, prop, slots)

        return out(humane, vendor_push=True, vendor_message=vendor_text, slots=slots, stage="resumen")

    # ——— FORM conversacional (SIN CAMBIOS) ———
    if stage in ("op", "operacion"):
        if slots.get("operacion"):
            slots["_stage"] = "zona"
            write_slots(db, sess, slots)
            q = "Perfecto. ¿En qué *zona* o *dirección exacta* estás interesado/a? (calle y número si lo tenés)"
            return out(q, slots=slots, stage="zona")

        op = detect_operacion(text_in)
        if op:
            slots["operacion"] = op
            slots["_stage"] = "zona"
            write_slots(db, sess, slots)
            q = f"Perfecto, {op}. ¿En qué *zona* o *dirección exacta* estás buscando? (calle y número si lo tenés)"
            return out(q, slots=slots, stage="zona")

        ask = "¿La búsqueda es para *alquiler* o para *venta*?"
        return out(ask, slots=slots, stage="op")

    if stage == "zona":
        if has_address_number(text_in):
            slots["direccion"] = text_in
        else:
            slots["zona"] = text_in
        slots["_stage"] = "pmin"
        write_slots(db, sess, slots)
        q = "¿Cuál sería tu *presupuesto mínimo* aproximado (en ARS)?"
        return out(q, slots=slots, stage="pmin")

    if stage == "pmin":
        val = parse_money(text_in)
        if val is None:
            q = "No me quedó claro. Decime un número aproximado para el *presupuesto mínimo* (en ARS)."
            return out(q, slots=slots, stage="pmin")
        slots["presupuesto_min"] = val
        slots["_stage"] = "pmax"
        write_slots(db, sess, slots)
        q = "Genial. ¿Y el *presupuesto máximo* aproximado (en ARS)?"
        return out(q, slots=slots, stage="pmax")

    if stage == "pmax":
        val = parse_money(text_in)
        if val is None:
            q = "Entendido. ¿Podés indicarme un número para el *presupuesto máximo* (en ARS)?"
            return out(q, slots=slots, stage="pmax")
        slots["presupuesto_max"] = val
        slots["_stage"] = "dorm"
        write_slots(db, sess, slots)
        q = "Para afinar la búsqueda: ¿Cuántos *dormitorios* te gustaría?"
        return out(q, slots=slots, stage="dorm")

    if stage == "dorm":
        n = parse_int(text_in)
        if n is None:
            q = "¿Cuántos *dormitorios* querés? (ej.: 2)"
            return out(q, slots=slots, stage="dorm")
        slots["dormitorios"] = n
        slots["_stage"] = "cochera"
        write_slots(db, sess, slots)
        q = "¿Necesitás *cochera*? (sí/no)"
        return out(q, slots=slots, stage="cochera")

    if stage == "cochera":
        yn = yes_no(text_in)
        if yn is None:
            q = "¿Te sirve con *cochera*? (sí/no)"
            return out(q, slots=slots, stage="cochera")
        slots["cochera"] = bool(yn)
        slots["_stage"] = "mascotas"
        write_slots(db, sess, slots)
        q = "¿Tenés *mascotas* que debamos contemplar?"
        return out(q, slots=slots, stage="mascotas")

    if stage == "mascotas":
        t = _norm(text_in)
        tiene = None
        desc = None
        if any(w in t for w in ("si", "sí", "perro", "gato", "mascota", "perros", "gatos")):
            tiene = True; desc = text_in
        elif t.startswith("no") or t == "no":
            tiene = False

        if tiene is None:
            q = "¿Tenés *mascotas*? (Podés decirme *no* o contame: perros, gatos, etc.)"
            return out(q, slots=slots, stage="mascotas")

        slots["mascotas"] = desc if tiene else "no"
        if not slots.get("direccion"):
            slots["_stage"] = "direccion"
            write_slots(db, sess, slots)
            q = "¿Tenés una *dirección exacta*? (calle y número) Si no, decime *no tengo* y sigo."
            return out(q, slots=slots, stage="direccion")

        slots["_stage"] = "resumen"
        write_slots(db, sess, slots)

    if stage == "direccion":
        if has_address_number(text_in):
            slots["direccion"] = text_in
        slots["_stage"] = "resumen"
        write_slots(db, sess, slots)

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

        header = f"Perfecto 👍\n{op.capitalize()} en {zona if zona!='zona a definir' else 'una zona a definir'}"
        if d:
            header += f" (dirección: {d})"
        if rango:
            header += f". Presupuesto: {rango}."
        header += f" {dorm} dorm, {coch}, {masc}."

        follow = "¿Querés que te envíe opciones que coincidan, o querés ajustar algún dato?"
        text_out = f"{header}\n\n{follow}"
        return out(text_out, slots=slots, stage="resumen")

    # Fallback
    slots["_stage"] = "op"
    write_slots(db, sess, slots)
    ask = "¿La búsqueda es para *alquiler* o para *venta*?"
    return out(ask, slots=slots, stage="op")
