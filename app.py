# app.py
import os
import re
import json
import unicodedata
from datetime import datetime
from typing import Optional, Dict, Any, List

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
# Soporte MySQL sin romper si ya usás mysql://
if DATABASE_URL.startswith("mysql://"):
    DATABASE_URL = DATABASE_URL.replace("mysql://", "mysql+pymysql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


# ===========================
# MODELOS
# ===========================
# Memoria de conversación por usuario (muy liviana)
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True)
    user_phone = Column(String(32), unique=True, index=True)
    slots_json = Column(Text, default="{}")
    updated_at = Column(DateTime, default=datetime.utcnow)

# Crea tabla si no existe (no toca tus otras tablas)
Base.metadata.create_all(bind=engine)


# ===========================
# UTILS NLP LIGERO
# ===========================
CODIGO_RE = re.compile(r"\b([A-Za-z]\d{2,4})\b")

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).strip().lower()

def detect_operacion(txt: str) -> Optional[str]:
    """
    Detecta operación por intención: alquiler / venta
    """
    t = _norm(txt)
    if any(k in t for k in (
        "alquiler", "alquilo", "alquilar", "renta", "rent", "quiero alquilar",
        "me gustaria alquilar", "busco alquiler", "para alquilar", "en alquiler"
    )):
        return "alquiler"
    if any(k in t for k in (
        "venta", "vendo", "vender", "comprar", "compro", "quiero comprar",
        "para comprar", "en venta"
    )):
        return "venta"
    return None

def parse_money(txt: str) -> Optional[int]:
    t = re.sub(r"[^\d]", "", txt or "")
    if not t:
        return None
    try:
        return int(t)
    except:
        return None

def parse_int(txt: str) -> Optional[int]:
    try:
        return int(re.sub(r"[^\d]", "", txt or ""))
    except:
        return None

def yes_no(txt: str) -> Optional[bool]:
    t = _norm(txt)
    if t in ("si", "sí", "s", "ok", "dale", "de acuerdo", "claro", "affirmative", "afirmativo"):
        return True
    if t in ("no", "n", "negativo"):
        return False
    return None

def has_address_number(txt: str) -> bool:
    # calle + número rápido (Av / Calle .. + 123)
    return bool(re.search(r"\b(\d{1,5})\b", txt or "")) and bool(re.search(r"[a-zA-Z]{3,}", txt or ""))


# ===========================
# HELPERS PROPIEDADES
# ===========================
def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def _ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()

def find_property_by_user_text(db: Session, text_in: str) -> Optional[Dict[str, Any]]:
    """
    Búsqueda flexible por:
    - Código (A101…)
    - Dirección aproximada (contiene calle y número)
    - Zona (si coincide por similitud, devuelve la más barata)
    """
    t_raw = text_in or ""
    t = _normalize(t_raw)

    # 1) Código tipo A101/B202
    m = CODIGO_RE.search(t_raw)
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

    # 2) Dirección con número
    if has_address_number(t_raw):
        res = db.execute(
            sql_text(
                "SELECT id, codigo, direccion, zona, precio, dormitorios, cochera "
                "FROM propiedades ORDER BY ABS(precio) ASC LIMIT 300"
            )
        )
        best = None
        best_score = 0.0
        for r in res.mappings().all():
            score = _ratio(t, _normalize(r["direccion"]))
            if score > best_score:
                best = dict(r)
                best_score = score
        if best and best_score >= 0.55:
            return best

    # 3) Coincidencia por zona
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

def list_properties_by_zone(db: Session, zona: str, limit: int = 5) -> List[dict]:
    """
    Devuelve hasta `limit` propiedades de la zona (ordenadas por precio asc).
    Nota: no filtro por 'operacion' ni 'tipo' para no romper tu esquema actual.
    Si tenés esas columnas, podés sumarlas fácil en el WHERE.
    """
    try:
        res = db.execute(
            sql_text(
                "SELECT id, codigo, direccion, zona, precio, dormitorios, cochera "
                "FROM propiedades WHERE zona=:z ORDER BY precio ASC LIMIT :lim"
            ),
            {"z": zona, "lim": limit}
        )
        return [dict(r) for r in res.mappings().all()]
    except Exception:
        return []

def format_zone_list(props: List[dict]) -> str:
    if not props:
        return ""
    lines = ["*Opciones en la zona:*"]
    for p in props:
        precio = int(p.get("precio") or 0)
        precio_txt = f"${precio:,}".replace(",", ".") if precio else "a consultar"
        cochera_txt = "con cochera" if (p.get("cochera") in (1, True)) else "sin cochera"
        lines.append(
            f"• *{p.get('codigo','N/D')}*: {p.get('direccion','N/D')} – {precio_txt}, "
            f"{p.get('dormitorios','N/D')} dorm, {cochera_txt}"
        )
    lines.append("\nSi te interesa alguna, decime el *código* o *dirección* y seguimos. "
                 "Si preferís, puedo ajustar la búsqueda.")
    return "\n".join(lines)

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
        "¿Querés que coordine una visita o te envío opciones parecidas en la zona?"
    )

def build_vendor_summary(user_phone: str, p: Optional[Dict[str, Any]], slots: Dict[str, Any]) -> str:
    prop_line = ""
    if p:
        prop_line = f"COD {p.get('codigo','N/D')} – {p.get('direccion','N/D')} ({p.get('zona','N/D')}). "
    return (
        f"Lead {user_phone}. {prop_line}"
        f"Operación: {slots.get('operacion','N/D')} | "
        f"Presup.: min {slots.get('presupuesto_min','N/D')} / max {slots.get('presupuesto_max','N/D')} | "
        f"Dorms: {slots.get('dormitorios','N/D')} | "
        f"Cochera: {slots.get('cochera','N/D')} | "
        f"Mascotas: {slots.get('mascotas','N/D')} | "
        f"Ingresos: {slots.get('ingresos','N/D')} | "
        f"Garantía: {slots.get('garantia','N/D')}."
    )


# ===========================
# FASTAPI
# ===========================
app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_or_create_session(db: Session, user_phone: str) -> ChatSession:
    s = db.query(ChatSession).filter(ChatSession.user_phone == user_phone).first()
    if not s:
        s = ChatSession(user_phone=user_phone, slots_json="{}")
        db.add(s)
        db.commit()
        db.refresh(s)
    return s

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
    )

def welcome_reset_message() -> str:
    return (
        "¡Arranquemos de nuevo! 😊\n"
        "Contame: ¿la búsqueda es para *alquiler* o para *venta*?\n"
        "Nota: cuando quieras reiniciar la conversación, escribí *reset* y empezamos de cero."
    )


# ===========================
# REQUEST / RESPONSE MODELS
# ===========================
class QualifyPayload(BaseModel):
    user_phone: str = Field(..., description="Número del cliente (solo dígitos, con código país)")
    text: str = Field(..., description="Mensaje del cliente")

class BotResponse(BaseModel):
    reply_text: str
    closing_text: str = ""
    vendor_push: bool = False
    vendor_message: Optional[str] = None


# ===========================
# ENDPOINTS
# ===========================
@app.get("/health")
def healthz():
    return {"ok": True}

@app.get("/debug")
def debug_vars():
    return {
        "DATABASE_URL_set": bool(DATABASE_URL),
    }

@app.post("/qualify", response_model=BotResponse)
def qualify(payload: QualifyPayload, db: Session = Depends(get_db)):
    phone = (payload.user_phone or "").strip()
    text_in = (payload.text or "").strip()

    if not phone:
        raise HTTPException(status_code=422, detail="user_phone is required")

    sess = get_or_create_session(db, phone)
    slots = read_slots(sess)
    stage = slots.get("_stage") or "op"

    # RESET
    if _norm(text_in) == "reset":
        slots = {"_stage": "op"}
        write_slots(db, sess, slots)
        return BotResponse(reply_text=welcome_reset_message())

    # Si trae una dirección/código en cualquier momento, respondemos ficha
    prop_now = None
    if text_in:
        prop_now = find_property_by_user_text(db, text_in)

    # ========= STAGE MACHINE =========
    # 1) Elegir operación
    if stage == "op":
        op = detect_operacion(text_in)
        if not op:
            ask = (
                "Gracias por contactarte con el área comercial de *Veglienzone Gestión Inmobiliaria* 🏡\n"
                "¿Cómo podemos ayudarte hoy?\n"
                "1- Alquileres\n2- Ventas\n3- Tasaciones\n\n"
                "_Tip: escribí *reset* en cualquier momento para reiniciar la conversación._"
            )
            return BotResponse(reply_text=ask)

        slots["operacion"] = op
        slots["_stage"] = "zona"
        write_slots(db, sess, slots)
        ask = (
            "¿Tenías una *dirección* o *link* exacto, o estás averiguando por alguna *zona* en particular?\n"
            "Si tenés dirección, escribila con calle y *número*; si no, decime la *zona/barrio*."
        )
        return BotResponse(reply_text=ask)

    # 2) Zona o dirección
    if stage == "zona":
        if has_address_number(text_in):
            slots["direccion"] = text_in
            zona_only = False
        else:
            slots["zona"] = text_in
            zona_only = True

        # Si es alquiler y solo zona: listamos opciones y seguimos al presupuesto
        extra = ""
        if zona_only and slots.get("zona"):
            props = list_properties_by_zone(db, slots["zona"], limit=5)
            if props:
                extra = "\n\n" + format_zone_list(props)

        slots["_stage"] = "pmin"
        write_slots(db, sess, slots)
        ask = "¿Cuál sería tu *presupuesto mínimo* aproximado (en ARS)?"
        return BotResponse(reply_text=ask + (extra or ""))

    # 3) Presupuesto mínimo
    if stage == "pmin":
        val = parse_money(text_in)
        if val is None:
            return BotResponse(reply_text="Decime un número aproximado para el *presupuesto mínimo* (ej.: 120000).")
        slots["presupuesto_min"] = val
        slots["_stage"] = "pmax"
        write_slots(db, sess, slots)
        return BotResponse(reply_text="¿Y tu *presupuesto máximo* (en ARS)?")

    # 4) Presupuesto máximo
    if stage == "pmax":
        val = parse_money(text_in)
        if val is None:
            return BotResponse(reply_text="No me quedó claro. ¿Cuál sería tu *presupuesto máximo* (ej.: 180000)?")
        slots["presupuesto_max"] = val
        slots["_stage"] = "dorm"
        write_slots(db, sess, slots)
        return BotResponse(reply_text="¿Cuántos *dormitorios* querés? (ej.: 2)")

    # 5) Dormitorios
    if stage == "dorm":
        val = parse_int(text_in)
        if val is None:
            return BotResponse(reply_text="Decime un número: ¿cuántos *dormitorios* buscás?")
        slots["dormitorios"] = val
        slots["_stage"] = "cochera"
        write_slots(db, sess, slots)
        return BotResponse(reply_text="¿Necesitás *cochera*? (sí/no)")

    # 6) Cochera
    if stage == "cochera":
        yn = yes_no(text_in)
        if yn is None:
            return BotResponse(reply_text="¿Te sirve con *cochera*? (sí/no)")
        slots["cochera"] = bool(yn)
        slots["_stage"] = "mascotas"
        write_slots(db, sess, slots)
        return BotResponse(reply_text="¿Tenés *mascotas* que debamos contemplar?")

    # 7) Mascotas
    if stage == "mascotas":
        t = _norm(text_in)
        if t.startswith("no"):
            slots["mascotas"] = "no"
        else:
            slots["mascotas"] = text_in or "si"
        slots["_stage"] = "ingresos"
        write_slots(db, sess, slots)
        return BotResponse(reply_text="¿Tenés *ingresos demostrables* que tripliquen el costo del alquiler?")

    # 8) Ingresos demostrables
    if stage == "ingresos":
        slots["ingresos"] = text_in or ""
        slots["_stage"] = "garantia"
        write_slots(db, sess, slots)
        return BotResponse(reply_text="¿Qué *tipo de garantía* podrías usar? (por ejemplo: *seguro de caución* de CABA, o *garantía propietaria*)")

    # 9) Garantía
    if stage == "garantia":
        slots["garantia"] = text_in or ""
        slots["_stage"] = "resumen"
        write_slots(db, sess, slots)

        # Resumen + Handoff
        resumen = [
            "Con esos datos ya puedo ayudarte mejor. Resumen de tu búsqueda:",
            f"• Operación: *{slots.get('operacion','N/D')}*",
            f"• Zona/Dirección: *{slots.get('zona') or slots.get('direccion','N/D')}*",
            f"• Presupuesto: *{slots.get('presupuesto_min')}* a *{slots.get('presupuesto_max')}*",
            f"• Dormitorios: *{slots.get('dormitorios','N/D')}*",
            f"• Cochera: *{ 'sí' if slots.get('cochera') else 'no' }*",
            f"• Mascotas: *{slots.get('mascotas','N/D')}*",
            f"• Ingresos: *{slots.get('ingresos','N/D')}*",
            f"• Garantía: *{slots.get('garantia','N/D')}*",
            "",
            "¿Querés que te derive con un *asesor humano* para coordinar visitas o ver más opciones por este WhatsApp?"
        ]
        vendor_msg = build_vendor_summary(phone, None, slots)
        return BotResponse(
            reply_text="\n".join(resumen),
            vendor_push=True,                      # Ya está calificado
            vendor_message=vendor_msg,
        )

    # 10) Fallback: si en cualquier momento aparece una propiedad concreta
    if prop_now:
        txt = build_humane_property_reply(prop_now)
        slots["prop_consulta"] = prop_now.get("codigo")
        write_slots(db, sess, slots)
        return BotResponse(reply_text=txt)

    # Fallback general
    slots["_stage"] = "op"
    write_slots(db, sess, slots)
    return BotResponse(
        reply_text="Para ayudarte mejor, ¿la búsqueda es para *alquiler* o para *venta*?"
    )
