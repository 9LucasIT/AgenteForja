# app.py
import os
import re
import json
import unicodedata
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from pydantic import ConfigDict
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, text as sql_text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# Extras para manejo de links y fetch
from urllib.parse import urlparse, parse_qs
import httpx

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
# MODELOS (solo columnas existentes)
# chat_session: id, user_phone, slots_json, created_at, updated_at
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
# FIND PROPERTY / REPLIES (atajo de asesor)
# ===========================
STREET_HINTS = [
    "calle", "c/", "av", "avenida", "pasaje", "pas", "pje", "ruta", "rn", "rp",
    "boulevard", "bvard", "bv", "diagonal", "diag", "esquina", "entre", "altura"
]
CODIGO_RE = re.compile(r"\b([A-Z]\d{3,5})\b", re.IGNORECASE)

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

# --- Column discovery (para no romper si el schema difiere) ---
_PROP_COLS_CACHE: Optional[List[str]] = None

def get_property_columns(db: Session) -> List[str]:
    global _PROP_COLS_CACHE
    if _PROP_COLS_CACHE is not None:
        return _PROP_COLS_CACHE
    cols = []
    try:
        res = db.execute(sql_text("SHOW COLUMNS FROM propiedades"))
        cols = [r[0] for r in res.fetchall()]
    except Exception:
        cols = []
    _PROP_COLS_CACHE = cols
    return cols

def select_clause_for_props(cols: List[str]) -> str:
    base = ["id", "codigo", "direccion", "zona", "precio", "dormitorios", "cochera"]
    optional = ["operacion", "tipo_operacion", "precio_venta", "precio_alquiler", "ambientes", "tipo_propiedad", "total_construido"]
    selected = [c for c in base if c in cols]
    selected += [c for c in optional if c in cols]
    if not selected:
        selected = ["id", "codigo", "direccion", "zona"]
    return ", ".join(selected)

def get_prop_operacion(p: Dict[str, Any]) -> Optional[str]:
    for k in ("operacion", "tipo_operacion"):
        v = p.get(k)
        if isinstance(v, str) and v.strip():
            t = detect_operacion(v)
            if t:
                return t
            vs = _norm(v)
            if vs in ("venta", "alquiler"):
                return vs
        elif isinstance(v, (int, bool)):
            return "venta" if int(v) == 1 else "alquiler"
    if p.get("precio_venta") and not p.get("precio_alquiler"):
        return "venta"
    if p.get("precio_alquiler") and not p.get("precio_venta"):
        return "alquiler"
    return None

# --- Helpers de links ---
URL_RE = re.compile(r"https?://[^\s>]+", re.I)
KNOWN_DOMAINS = {"zonaprop.com.ar", "argenprop.com", "argenprop.com.ar", "veglienzone.com.ar", "veglienzone.com"}

def _extract_first_url(text: str) -> Optional[str]:
    if not text:
        return None
    m = URL_RE.search(text)
    return m.group(0) if m else None

def _domain(host: Optional[str]) -> str:
    if not host:
        return ""
    host = host.lower()
    parts = host.split(".")
    return ".".join(parts[-3:]) if len(parts) >= 3 else host

def _try_extract_codigo_from_url(url: str) -> Optional[str]:
    try:
        u = urlparse(url)
        qs = parse_qs(u.query or "")
        for key in ("codigo", "cod", "c", "id"):
            if key in qs and qs[key]:
                return str(qs[key][0]).strip().upper()
        m = CODIGO_RE.search(u.path or "")
        if m:
            return m.group(1).upper()
    except Exception:
        pass
    return None

def _fetch_page_title_or_og(url: str) -> Optional[str]:
    try:
        r = httpx.get(
            url,
            timeout=4.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Veglienzone-Agent)"},
        )
        if r.status_code >= 400 or not r.text:
            return None
        html = r.text
        m = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
        if m:
            return m.group(1).strip()
        m = re.search(r"<title>(.*?)</title>", html, re.I | re.S)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    except Exception:
        pass
    return None

def _match_address_like(db: Session, source_text: str, cols: List[str], sel: str):
    street, number, words = _address_tokens(source_text)

    if street and number and "direccion" in cols:
        like = f"%{street.split()[0]}%{number}%"
        res = db.execute(
            sql_text(f"SELECT {sel} FROM propiedades WHERE direccion LIKE :like LIMIT 1"),
            {"like": like}
        )
        row = res.mappings().first()
        if row:
            return row

    if "direccion" in cols:
        like_parts = [w for w in words if w not in {"calle", "avenida"}][:2]
        if like_parts:
            where = " AND ".join([f"direccion LIKE :w{i}" for i in range(len(like_parts))])
            params = {f"w{i}": f"%{w}%" for i, w in enumerate(like_parts)}
            res = db.execute(
                sql_text(f"SELECT {sel} FROM propiedades WHERE {where} LIMIT 3"),
                params
            )
            rows = res.mappings().all()
            if rows:
                best = max(rows, key=lambda r: _ratio(" ".join(words), r.get("direccion", "")))
                return best

    return None

def find_property_by_user_text(db: Session, text_in: str) -> Optional[Dict[str, Any]]:
    """Versión extendida: también intenta con URL y título si viene un link."""
    t = _normalize(text_in or "")
    cols = get_property_columns(db)
    sel = select_clause_for_props(cols)

    # 0) Si viene URL, probamos directo
    url = _extract_first_url(text_in or "")
    if url:
        try:
            u = urlparse(url)
            host = _domain(u.hostname)
            if any(host.endswith(kd) for kd in KNOWN_DOMAINS):
                codigo = _try_extract_codigo_from_url(url)
                if codigo:
                    res = db.execute(
                        sql_text(f"SELECT {sel} FROM propiedades WHERE UPPER(codigo)=:codigo LIMIT 1"),
                        {"codigo": codigo.upper()}
                    )
                    row = res.mappings().first()
                    if row:
                        return dict(row)

                # sin código → og:title/title
                title = _fetch_page_title_or_og(url)
                if title:
                    maybe = _match_address_like(db, title, cols, sel)
                    if maybe:
                        return dict(maybe)
        except Exception:
            pass

    # 1) Código tipo A101/B202 en texto
    m = CODIGO_RE.search(text_in or "")
    if m:
        codigo = m.group(1).upper()
        res = db.execute(
            sql_text(f"SELECT {sel} FROM propiedades WHERE codigo=:codigo LIMIT 1"),
            {"codigo": codigo}
        )
        row = res.mappings().first()
        if row:
            return dict(row)

    # 2) Dirección aparente
    if _looks_like_address(t):
        maybe = _match_address_like(db, text_in, cols, sel)
        if maybe:
            return dict(maybe)

    # 3) Zona exacta
    if "zona" in cols:
        zonas = db.execute(sql_text("SELECT DISTINCT zona FROM propiedades")).scalars().all()
        for z in zonas:
            if _normalize(z) in t:
                res = db.execute(
                    sql_text(f"SELECT {sel} FROM propiedades WHERE zona=:z ORDER BY precio ASC LIMIT 1"),
                    {"z": z}
                )
                row = res.mappings().first()
                if row:
                    return dict(row)

    return None

def build_humane_property_reply(p: Dict[str, Any]) -> str:
    cochera_txt = "con cochera" if (p.get("cochera") in (1, True, "1", "true", "sí", "si")) else "sin cochera"
    precio = None
    if p.get("precio"):
        try:
            precio = int(float(p["precio"]))
        except Exception:
            pass
    if not precio and p.get("precio_alquiler"):
        try:
            precio = int(float(p["precio_alquiler"]))
        except Exception:
            pass
    if not precio and p.get("precio_venta"):
        try:
            precio = int(float(p["precio_venta"]))
        except Exception:
            pass

    precio_txt = f"${precio:,}".replace(",", ".") if precio else "a consultar"
    op = get_prop_operacion(p)
    op_txt = f"• Operación: {op}\n" if op else ""

    return (
        "¡Genial! Sobre esa propiedad:\n"
        f"• Código: {p.get('codigo', 'N/D')}\n"
        f"• Dirección: {p.get('direccion','N/D')} ({p.get('zona','N/D')})\n"
        f"{op_txt}"
        f"• Precio: {precio_txt}\n"
        f"• Dormitorios: {p.get('dormitorios','N/D')} – {cochera_txt}\n\n"
        "¿Querés que coordinemos una visita o te envío opciones parecidas en la zona?"
    )

def build_vendor_summary(user_phone: str, p: Dict[str, Any], slots: Dict[str, Any]) -> str:
    return (
        f"Lead {user_phone} consultó por COD {p.get('codigo','N/D')} – {p.get('direccion','N/D')} ({p.get('zona','N/D')}).\n"
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
# REQUEST / RESPONSE MODELS
# ===========================
class QualifyPayload(BaseModel):
    user_phone: Optional[str] = Field(None, description="Número del cliente (solo dígitos, con código país)")
    text: Optional[str] = Field("", description="Mensaje del cliente")
    message_id: Optional[str] = Field(None, description="ID mensaje (opcional)")
    model_config = ConfigDict(extra="allow")  # permite campos extra (Green/Twilio)

class BotResponse(BaseModel):
    text: str
    next_question: Optional[str] = None
    updates: Dict[str, Any] = {}
    vendor_push: Optional[bool] = None
    vendor_message: Optional[str] = None

# ===========================
# NORMALIZACIÓN DE PAYLOADS AJENOS (Green/Twilio)
# ===========================
def _extract_from_any_payload(p: QualifyPayload) -> Dict[str, str]:
    """
    Acepta formato 'propio' (user_phone, text, message_id) o crudos de Green/Twilio:
    - Green: chatId, message, senderData.sender, messageData.textMessageData.textMessage
    - Twilio: From, Body, SmsSid/MessageSid
    Devuelve dict con { user_phone, text, message_id }.
    """
    phone = (p.user_phone or "").strip()
    text  = (p.text or "").strip()
    mid   = (p.message_id or "").strip()

    x = getattr(p, "model_extra", {}) or {}

    # PHONE
    candidates_phone = [
        phone,
        str(x.get("chatId", "")),
        str(x.get("waId", "")),
        str(x.get("From", "")),
        str(x.get("senderData", {}).get("sender", "")),
    ]
    for cand in candidates_phone:
        if cand:
            cand = cand.replace("whatsapp:", "").replace("@c.us", "")
            cand = re.sub(r"\D", "", cand)
            if cand:
                phone = cand
                break

    # TEXT
    candidates_text = [
        text,
        str(x.get("message", "")),
        str(x.get("Body", "")),
        str(x.get("messageData", {}).get("textMessageData", {}).get("textMessage", "")),
    ]
    for cand in candidates_text:
        if cand and cand.strip():
            text = cand.strip()
            break

    # MESSAGE ID
    candidates_mid = [
        mid,
        str(x.get("idMessage", "")),
        str(x.get("MessageSid", "")),
        str(x.get("SmsSid", "")),
    ]
    for cand in candidates_mid:
        if cand and cand.strip():
            mid = cand.strip()
            break

    return {"user_phone": phone, "text": text, "message_id": mid}

# ===========================
# ENDPOINTS
# ===========================
@app.get("/healthz")
def healthz():
    return {"ok": True, "ts": now_iso()}

def _enforce_operacion_with_property(slots: Dict[str, Any], prop: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Devuelve (error_msg, operacion_final).
    - Si hay mismatch entre slots['operacion'] y la propiedad, retorna error_msg (texto para el usuario).
    - Si no había operacion definida, la setea según la propiedad.
    """
    prop_op = get_prop_operacion(prop)
    user_op = slots.get("operacion")

    if prop_op and user_op and prop_op != user_op:
        return (
            f"La propiedad es de *{prop_op}*, pero tu búsqueda venía como *{user_op}*.\n"
            f"¿Querés cambiar la búsqueda a *{prop_op}* para avanzar con esta ficha, o preferís que te muestre opciones de *{user_op}*?",
            None
        )
    final_op = user_op or prop_op
    return (None, final_op)

@app.post("/qualify", response_model=BotResponse)
def qualify(payload: QualifyPayload, db: Session = Depends(get_db)):
    # Normaliza cualquier payload entrante (nuestro formato o Green/Twilio)
    norm = _extract_from_any_payload(payload)
    phone = norm["user_phone"]
    text_in = norm["text"]
    message_id = norm["message_id"]

    if not phone:
        return BotResponse(
            text="No pude leer tu número desde el mensaje. ¿Podés reenviarme el mensaje o escribir *reset* para empezar?",
            next_question="Contame: ¿la búsqueda es para *alquiler* o para *venta*?",
            updates={"error": "missing_user_phone"},
            vendor_push=False
        )

    sess = get_or_create_session(db, phone)
    slots = read_slots(sess)
    stage = slots.get("_stage") or "op"

    # 0) RESET
    if _norm(text_in) == "reset":
        slots = {"_stage": "op"}
        write_slots(db, sess, slots)
        msg = welcome_reset_message()
        return BotResponse(text=msg, next_question=msg, updates={"slots": slots, "stage": "op"}, vendor_push=False)

    # 1) ATAJO: consulta por propiedad (código, dirección/zona o LINK)
    try:
        prop = find_property_by_user_text(db, text_in)
    except Exception:
        prop = None

    if prop:
        # validar / fijar operacion en función de la propiedad
        err, op_final = _enforce_operacion_with_property(slots, prop)
        if err:
            # mismatch → no empujamos al vendedor todavía
            slots["_stage"] = "op_confirm_from_link"
            slots["inmueble_interes"] = prop.get("codigo")
            slots["direccion"] = slots.get("direccion") or prop.get("direccion")
            write_slots(db, sess, slots)
            return BotResponse(
                text=err,
                next_question="Decime si cambiamos la búsqueda a esa operación o preferís seguir con la original.",
                updates={"slots": slots, "stage": "op_confirm_from_link"},
                vendor_push=False,
                vendor_message=None
            )

        # fijamos operacion si venía vacía y seguimos
        if op_final:
            slots["operacion"] = op_final

        slots.setdefault("zona", prop.get("zona"))
        slots.setdefault("direccion", prop.get("direccion"))
        slots["inmueble_interes"] = prop.get("codigo")
        slots["_stage"] = "resumen"
        write_slots(db, sess, slots)

        humane = build_humane_property_reply(prop)
        vendor_text = build_vendor_summary(phone, prop, slots)

        return BotResponse(
            text=humane,
            next_question=None,
            updates={"slots": slots, "stage": "resumen"},
            vendor_push=True,              # n8n: dispara mensaje al vendedor
            vendor_message=vendor_text
        )

    # 2) FLUJO POR ETAPAS (formulario conversacional humanizado)
    if stage in ("op", "operacion"):
        if slots.get("operacion"):
            slots["_stage"] = "zona"
            write_slots(db, sess, slots)
            q = "Perfecto. ¿En qué *zona* o *dirección exacta* estás interesado/a? (calle y número si lo tenés)"
            return BotResponse(text=q, next_question=q, updates={"slots": slots, "stage": "zona"}, vendor_push=have_minimum_for_vendor(slots))

        op = detect_operacion(text_in)
        if op:
            slots["operacion"] = op
            slots["_stage"] = "zona"
            write_slots(db, sess, slots)
            q = f"Perfecto, {op}. ¿En qué *zona* o *dirección exacta* estás buscando? (calle y número si lo tenés)"
            return BotResponse(text=q, next_question=q, updates={"slots": slots, "stage": "zona"}, vendor_push=have_minimum_for_vendor(slots))

        ask = "¿La búsqueda es para *alquiler* o para *venta*?"
        return BotResponse(text=ask, next_question=ask, updates={"stage": "op"}, vendor_push=False)

    if stage == "zona":
        if has_address_number(text_in):
            slots["direccion"] = text_in
        else:
            slots["zona"] = text_in
        slots["_stage"] = "pmin"
        write_slots(db, sess, slots)
        q = "¿Cuál sería tu *presupuesto mínimo* aproximado (en ARS)?"
        return BotResponse(text=q, next_question=q, updates={"slots": slots, "stage": "pmin"}, vendor_push=have_minimum_for_vendor(slots))

    if stage == "pmin":
        val = parse_money(text_in)
        if val is None:
            q = "No me quedó claro. Decime un número aproximado para el *presupuesto mínimo* (en ARS)."
            return BotResponse(text=q, next_question=q, updates={"stage": "pmin"}, vendor_push=have_minimum_for_vendor(slots))
        slots["presupuesto_min"] = val
        slots["_stage"] = "pmax"
        write_slots(db, sess, slots)
        q = "Genial. ¿Y el *presupuesto máximo* aproximado (en ARS)?"
        return BotResponse(text=q, next_question=q, updates={"slots": slots, "stage": "pmax"}, vendor_push=have_minimum_for_vendor(slots))

    if stage == "pmax":
        val = parse_money(text_in)
        if val is None:
            q = "Entendido. ¿Podés indicarme un número para el *presupuesto máximo* (en ARS)?"
            return BotResponse(text=q, next_question=q, updates={"stage": "pmax"}, vendor_push=have_minimum_for_vendor(slots))
        slots["presupuesto_max"] = val
        slots["_stage"] = "dorm"
        write_slots(db, sess, slots)
        q = "Para afinar la búsqueda: ¿Cuántos *dormitorios* te gustaría?"
        return BotResponse(text=q, next_question=q, updates={"slots": slots, "stage": "dorm"}, vendor_push=have_minimum_for_vendor(slots))

    if stage == "dorm":
        n = parse_int(text_in)
        if n is None:
            q = "¿Cuántos *dormitorios* querés? (ej.: 2)"
            return BotResponse(text=q, next_question=q, updates={"stage": "dorm"}, vendor_push=have_minimum_for_vendor(slots))
        slots["dormitorios"] = n
        slots["_stage"] = "cochera"
        write_slots(db, sess, slots)
        q = "¿Necesitás *cochera*? (sí/no)"
        return BotResponse(text=q, next_question=q, updates={"slots": slots, "stage": "cochera"}, vendor_push=have_minimum_for_vendor(slots))

    if stage == "cochera":
        yn = yes_no(text_in)
        if yn is None:
            q = "¿Te sirve con *cochera*? (sí/no)"
            return BotResponse(text=q, next_question=q, updates={"stage": "cochera"}, vendor_push=have_minimum_for_vendor(slots))
        slots["cochera"] = bool(yn)
        slots["_stage"] = "mascotas"
        write_slots(db, sess, slots)
        q = "¿Tenés *mascotas* que debamos contemplar?"
        return BotResponse(text=q, next_question=q, updates={"slots": slots, "stage": "mascotas"}, vendor_push=have_minimum_for_vendor(slots))

    if stage == "mascotas":
        t = _norm(text_in)
        tiene = None
        desc = None
        if any(w in t for w in ("si", "sí", "perro", "gato", "mascota", "perros", "gatos")):
            tiene = True
            desc = text_in
        elif t.startswith("no") or t == "no":
            tiene = False

        if tiene is None:
            q = "¿Tenés *mascotas*? (Podés decirme *no* o contame: perros, gatos, etc.)"
            return BotResponse(text=q, next_question=q, updates={"stage": "mascotas"}, vendor_push=have_minimum_for_vendor(slots))

        slots["mascotas"] = desc if tiene else "no"
        if not slots.get("direccion"):
            slots["_stage"] = "direccion"
            write_slots(db, sess, slots)
            q = "¿Tenés una *dirección exacta*? (calle y número) Si no, decime *no tengo* y sigo."
            return BotResponse(text=q, next_question=q, updates={"slots": slots, "stage": "direccion"}, vendor_push=have_minimum_for_vendor(slots))

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

        push_guard = have_minimum_for_vendor(slots)
        follow = "¿Querés que te envíe opciones que coincidan, o querés ajustar algún dato?"
        text_out = f"{header}\n\n{follow}"
        return BotResponse(
            text=text_out,
            next_question=follow,
            updates={"slots": slots, "stage": "resumen"},
            vendor_push=push_guard,
            vendor_message=None
        )

    # Fallback: reiniciar a op
    slots["_stage"] = "op"
    write_slots(db, sess, slots)
    ask = "¿La búsqueda es para *alquiler* o para *venta*?"
    return BotResponse(text=ask, next_question=ask, updates={"stage": "op"}, vendor_push=False)
