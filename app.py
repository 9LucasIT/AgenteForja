import os
import re
import json
import logging
import unicodedata
from difflib import SequenceMatcher
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

# ====================================================================
# CONFIG BÁSICA
# ====================================================================

logger = logging.getLogger("agente_forja")
logging.basicConfig(level=logging.INFO)

SITE_URL = os.getenv("SITE_URL", "https://www.fincasdeleste.com.uy/").rstrip("/")

# MySQL
DATABASE_URL = os.getenv("DATABASE_URL", "") or os.getenv("MYSQL_URL", "")
MYSQL_HOST = os.getenv("MYSQLHOST") or os.getenv("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQLPORT") or os.getenv("MYSQL_PORT") or "3306")
MYSQL_USER = os.getenv("MYSQLUSER") or os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQLPASSWORD") or os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQLDATABASE") or os.getenv("MYSQL_DATABASE")
MYSQL_TABLE = os.getenv("MYSQL_TABLE", "propiedades")

# Green API
GREEN_API_URL = os.getenv("GREEN_API_URL", "https://api.green-api.com").rstrip("/")
GREEN_INSTANCE_ID = os.getenv("GREEN_INSTANCE_ID") or os.getenv("GREEN_API_INSTANCE_ID")
GREEN_API_TOKEN = os.getenv("GREEN_API_TOKEN") or os.getenv("GREEN_TOKEN")

# Chat del asesor
VENDOR_CHAT_ID = (os.getenv("VENDOR_CHAT_ID") or "").strip()  # ej: "5493412654593@c.us"

# IA - Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama-3.1-8b-instant")

if GROQ_API_KEY:
    groq_client: Optional[Groq] = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

# Estado en memoria
STATE: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="Agente Forja IA libre", version="2025-12-10")


# ====================================================================
# MODELOS I/O
# ====================================================================

class QualifyIn(BaseModel):
    chatId: str
    message: Optional[str] = ""
    isFromMe: Optional[bool] = False
    senderName: Optional[str] = ""


class QualifyOut(BaseModel):
    reply_text: str
    vendor_push: bool = False
    vendor_message: str = ""
    closing_text: str = ""


# ====================================================================
# HELPERS TEXTO
# ====================================================================

def _strip_accents(s: str) -> str:
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _s(v) -> str:
    try:
        if v is None:
            return ""
        return str(v).strip()
    except Exception:
        return ""


def _say_menu() -> str:
    return (
        "¡Hola! 👋 Soy el asistente virtual de *Inmobiliaria Finca del Este*.\n"
        "Contame con palabras qué estás necesitando y te voy guiando, "
        "pero si querés podés orientarte con estas opciones:\n\n"
        "1️⃣ *Alquileres*\n"
        "2️⃣ *Ventas*\n"
        "3️⃣ *Tasaciones*\n\n"
        "📝 Escribí con tus palabras, o mandá el número de la opción.\n"
        "🔄 En cualquier momento podés escribir *\"reset\"* para empezar de cero."
    )


# ====================================================================
# DB HELPERS
# ====================================================================

try:
    import pymysql
    from pymysql.cursors import DictCursor

    PYM_AVAILABLE = True
except Exception:
    PYM_AVAILABLE = False


def _parse_db_url(url: str):
    if not url:
        return None
    u = urlparse(url)
    return (u.hostname, u.port or 3306, u.username, u.password, (u.path or "").lstrip("/"))


def _db_params():
    if DATABASE_URL:
        h, p, u, pwd, db = _parse_db_url(DATABASE_URL)
        return {"host": h, "port": p, "user": u, "password": pwd, "database": db}
    return {
        "host": MYSQL_HOST,
        "port": MYSQL_PORT,
        "user": MYSQL_USER,
        "password": MYSQL_PASSWORD,
        "database": MYSQL_DB,
    }


def _safe_connect():
    if not PYM_AVAILABLE:
        return None
    params = _db_params()
    if not params.get("host") or not params.get("user") or not params.get("database"):
        return None
    try:
        return pymysql.connect(
            host=params["host"],
            port=int(params["port"]),
            user=params["user"],
            password=params["password"],
            database=params["database"],
            cursorclass=DictCursor,
            autocommit=True,
            charset="utf8mb4",
        )
    except Exception as e:
        logger.error(f"Error conectando a MySQL: {e}")
        return None


def _build_like_patterns(raw: str) -> List[str]:
    text = raw.strip()
    text_no_al = re.sub(r"\b(al|altura)\b", "", text, flags=re.I).strip()
    num_match = re.search(r"\d{1,5}", text)
    num = num_match.group(0) if num_match else ""
    street = re.sub(r"\d{1,5}", "", text).strip()

    pats = [f"%{text}%"]
    if text_no_al and text_no_al != text:
        pats.append(f"%{text_no_al}%")
    if street:
        pats.append(f"%{street}%")
    if street and num:
        pats += [f"%{street} {num}%", f"%{street}%{num}%", f"%{num}%{street}%"]
    if num:
        pats.append(f"%{num}%")

    seen, out = set(), []
    for p in pats:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _fetch_candidates_from_table(conn, table: str, patterns: List[str], limit_total: int = 30) -> List[dict]:
    rows: List[dict] = []
    with conn.cursor() as cur:
        for pat in patterns:
            if len(rows) >= limit_total:
                break
            try:
                cur.execute(
                    f"""
                    SELECT id, direccion, zona, tipo_propiedad, ambientes, dormitorios, cochera,
                           precio_venta, precio_alquiler, total_construido, expensas, superficie
                    FROM `{table}`
                    WHERE direccion LIKE %s
                    LIMIT %s
                    """,
                    (pat, max(5, limit_total // 3)),
                )
                rows.extend(cur.fetchall() or [])
            except Exception:
                # por si falta expensas o superficie en la tabla
                try:
                    cur.execute(
                        f"""
                        SELECT id, direccion, zona, tipo_propiedad, ambientes, dormitorios, cochera,
                               precio_venta, precio_alquiler, total_construido, superficie
                        FROM `{table}`
                        WHERE direccion LIKE %s
                        LIMIT %s
                        """,
                        (pat, max(5, limit_total // 3)),
                    )
                    batch = cur.fetchall() or []
                    for r in batch:
                        r.setdefault("expensas", None)
                        r.setdefault("superficie", None)
                    rows.extend(batch)
                except Exception:
                    pass
    return rows


def search_db_by_address(raw_text: str) -> Optional[dict]:
    conn = _safe_connect()
    if not conn:
        return None
    try:
        pats = _build_like_patterns(raw_text)
        cands = _fetch_candidates_from_table(conn, MYSQL_TABLE, pats)
        if not cands and MYSQL_TABLE != "propiedades":
            cands = _fetch_candidates_from_table(conn, "propiedades", pats)
        if not cands:
            return None

        qn = _strip_accents(raw_text)
        best, best_score = None, 0.0
        for r in cands:
            addr = _strip_accents(_s(r.get("direccion")))
            score = SequenceMatcher(None, qn, addr).ratio()
            if score > best_score:
                best, best_score = r, score
        return best if best_score >= 0.55 else None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def search_db_by_zone_token(token: str) -> Optional[dict]:
    token = token.strip()
    if not token:
        return None
    conn = _safe_connect()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    f"""
                    SELECT id, direccion, zona, tipo_propiedad, ambientes, dormitorios, cochera,
                           precio_venta, precio_alquiler, total_construido, expensas, superficie
                    FROM `{MYSQL_TABLE}`
                    WHERE zona LIKE %s
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (f"%{token}%",),
                )
                row = cur.fetchone()
                return row
            except Exception:
                cur.execute(
                    f"""
                    SELECT id, direccion, zona, tipo_propiedad, ambientes, dormitorios, cochera,
                           precio_venta, precio_alquiler, total_construido, superficie
                    FROM `{MYSQL_TABLE}`
                    WHERE zona LIKE %s
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (f"%{token}%",),
                )
                row = cur.fetchone() or None
                if row is not None:
                    row.setdefault("expensas", None)
                    row.setdefault("superficie", None)
                return row
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ====================================================================
# RENDER FICHA
# ====================================================================

def _fmt_expensas_guess(raw) -> str:
    if raw is None:
        return "—"
    s = _s(raw)
    if not s or s.lower() in {"null", "none", "-", "na"}:
        return "—"
    m = re.search(r"(\d+(?:[.,]\d+)?)", s.replace(" ", ""))
    if m:
        token = m.group(1).replace(",", ".")
        try:
            val = float(token)
            n = int(round(val))
            return f"$ {n:,}".replace(",", ".")
        except Exception:
            pass
    return s


def render_property_card_db(row: dict, intent: str) -> str:
    addr = _s(row.get("direccion")) or "Sin dirección"
    zona = _s(row.get("zona")) or "—"
    tprop = _s(row.get("tipo_propiedad")) or "Propiedad"

    def _to_int_safe(v):
        try:
            s = _s(v)
            if s == "":
                return 0
            return int(float(s))
        except Exception:
            return 0

    amb = _to_int_safe(row.get("ambientes"))
    dorm = _to_int_safe(row.get("dormitorios"))
    coch_raw = _s(row.get("cochera")).lower()
    coch_txt = "Sí" if coch_raw in {"1", "si", "sí", "true", "t", "y"} else "No"

    precio_venta = _s(row.get("precio_venta"))
    precio_alquiler = _s(row.get("precio_alquiler"))
    total_construido_raw = row.get("total_construido")
    superficie_raw = row.get("superficie")
    expensas_raw = row.get("expensas")
    expensas_txt = _fmt_expensas_guess(expensas_raw)

    def _is_empty(s: str) -> bool:
        if not s:
            return True
        s2 = s.lower().strip()
        return s2 in {"null", "none", "-", "consultar", "0"}

    if intent == "alquiler":
        operacion = "alquiler"
        valor = precio_alquiler if not _is_empty(precio_alquiler) else "Consultar"
    elif intent == "venta":
        operacion = "venta"
        valor = precio_venta if not _is_empty(precio_venta) else "Consultar"
    else:
        if not _is_empty(precio_alquiler):
            operacion = "alquiler"
            valor = precio_alquiler
        elif not _is_empty(precio_venta):
            operacion = "venta"
            valor = precio_venta
        else:
            operacion = "—"
            valor = "Consultar"

    def _fmt_m2(val) -> str:
        s = _s(val)
        if not s:
            return "—"
        s_clean = s.lower().replace("m2", "").replace("m²", "").strip()
        if s_clean.replace(".", "", 1).isdigit():
            return f"{s_clean} m²"
        return s

    total_construido_txt = _fmt_m2(total_construido_raw)
    superficie_txt = _fmt_m2(superficie_raw)

    ficha = (
        f"🏡 *{tprop}*\n"
        f"{addr} (Zona: {zona})\n\n"
        f"💰 *Operación:* {operacion.capitalize()}\n"
        f"💸 *Valor:* {valor}\n"
        f"🏗 *Total construido:* {total_construido_txt}\n"
        f"📐 *Superficie:* {superficie_txt}\n"
        f"🛏 *Ambientes:* {amb} | Dormitorios: {dorm}\n"
        f"🚗 *Cochera:* {coch_txt}\n"
    )

    if expensas_txt not in {"—", "Consultar"}:
        ficha += f"💬 *Expensas:* {expensas_txt}\n"

    ficha += f"\n🌐 Más info general: {SITE_URL}"
    return ficha


def _infer_intent_from_row(row: dict) -> Optional[str]:
    venta = _s(row.get("precio_venta")).lower()
    alqu = _s(row.get("precio_alquiler")).lower()
    if alqu not in {"", "0", "null", "none", "-"}:
        return "alquiler"
    if venta not in {"", "0", "null", "none", "-"}:
        return "venta"
    return None


# ====================================================================
# DETECCIÓN DE LINKS / DIRECCIONES
# ====================================================================

URL_RX = re.compile(r"(https?://[^\s]+)", re.IGNORECASE)


def _extract_urls(text: str) -> List[str]:
    return URL_RX.findall(text or "") or []


def _slug_to_candidate_text(url: str) -> str:
    try:
        p = urlparse(url)
        slug = (p.path or "").strip("/").replace("-", " ")
        slug = re.sub(r"[_/]+", " ", slug)
        slug = re.sub(r"%[0-9A-Fa-f]{2}", " ", slug)
        slug = re.sub(r"\s+", " ", slug)
        return slug.strip()
    except Exception:
        return ""


def _tokens_from_text(t: str) -> List[str]:
    t = _strip_accents(t)
    parts = re.split(r"[^\wáéíóúñü]+", t)
    return [w for w in parts if len(w) >= 4]


def _has_addr_number_strict(t: str) -> bool:
    return bool(re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ\.]{3,}\s+\d{1,6}", t or ""))


def _try_property_from_link_or_slug(text: str) -> Optional[dict]:
    urls = _extract_urls(text)
    if not urls:
        return None
    for u in urls:
        cand = _slug_to_candidate_text(u)
        if cand:
            row = search_db_by_address(cand)
            if row:
                return row
            for tk in _tokens_from_text(cand):
                row2 = search_db_by_zone_token(tk)
                if row2:
                    return row2
    return None


# ====================================================================
# IA: PARSEO DE JSON Y CEREBRO PRINCIPAL
# ====================================================================

def _parse_llama_json(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    # primer intento: todo como JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # buscar primer bloque {...}
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # fallback: usar raw como reply
    return {"reply": raw.strip()}


def _detect_property_context(user_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
    row = _try_property_from_link_or_slug(user_text)
    if not row and _has_addr_number_strict(user_text):
        row = search_db_by_address(user_text)

    card = ""
    if row:
        intent = state.get("operation") or _infer_intent_from_row(row) or "alquiler"
        card = render_property_card_db(row, intent)
        state["last_property_row"] = row
        state["last_property_card"] = card
    else:
        # reutilizar última ficha como contexto si existe
        card = state.get("last_property_card", "")

    return {"row": row, "card": card}


def _llm_brain(chat_id: str, user_text: str, prop_card: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cerebro principal: usa Groq para decidir qué contestar,
    y si corresponde, derivar al asesor.
    """
    history: List[Dict[str, str]] = state.setdefault("history", [])

    if not groq_client or not GROQ_API_KEY:
        # Sin IA → fallback simple
        logger.warning("GROQ no configurado, usando menú básico.")
        if prop_card:
            reply = prop_card + "\n\nSi te interesa esta propiedad, contame un poco de tu situación y disponibilidad."
        else:
            reply = _say_menu()
        return {
            "reply": reply,
            "operation": "",
            "vendor_push": False,
            "vendor_message": "",
            "closing_text": "",
        }

    system_msg = (
    "Sos un asistente inmobiliario argentino para *Inmobiliaria Finca del Este* en WhatsApp.\n"
    "Hablás en español rioplatense, cálido pero profesional.\n\n"
    "Tenés tres operaciones principales:\n"
    "- alquiler\n"
    "- venta\n"
    "- tasacion\n\n"
    "Objetivos:\n"
    "1) Conversar libremente con la persona, entender qué necesita.\n"
    "2) Hacer las preguntas mínimas necesarias (tipo de propiedad, zona/barrio, presupuesto, "
    "   fecha estimada, ambientes, dormitorios, etc.).\n"
    "3) Cuando haya una intención real y datos suficientes, marcá vendor_push=true.\n"
    "4) Si recibís una FICHA_PROPIEDAD_ACTUAL, podés usarla como referencia, pero NO inventes datos.\n\n"
    "Respondé SIEMPRE SOLO en JSON válido:\n\n"
    "{\n"
    '  "reply": "texto para el cliente",\n'
    '  "operation": "alquiler" | "venta" | "tasacion" | "",\n'
    '  "vendor_push": true | false,\n'
    '  "vendor_message": "resumen para el asesor si vendor_push es true",\n'
    '  "closing_text": "texto de cierre opcional"\n'
    "}\n\n"
    "Reglas adicionales:\n"
    "- En 'reply', usá tono cálido y argentino, siempre claro.\n"
    "- No menciones que sos IA.\n"
    "- Si faltan datos, vendor_push debe ser false.\n"
    "- En vendor_message, sé claro: operación, zona, presupuesto, ficha si corresponde.\n"
)


    messages = [{"role": "system", "content": system_msg}]

    if prop_card:
        messages.append(
            {
                "role": "system",
                "content": "FICHA_PROPIEDAD_ACTUAL (NO mostrar literal, sólo usar como contexto):\n"
                           + prop_card,
            }
        )

    # historial corto
    for h in history[-8:]:
        messages.append(h)

    messages.append({"role": "user", "content": user_text})

    try:
        logger.info(f"LLAMA - llamando modelo: {LLAMA_MODEL}")
        completion = groq_client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.4,
        )
        content = completion.choices[0].message.content.strip()
        logger.info(f"LLAMA - raw content: {content}")
        data = _parse_llama_json(content)
    except Exception as e:
        logger.error(f"Error llamando a Groq: {e}")
        return {
            "reply": _say_menu(),
            "operation": "",
            "vendor_push": False,
            "vendor_message": "",
            "closing_text": "",
        }

    if not data.get("reply"):
        data["reply"] = _say_menu()

    op = (data.get("operation") or "").lower().strip()
    if op in {"alquiler", "venta", "tasacion"}:
        state["operation"] = op

    # actualizar historial (sólo el texto para el cliente)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": data["reply"]})
    state["history"] = history[-20:]

    return data


# ====================================================================
# LÓGICA PRINCIPAL
# ====================================================================

def _process_qualify(body: QualifyIn) -> QualifyOut:
    chat_id = body.chatId
    text = (body.message or "").strip()

    print("=== PROCESS QUALIFY ===")
    print("chatId:", chat_id)
    print("message:", text)
    print("isFromMe:", body.isFromMe)

    if body.isFromMe or not text:
        return QualifyOut(reply_text="", vendor_push=False, vendor_message="", closing_text="")

    state = STATE.setdefault(chat_id, {})

    if _strip_accents(text) in {"reset", "reiniciar", "restart"}:
        STATE[chat_id] = {}
        state = STATE[chat_id]
        reply = "Perfecto, empezamos de cero. Contame qué estás necesitando (alquiler, venta o tasación)."
        return QualifyOut(reply_text=reply, vendor_push=False, vendor_message="", closing_text="")

    prop_ctx = _detect_property_context(text, state)
    prop_row = prop_ctx.get("row")
    prop_card = prop_ctx.get("card") or ""

    brain = _llm_brain(chat_id, text, prop_card, state)

    reply_text = brain.get("reply", "") or ""
    vendor_push = bool(brain.get("vendor_push"))
    vendor_message = brain.get("vendor_message") or ""
    closing_text = brain.get("closing_text") or ""

    # Si vamos a derivar y tenemos una propiedad detectada, agregamos ficha al resumen
    if vendor_push and prop_row:
        op = state.get("operation") or _infer_intent_from_row(prop_row) or "alquiler"
        ficha = render_property_card_db(prop_row, op)
        extra = f"\n\n--- Ficha detectada desde la conversación ---\n{ficha}"
        vendor_message = (vendor_message + extra).strip()

    return QualifyOut(
        reply_text=reply_text,
        vendor_push=vendor_push,
        vendor_message=vendor_message,
        closing_text=closing_text,
    )


# ====================================================================
# ENVÍO A WHATSAPP (GREEN API)
# ====================================================================

async def send_whatsapp_message(chat_id: str, text: str):
    if not text or not chat_id:
        return
    if not (GREEN_INSTANCE_ID and GREEN_API_TOKEN):
        logger.error("Green API no configurado, no se puede enviar mensaje.")
        return

    url = f"{GREEN_API_URL}/waInstance{GREEN_INSTANCE_ID}/sendMessage/{GREEN_API_TOKEN}"
    payload = {"chatId": chat_id, "message": text}

    print("### INTENTO DE ENVÍO A WHATSAPP ###")
    print("chat_id:", chat_id)
    print("mensaje:", text)
    print("URL:", url)
    print("Payload:", payload)

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
            print("RESPUESTA GREEN STATUS:", resp.status_code)
            print("RESPUESTA GREEN BODY:", resp.text)
    except Exception as e:
        logger.error(f"Error enviando mensaje a WhatsApp: {e}")


# ====================================================================
# ENDPOINTS
# ====================================================================

@app.post("/qualify", response_model=QualifyOut)
async def qualify_endpoint(body: QualifyIn) -> QualifyOut:
    out = _process_qualify(body)
    return out


@app.post("/webhook")
async def green_webhook(payload: dict):
    """
    Webhook directo desde Green API.
    Configurá en Green:
        incomingWebhook = https://TU-APP.railway.app/webhook
    """
    print("=== RAW WEBHOOK PAYLOAD ===")
    print(payload)

    if payload.get("typeWebhook") != "incomingMessageReceived":
        return {"status": "ignored"}

    sender = payload.get("senderData") or {}
    msg_data = payload.get("messageData") or {}

    chat_id = sender.get("chatId") or sender.get("sender")
    sender_name = (
        sender.get("senderName")
        or sender.get("chatName")
        or sender.get("senderContactName")
        or ""
    )

    text = ""
    if msg_data.get("typeMessage") == "textMessage":
        text = (msg_data.get("textMessageData") or {}).get("textMessage", "") or ""
    else:
        return {"status": "no_text"}

    if not chat_id or not text.strip():
        return {"status": "no_chat_or_text"}

    body = QualifyIn(chatId=chat_id, message=text, isFromMe=False, senderName=sender_name)
    out = _process_qualify(body)

    # respuesta al cliente
    if out.reply_text:
        await send_whatsapp_message(chat_id, out.reply_text)

    # derivación al asesor
    if out.vendor_push and out.vendor_message and VENDOR_CHAT_ID:
        await send_whatsapp_message(VENDOR_CHAT_ID, out.vendor_message)

    return {"status": "ok"}
