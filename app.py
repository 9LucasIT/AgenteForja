print("####################################")
print("### EJECUTANDO ESTE APP.PY NUEVO ###")
print("####################################")


import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq


# ==================== CONFIG BÁSICA ====================

SITE_URL = os.getenv("SITE_URL", "https://www.fincasdeleste.com.uy/")

# MySQL (igual que antes)
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

# Chat del asesor (puede ser número o grupo)
VENDOR_CHAT_ID = os.getenv("VENDOR_CHAT_ID", "").strip()  # ej: "5493412654593@c.us"

# IA - Groq / LLaMA-3
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama-3.1-8b-instant")
groq_client: Optional[Groq] = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

print("#########################")
print("GROQ_API_KEY RAW VALUE:", repr(GROQ_API_KEY))
print("GROQ_API_KEY LENGTH:", len(GROQ_API_KEY) if GROQ_API_KEY else 0)
print("CLIENT CREATED:", groq_client is not None)
print("#########################")


# Estado en memoria
STATE: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="WhatsApp Inmo Agent (Forja/Fincas, sin n8n)", version="2025-12-10")


# ==================== MODELOS I/O ====================

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


# ==================== HELPERS DE TEXTO ====================

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
        "Gracias por contactarte con nosotros. ¿En qué te puedo ayudar hoy?\n\n"
        "1️⃣ *Alquileres*\n"
        "2️⃣ *Ventas*\n"
        "3️⃣ *Tasaciones*\n\n"
        "📝 Podés escribir el *número* o el *nombre* de la opción.\n"
        "🔄 Si querés empezar de nuevo, escribí *\"reset\"*."
    )


def _ask_zone_or_address() -> str:
    return "¿Tenés dirección o link exacto de la propiedad, o estás averiguando por una zona/barrio?"


def _ask_disponibilidad() -> str:
    return "¡Perfecto! 🕓 Antes de que te contacte nuestro asesor, ¿podrías contarme tu *disponibilidad horaria*?"


def _ask_qualify_prompt(intent: str) -> str:
    if intent == "alquiler":
        return "Para avanzar con *alquiler*, ¿contás con *ingresos demostrables* que tripliquen el valor del alquiler?"
    return _ask_disponibilidad()


def _ask_income_question() -> str:
    return (
        "Para avanzar con *alquiler*, ¿contás con *ingresos demostrables* que tripliquen el valor del alquiler?\n\n"
        "✅ *También es válido* si un familiar puede compartir su *recibo de sueldo* para acompañarte en la operación."
    )


def _ask_guarantee_question() -> str:
    return (
        "🤝 ¿Qué *tipo de garantía* tenés?\n"
        "1️⃣ Garantía de propietario de CABA\n"
        "2️⃣ Seguro de caución FINAER\n"
        "3️⃣ Ninguna de las anteriores\n\n"
        "✍️ *Escribí el número* de la opción que prefieras."
    )


def _farewell() -> str:
    return "Perfecto, quedo atento a tus consultas. ¡Gracias por escribir! 😊"


# ==================== DB (igual esquema anterior) ====================

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
    except Exception:
        return None


def _build_like_patterns(raw: str) -> List[str]:
    text = raw.strip()
    text_no_al = re.sub(r"\b(al|altura)\b", "", text, flags=re.I).strip()
    num = (re.search(r"\d{1,5}", text) or re.match("", "")).group(0) if re.search(r"\d{1,5}", text) else ""
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
                           precio_venta, precio_alquiler, total_construido, superficie, expensas
                    FROM `{table}`
                    WHERE direccion LIKE %s
                    LIMIT %s
                    """,
                    (pat, max(5, limit_total // 3)),
                )
                rows.extend(cur.fetchall() or [])
                continue
            except Exception:
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
        if not cands and MYSQL_TABLE != "propiedad":
            cands = _fetch_candidates_from_table(conn, "propiedad", pats)
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
                           precio_venta, precio_alquiler, total_construido, superficie, expensas
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


# ==================== RENDER FICHA ====================

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

    ficha += f"\n🌐 Más info: {SITE_URL}"

    if intent == "alquiler":
        ficha += "\n\n📝 *Importante:* Se realizan contratos a 24 meses con ajuste cada 3 meses por IPC."

    return ficha


# ==================== LINKS / INTENTOS ====================

URL_RX = re.compile(r"(https?://[^\s]+)", re.IGNORECASE)
STOPWORDS = {"en", "de", "del", "la", "el", "y", "a", "con", "por", "para", "un", "una", "los", "las", "—", "–"}


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


def _infer_intent_from_row(row: dict) -> Optional[str]:
    venta = _s(row.get("precio_venta")).lower()
    alqu = _s(row.get("precio_alquiler")).lower()
    if alqu not in {"", "0", "null", "none", "-"}:
        return "alquiler"
    if venta not in {"", "0", "null", "none", "-"}:
        return "venta"
    return None


def _tokens_from_text(t: str) -> List[str]:
    t = _strip_accents(t)
    parts = re.split(r"[^\wáéíóúñü]+", t)
    return [w for w in parts if len(w) >= 4 and w not in STOPWORDS]


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


def _is_yes(t: str) -> bool:
    t = _strip_accents(t)
    return t in {
        "si",
        "sí",
        "ok",
        "dale",
        "claro",
        "perfecto",
        "de una",
        "si, claro",
        "listo",
        "afirmativo",
        "si quiero",
        "si, quiero",
    }


def _is_no(t: str) -> bool:
    t = _strip_accents(t)
    return t in {"no", "nop", "no gracias", "nah", "negativo", "no quiero", "no, gracias"}


def _wants_reset(t: str) -> bool:
    t = _strip_accents(t)
    return t in {"reset", "reiniciar", "restart"}


def _is_rental_intent(t: str) -> bool:
    t = _strip_accents(t)
    keys = [
        "alquiler",
        "alquilo",
        "alquilar",
        "quiero alquilar",
        "busco alquiler",
        "estoy buscando alquiler",
        "rentar",
        "arrendar",
    ]
    return any(k in t for k in keys) or t.strip() in {"1", "1-", "1 -", "alquileres"}


def _is_sale_intent(t: str) -> bool:
    t = _strip_accents(t)
    keys = ["venta", "vender", "comprar", "compro", "quiero comprar", "ventas"]
    return any(k in t for k in keys) or t.strip() in {"2", "2-", "2 -", "ventas"}


def _is_valuation_intent(t: str) -> bool:
    t = _strip_accents(t)
    keys = ["tasacion", "tasación", "tasar", "tasaciones"]
    return any(k in t for k in keys) or t.strip() in {"3", "3-", "3 -"}


def _is_zone_search(t: str) -> bool:
    nt = _strip_accents(t)
    patterns = [
        r"\bno tengo (la )?direccion\b",
        r"\bno tengo link\b",
        r"\bsolo (zona|barrio)\b",
        r"\bestoy averiguando\b",
        r"\bbusco.*(zona|barrio)\b",
    ]
    return any(re.search(p, nt) for p in patterns)


def _num_from_text(t: str) -> Optional[int]:
    m = re.search(r"\b(\d{1,5})\b", t or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _money_from_text(t: str) -> Optional[int]:
    if not t:
        return None
    m = re.search(r"\b(\d{1,3}(?:[.,]?\d{3})+|\d+)\b", t.replace(" ", ""))
    if not m:
        return None
    raw = m.group(1).replace(".", "").replace(",", "")
    try:
        return int(raw)
    except Exception:
        return None


def _has_addr_number_strict(t: str) -> bool:
    return bool(re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ\.]{3,}\s+\d{1,6}", t or ""))


def _reset(chat_id: str):
    STATE[chat_id] = {"stage": "menu"}


def _ensure_session(chat_id: str):
    if chat_id not in STATE:
        _reset(chat_id)


# ==================== MOTOR IA: REESCRIBIR RESPUESTA ====================

def _rewrite_with_llama(chat_id: str, user_text: str, base_reply: str) -> str:
    """
    Usa LLaMA-3 (Groq) para generar una respuesta conversacional y humana,
    usando el flujo (_process_qualify) solo como guía.
    Mantiene datos numéricos, direcciones y links, pero puede reescribir
    completamente el texto para que suene natural.
    """
    state = STATE.setdefault(chat_id, {})
    history: List[Dict[str, str]] = state.setdefault("history", [])

    # Si no hay IA o no hay respuesta base, salimos rápido
    if not base_reply or groq_client is None or not GROQ_API_KEY:
        if user_text:
            history.append({"role": "user", "content": user_text})
        if base_reply:
            history.append({"role": "assistant", "content": base_reply})
        state["history"] = history[-20:]
        return base_reply

    stage = state.get("stage", "menu")
    intent = state.get("intent") or "no definido"

    # Links detectados en el mensaje del usuario o en la respuesta base
    urls = _extract_urls((user_text or "") + " " + (base_reply or ""))
    has_link = bool(urls)

    # Resumen cortito del estado para la IA
    resumen_estado = [
        f"Etapa actual del flujo interno: {stage}.",
        f"Intención principal: {intent}.",
    ]
    if state.get("tas_op"):
        resumen_estado.append(f"Operación de tasación: {state.get('tas_op')}.")
    if state.get("tas_m2"):
        resumen_estado.append(f"Metros cuadrados mencionados: {state.get('tas_m2')}.")
    if state.get("tas_dir"):
        resumen_estado.append(f"Dirección cargada para tasación: {state.get('tas_dir')}.")

    if has_link:
        resumen_estado.append(f"El usuario envió estos links: {', '.join(urls)}.")

    estado_texto = "\n".join(resumen_estado)

    system_msg = (
        "Sos un asistente virtual inmobiliario argentino, profesional pero cercano.\n"
        "Usás un flujo interno (etapas de menú, tasación, alquiler, venta, derivación) "
        "solo como guía, pero VOS redactás la respuesta final de forma natural.\n\n"
        "Instrucciones IMPORTANTES:\n"
        "- Usá el *mensaje base* y el *estado del flujo* como guía de qué temas cubrir, "
        "pero podés reescribir el texto por completo para que suene conversacional.\n"
        "- Mantené SIEMPRE correctos los datos numéricos, montos de dinero, metros cuadrados, "
        "direcciones, nombres de zonas y URLs. No inventes propiedades, precios ni condiciones.\n"
        "- Podés agregar pequeñas frases de empatía o una pregunta de seguimiento si tiene sentido.\n"
        "- Respondé SIEMPRE en español rioplatense, tono profesional pero humano.\n"
        "- Si el usuario envía un link (aunque sea de Argenprop, Zonaprop u otra página externa), "
        "no digas que lo analizaste internamente; en cambio, reconocé que es un link de una propiedad "
        "y preguntale qué quiere que le aclares o si busca algo similar.\n"
        "- No alargues innecesariamente la respuesta; apuntá a 2–5 líneas salvo que el contexto necesite más."
    )

    # Construimos los mensajes para la IA
    messages = [{"role": "system", "content": system_msg}]

    # Un poco de contexto de conversación previa (recortado)
    for h in history[-6:]:
        messages.append(h)

    # Mensaje del usuario actual
    if user_text:
        messages.append({"role": "user", "content": user_text})

    # Contexto estructurado + mensaje base como guía
    contexto = (
        "Contexto del flujo interno y mensaje base para que lo tengas como referencia:\n\n"
        f"{estado_texto}\n\n"
        "Mensaje base generado por la lógica del flujo (no lo repitas tal cual, usalo solo como guía):\n"
        f"{base_reply}"
    )

    messages.append({"role": "assistant", "content": contexto})

    try:
        completion = groq_client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=messages,
            max_tokens=400,
            temperature=0.7,
        )
        final_reply = completion.choices[0].message.content.strip()
    except Exception as e:
        print("DEBUG - ERROR GROQ:", e)
        final_reply = base_reply

    # Guardamos historia con la respuesta ya reescrita
    if user_text:
        history.append({"role": "user", "content": user_text})
    if final_reply:
        history.append({"role": "assistant", "content": final_reply})

    state["history"] = history[-20:]
    return final_reply



# ==================== MOTOR ORIGINAL DE CALIFICACIÓN ====================

def _process_qualify(body: QualifyIn) -> QualifyOut:
    chat_id = body.chatId
    text = (body.message or "").strip()

    _ensure_session(chat_id)
    s = STATE[chat_id]

    if _wants_reset(text):
        _reset(chat_id)
        return QualifyOut(reply_text=_say_menu())

    stage = s.get("stage", "menu")

    # --- MENU ---
    if stage == "menu":
        if not text:
            return QualifyOut(reply_text=_say_menu())

        user_op = "alquiler" if _is_rental_intent(text) else "venta" if _is_sale_intent(text) else None

        row_link = _try_property_from_link_or_slug(text)
        if row_link:
            prop_op = _infer_intent_from_row(row_link) or (user_op or "venta")
            s["prop_row"] = row_link
            s["intent"] = user_op or prop_op
            brief = render_property_card_db(row_link, intent=s["intent"])
            s["prop_brief"] = brief
            s["stage"] = "show_property_asked_qualify"
            s["last_prompt"] = "qual_disp_alq" if s["intent"] == "alquiler" else "qual_disp_venta"
            return QualifyOut(
                reply_text=brief
                + "\n\n"
                + (_ask_disponibilidad() if s["intent"] == "alquiler" else _ask_qualify_prompt("venta"))
            )

        if user_op or _is_valuation_intent(text):
            s["intent"] = user_op or "tasacion"
            if s["intent"] == "tasacion":
                s["stage"] = "tas_op"
                s["tas_op"] = None
                s["tas_prop"] = None
                s["tas_m2"] = None
                s["tas_dir"] = None
                s["tas_exp"] = None
                s["tas_feat"] = None
                s["tas_disp"] = None
                return QualifyOut(
                    reply_text="¡Genial! Para la *tasación*, decime el *tipo de operación*: ¿venta o alquiler?"
                )
            s["stage"] = "ask_zone_or_address"
            return QualifyOut(reply_text=_ask_zone_or_address())

        return QualifyOut(reply_text=_say_menu())

    # --- TASACIÓN ---
    if stage == "tas_op":
        t = _strip_accents(text)
        if "venta" in t:
            s["tas_op"] = "venta"
        elif "alquiler" in t or "renta" in t or "alquilar" in t:
            s["tas_op"] = "alquiler"
        else:
            return QualifyOut(reply_text="¿Me confirmás el *tipo de operación*? (venta o alquiler)")
        s["stage"] = "tas_prop"
        return QualifyOut(
            reply_text="Perfecto. ¿Cuál es el *tipo de propiedad*? (ej.: departamento, casa, local, oficina)"
        )

    if stage == "tas_prop":
        s["tas_prop"] = text.strip() or "no informado"
        s["stage"] = "tas_m2"
        return QualifyOut(reply_text="Gracias. ¿Cuántos *metros cuadrados* aproximados tiene la propiedad?")

    if stage == "tas_m2":
        n = _num_from_text(text)
        if n is None:
            return QualifyOut(reply_text="¿Me pasás un *número* aproximado de metros cuadrados? (ej.: 65)")
        s["tas_m2"] = n
        s["stage"] = "tas_dir"
        return QualifyOut(
            reply_text="Anotado. ¿Cuál es la *dirección exacta* del inmueble? (calle y número; si podés, piso/depto)"
        )

    if stage == "tas_dir":
        if not _has_addr_number_strict(text):
            return QualifyOut(
                reply_text="¿Podés pasarme *calle y número*? Si tenés piso/depto, mejor."
            )
        s["tas_dir"] = text.strip()
        s["stage"] = "tas_exp"
        return QualifyOut(
            reply_text="¿La propiedad tiene *expensas*? Si tiene, ¿de cuánto es el *costo mensual* (ARS)? Si no, decime *no tiene*."
        )

    if stage == "tas_exp":
        t = _strip_accents(text)
        if any(x in t for x in ("no tiene", "sin expensas", "no")):
            s["tas_exp"] = "no tiene"
        else:
            val = _money_from_text(text)
            s["tas_exp"] = f"${val:,}".replace(",", ".") if val else (text.strip() or "no informado")
        s["stage"] = "tas_feat"
        return QualifyOut(
            reply_text="¿Dispone *balcón, patio, amenities o estudio de factibilidad*? Podés responder con una lista o “no”."
        )

    if stage == "tas_feat":
        t = _strip_accents(text)
        feats = []
        if "balcon" in t or "balcón" in text.lower():
            feats.append("balcón")
        if "patio" in t:
            feats.append("patio")
        if "amenities" in t:
            feats.append("amenities")
        if "estudio" in t or "factibilidad" in t:
            feats.append("estudio factibilidad")
        if t in {"no", "ninguno", "ninguna", "ningunos"}:
            feats = []
        s["tas_feat"] = ", ".join(feats) if feats else "no"
        s["stage"] = "tas_disp"
        return QualifyOut(
            reply_text="¡Último dato! ¿Cuál es tu *disponibilidad horaria* aproximada para que te contacte un asesor?"
        )

    if stage == "tas_disp":
        s["tas_disp"] = text.strip() or "no informado"
        s["stage"] = "done"
        resumen = (
            "Tasación solicitada ✅\n"
            f"• Operación: {s.get('tas_op','N/D')}\n"
            f"• Propiedad: {s.get('tas_prop','N/D')}\n"
            f"• Metros²: {s.get('tas_m2','N/D')}\n"
            f"• Dirección: {s.get('tas_dir','N/D')}\n"
            f"• Expensas: {s.get('tas_exp','N/D')}\n"
            f"• Extras: {s.get('tas_feat','N/D')}\n"
            f"• Disponibilidad: {s.get('tas_disp','N/D')}\n"
            f"• Chat: {chat_id}"
        )
        cierre = (
            "Perfecto, con todos estos datos ya contamos con lo suficiente para derivarte con un asesor. "
            "¡Muchísimas gracias por tu tiempo!"
        )
        return QualifyOut(
            reply_text=cierre,
            vendor_push=True,
            vendor_message=resumen,
            closing_text="",
        )

    # --- BÚSQUEDA DIRECCIÓN / ZONA ---
    if stage == "ask_zone_or_address":
        row_link = _try_property_from_link_or_slug(text)
        if row_link:
            intent_infer = _infer_intent_from_row(row_link) or s.get("intent") or "venta"
            s["prop_row"] = row_link
            s["intent"] = s.get("intent") or intent_infer
            brief = render_property_card_db(row_link, intent=s["intent"])
            s["prop_brief"] = brief
            s["stage"] = "show_property_asked_qualify"
            s["last_prompt"] = "qual_disp_alq" if s["intent"] == "alquiler" else "qual_disp_venta"
            return QualifyOut(
                reply_text=brief
                + "\n\n"
                + (_ask_disponibilidad() if s["intent"] == "alquiler" else _ask_qualify_prompt("venta"))
            )

        if _is_zone_search(text):
            s["stage"] = "done"
            msg = (
                "Perfecto. Te dejo el link donde están todas nuestras propiedades para que puedas ver si alguna te interesa:\n"
                f"{SITE_URL}\n\n"
                "Cualquier consulta puntual de una ficha me escribís por acá."
            )
            return QualifyOut(reply_text=msg, closing_text=_farewell())

        intent = s.get("intent", "alquiler")
        row = search_db_by_address(text)

        if row:
            intent_infer = _infer_intent_from_row(row) or intent
            brief = render_property_card_db(row, intent=intent_infer)
            s["prop_row"] = row
            s["prop_brief"] = brief
            s["intent"] = intent_infer
            s["stage"] = "show_property_asked_qualify"
            s["last_prompt"] = "qual_disp_alq" if s["intent"] == "alquiler" else "qual_disp_venta"
            return QualifyOut(
                reply_text=brief
                + "\n\n"
                + (_ask_disponibilidad() if s["intent"] == "alquiler" else _ask_qualify_prompt("venta"))
            )

        return QualifyOut(
            reply_text=(
                "No pude identificar la ficha a partir del texto. "
                "¿Podés confirmarme la *dirección exacta* tal como figura en la publicación?"
            )
        )
        

    # --- MOSTRAR PROPIEDAD Y CALIFICAR ---
    if stage == "show_property_asked_qualify":
        intent = s.get("intent", "alquiler")

        if intent == "alquiler":
            if s.get("last_prompt") != "qual_disp_alq":
                s["last_prompt"] = "qual_disp_alq"
                return QualifyOut(reply_text=_ask_disponibilidad())
            else:
                s["disp_alquiler"] = text.strip() or "no informado"
                s["stage"] = "ask_handover"
                s.pop("last_prompt", None)
                return QualifyOut(
                    reply_text=(
                        "Perfecto 😊 ¿Querés que te contacte un asesor humano por este WhatsApp para avanzar? "
                        "Respondé *sí* o *no*."
                    )
                )

        if intent == "venta":
            if s.get("last_prompt") != "qual_disp_venta":
                s["last_prompt"] = "qual_disp_venta"
                return QualifyOut(reply_text=_ask_disponibilidad())
            else:
                s["disp_venta"] = text.strip() or "no informado"
                s["stage"] = "ask_handover"
                s.pop("last_prompt", None)
                return QualifyOut(
                    reply_text=(
                        "Perfecto 😊 ¿Querés que te contacte un asesor humano por este WhatsApp para avanzar? "
                        "Respondé *sí* o *no*."
                    )
                )

    # --- PREGUNTAR DERIVACIÓN ---
    if stage == "ask_handover":
        s.pop("last_prompt", None)

        if _is_yes(text):
            s["stage"] = "done"
            disp = ""
            if s.get("disp_alquiler"):
                disp = f"Disponibilidad: {s['disp_alquiler']}\n"
            elif s.get("disp_venta"):
                disp = f"Disponibilidad: {s['disp_venta']}\n"

            op_line = f"Operación seleccionada: {s['intent'].capitalize()}\n" if s.get("intent") else ""
            vendor_msg = (
                "Lead calificado desde WhatsApp.\n"
                f"Chat: {chat_id}\n"
                f"{op_line}"
                f"{disp}"
                f"{s.get('prop_brief','')}\n"
            )

            return QualifyOut(
                reply_text=(
                    "Perfecto, te derivo con un asesor humano que te va a contactar por acá en breve. "
                    "¡Gracias por escribir!"
                ),
                vendor_push=True,
                vendor_message=vendor_msg,
                closing_text=_farewell(),
            )

        if _is_no(text):
            s["stage"] = "done"
            return QualifyOut(
                reply_text=(
                    "¡Gracias por tu consulta! Quedamos a disposición por cualquier otra duda.\n"
                    "Cuando quieras retomar, escribí *reset* y arrancamos desde cero."
                ),
                closing_text=_farewell(),
            )

        return QualifyOut(
            reply_text="¿Querés que te contacte un asesor humano por este WhatsApp para avanzar? Respondé *sí* o *no*."
        )

    # Fallback: volvemos al menú
    _reset(chat_id)
    return QualifyOut(reply_text=_say_menu())


# ==================== ENVÍO A WHATSAPP (GREEN API) ====================

async def send_whatsapp_message(chat_id: str, text: str):
    if not text or not chat_id:
        return
    if not (GREEN_INSTANCE_ID and GREEN_API_TOKEN):
        # sin credenciales no hacemos nada
        return

    url = f"{GREEN_API_URL}/waInstance{GREEN_INSTANCE_ID}/sendMessage/{GREEN_API_TOKEN}"
    payload = {"chatId": chat_id, "message": text}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(url, json=payload)
    except Exception:
        # en producción podrías loguear el error
        pass


# ==================== ENDPOINT /qualify (para pruebas) ====================

@app.post("/qualify", response_model=QualifyOut)
async def qualify_endpoint(body: QualifyIn) -> QualifyOut:
    # ignore mensajes que sean del propio bot (por si alguien llama esto manualmente)
    if body.isFromMe:
        return QualifyOut(reply_text="", vendor_push=False, vendor_message="", closing_text="")

    out = _process_qualify(body)
    out.reply_text = _rewrite_with_llama(body.chatId, body.message or "", out.reply_text)
    return out


# ==================== ENDPOINT WEBHOOK DIRECTO DE GREEN ====================

@app.post("/webhook")
async def green_webhook(payload: dict):
    """
    Endpoint para recibir webhooks DIRECTO desde Green API.
    """

    print("=== RAW WEBHOOK PAYLOAD ===")
    print(payload)

    type_webhook = payload.get("typeWebhook")

    # Solo procesamos mensajes entrantes
    if type_webhook != "incomingMessageReceived":
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

    # ===========================
    # EXTRAER TEXTO DE CUALQUIER TIPO
    # ===========================
    text = ""

    # Tipo estándar
    if msg_data.get("textMessageData"):
        text = msg_data["textMessageData"].get("textMessage", "") or ""

    # Tipo extendido (Green API lo usa mucho)
    elif msg_data.get("extendedTextMessageData"):
        text = msg_data["extendedTextMessageData"].get("textMessage", "") or ""

    # Tipos nuevos (Green API los usa en grupos)
    elif msg_data.get("chatMessage"):
        text = msg_data["chatMessage"].get("body", "") or ""

    # Si aún no hay texto → ignoramos
    if not text or not text.strip():
        return {"status": "no_text"}

    # ===========================
    # PROCESAR MENSAJE NORMAL
    # ===========================

    body = QualifyIn(
        chatId=chat_id,
        message=text,
        isFromMe=False,
        senderName=sender_name
    )

    out = _process_qualify(body)

    # IA – reescritura humana
    out.reply_text = _rewrite_with_llama(chat_id, text, out.reply_text)

    # Responder al usuario
    if out.reply_text:
        await send_whatsapp_message(chat_id, out.reply_text)

    # Derivación al asesor
    if out.vendor_push and out.vendor_message and VENDOR_CHAT_ID:
        await send_whatsapp_message(VENDOR_CHAT_ID, out.vendor_message)

    return {"status": "ok"}

