# app.py
import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

SITE_URL = os.getenv("SITE_URL", "https://www.veglienzone.com.ar/").strip()

DATABASE_URL = os.getenv("DATABASE_URL", "") or os.getenv("MYSQL_URL", "")
MYSQL_HOST = os.getenv("MYSQLHOST") or os.getenv("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQLPORT") or os.getenv("MYSQL_PORT") or "3306")
MYSQL_USER = os.getenv("MYSQLUSER") or os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQLPASSWORD") or os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQLDATABASE") or os.getenv("MYSQL_DATABASE")
MYSQL_TABLE = os.getenv("MYSQL_TABLE", "propiedades")

STATE: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="FastAPI WhatsApp Agent (DB)", version="2025-11-03")

# =============== IO ===============
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


# =============== Texto helpers ===============
def _strip_accents(s: str) -> str:
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _say_menu() -> str:
    return (
        "Gracias por contactarte con el área comercial de Veglienzone Gestión Inmobiliaria. "
        "¿Cómo podemos ayudarte hoy?\n"
        "1- Alquileres\n2- Ventas\n3- Tasaciones\n\n"
        "Nota: si en cualquier momento escribís *reset*, la conversación se reinicia desde cero."
    )


def _ask_zone_or_address() -> str:
    return "¿Tenés dirección o link exacto de la propiedad, o estás averiguando por una zona/barrio?"


def _ask_disponibilidad() -> str:
    return "¡Perfecto! 🕓 Antes de que te contacte nuestro asesor, ¿podrías contarme tu *disponibilidad horaria*?"


def _ask_qualify_prompt(intent: str) -> str:
    # Para alquiler, mantenemos la primera pregunta de ingresos (excluyente).
    if intent == "alquiler":
        return "Para avanzar con *alquiler*, ¿contás con *ingresos demostrables* que tripliquen el valor del alquiler?"
    # Para venta, reemplazamos por disponibilidad (cualquier respuesta válida).
    return _ask_disponibilidad()


def _ask_income_question() -> str:
    return "Para avanzar con *alquiler*, ¿contás con *ingresos demostrables* que tripliquen el valor del alquiler?"


def _ask_guarantee_question() -> str:
    return (
        "¿Qué *tipo de garantía* tenés?\n"
        "1) Garantía de propietario de CABA\n"
        "2) Seguro de caución FINAER\n"
        "3) Ninguna de las anteriores"
    )


def _farewell() -> str:
    return "Perfecto, quedo atento a tus consultas. ¡Gracias por escribir! 😊"


# =============== DB ===============
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
            out.append(p); seen.add(p)
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
                           precio_venta, precio_alquiler, total_construido
                    FROM `{table}`
                    WHERE direccion LIKE %s
                    LIMIT %s
                    """,
                    (pat, max(5, limit_total // 3)),
                )
                rows.extend(cur.fetchall() or [])
            except Exception:
                return rows
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
            addr = _strip_accents(r.get("direccion") or "")
            score = SequenceMatcher(None, qn, addr).ratio()
            if score > best_score:
                best, best_score = r, score
        return best if best_score >= 0.55 else None
    finally:
        try:
            conn.close()
        except Exception:
            pass

# === búsqueda por zona (para fallback de enlaces) ===
def search_db_by_zone_token(token: str) -> Optional[dict]:
    token = token.strip()
    if not token:
        return None
    conn = _safe_connect()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, direccion, zona, tipo_propiedad, ambientes, dormitorios, cochera,
                       precio_venta, precio_alquiler, total_construido
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
        return None
    finally:
        try: conn.close()
        except Exception: pass


# =============== Render ficha ===============
def _to_int(x, default=0):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def _fmt_money(v) -> str:
    try:
        if v is None:
            return "Consultar"
        s = str(v).strip()
        if s == "" or s == "0" or s.lower() in {"null", "none"}:
            return "Consultar"
        f = float(s)
        if f <= 0:
            return "Consultar"
        return f"USD {int(f):,}".replace(",", ".")
    except Exception:
        return "Consultar"


def _has_price(v) -> bool:
    try:
        if v is None:
            return False
        s = str(v).strip()
        if s == "" or s.lower() in {"null", "none"}:
            return False
        f = float(s)
        return f > 0
    except Exception:
        return False


def render_property_card_db(row: dict, intent: str) -> str:
    # Título y básicos
    addr = (row.get("direccion") or "Sin dirección").strip()
    zona = (row.get("zona") or "—").strip()
    tprop = (row.get("tipo_propiedad") or "Propiedad").strip()

    # Amb / Dorm / Cochera (mantengo tu lógica actual)
    def _to_int_safe(v):
        try:
            if v is None:
                return 0
            s = str(v).strip()
            if s == "":
                return 0
            return int(float(s))
        except Exception:
            return 0

    amb = _to_int_safe(row.get("ambientes"))
    dorm = _to_int_safe(row.get("dormitorios"))
    coch_raw = str(row.get("cochera") or "").strip().lower()
    coch_txt = "Sí" if coch_raw in {"1", "si", "sí", "true", "t", "y"} else "No"

    # === VALORES DESDE BD, SIN CONVERTIR A NÚMERO ===
    precio_venta = (row.get("precio_venta") or "").strip()
    precio_alquiler = (row.get("precio_alquiler") or "").strip()
    total_construido = (row.get("total_construido") or "").strip()

    def _is_empty(s: str) -> bool:
        if not s:
            return True
        s2 = s.lower().strip()
        return s2 in {"null", "none", "-", "consultar", "0"}

    # Operación + Valor: usamos el que esté cargado en la BD
    if not _is_empty(precio_alquiler):
        operacion = "alquiler"
        valor = precio_alquiler
    elif not _is_empty(precio_venta):
        operacion = "venta"
        valor = precio_venta
    else:
        operacion = "—"
        valor = "Consultar"

    # Superficie: usamos el texto tal cual; si viniera solo un número, agregamos m²
    if _is_empty(total_construido):
        sup_txt = "—"
    else:
        sup_txt = total_construido
        # Si es solo número, le agrego m²
        if sup_txt.replace(".", "", 1).isdigit():
            sup_txt = f"{sup_txt} m²"

    cod = row.get("id") or "—"

    # Ficha final (mismo formato que ya usás)
    return (
        f"*{tprop}*\n"
        f"{addr} (Zona: {zona})\n\n"
        f"• Operación: {operacion.capitalize()}\n"
        f"• Valor: {valor}\n"
        f"• Sup. construida: {sup_txt}\n"
        f"• Amb: {amb} | Dorm: {dorm} | Cochera: {coch_txt}\n"
        f"• Código: {cod}\n\n"
        f"{SITE_URL}"
    )

# === LINKS ===
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
    venta = str(row.get("precio_venta") or "").strip().lower()
    alqu = str(row.get("precio_alquiler") or "").strip().lower()
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

# === Validación operación vs propiedad ===
def _mismatch_msg(user_op: str, prop_op: str) -> str:
    return (
        f"Atenti 👀 La propiedad que enviaste está publicada para *{prop_op}*, "
        f"pero seleccionaste *{user_op}*.\n\n"
        "Te vuelvo al inicio así elegís la operación correcta o compartís otra propiedad.\n\n"
        + _say_menu()
    )

# === YES/NO y parsing guarantee ===
def _is_yes(t: str) -> bool:
    t = _strip_accents(t)
    return t in {"si", "sí", "ok", "dale", "claro", "perfecto", "de una", "si, claro", "listo", "afirmativo"}

def _is_no(t: str) -> bool:
    t = _strip_accents(t)
    return t in {"no", "nop", "no gracias", "nah", "negativo"}

def _parse_guarantee_choice(t: str) -> str:
    nt = _strip_accents(t)
    if nt.strip() in {"1", "1-", "1 -"} or "propietar" in nt or "caba" in nt:
        return "Propietario CABA"
    if nt.strip() in {"2", "2-", "2 -"} or "finaer" in nt or "caucion" in nt or "caución" in t.lower():
        return "Caución FINAER"
    if nt.strip() in {"3", "3-", "3 -"} or "ninguna" in nt or "no tengo" in nt or "sin garantia" in nt or "sin garantía" in t.lower():
        return "Ninguna"
    # por defecto, si menciona palabra "garantia" pero no clara, asumimos "Ninguna" para no trabar
    if "garantia" in nt or "garantía" in t.lower():
        return "Ninguna"
    return "Ninguna"

# =============== Intents básicos ===============
def _wants_reset(t: str) -> bool:
    t = _strip_accents(t)
    return t in {"reset", "reiniciar", "restart"}

def _is_rental_intent(t: str) -> bool:
    t = _strip_accents(t)
    keys = [
        "alquiler", "alquilo", "alquilar", "quiero alquilar",
        "busco alquiler", "estoy buscando alquiler", "rentar", "arrendar"
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

# ======== Tasación (7 pasos, sin cambios) ========
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

# =============== Conversación y estado ===============
def _reset(chat_id: str):
    STATE[chat_id] = {"stage": "menu"}

def _ensure_session(chat_id: str):
    if chat_id not in STATE:
        _reset(chat_id)

# =============== Endpoint principal ===============
@app.post("/qualify", response_model=QualifyOut)
async def qualify(body: QualifyIn) -> QualifyOut:
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

        # LINK directo en el primer mensaje
        row_link = _try_property_from_link_or_slug(text)
        if row_link:
            prop_op = _infer_intent_from_row(row_link) or "venta"
            if user_op and user_op != prop_op:
                _reset(chat_id)
                return QualifyOut(reply_text=_mismatch_msg(user_op, prop_op))
            s["prop_row"] = row_link
            s["intent"] = user_op or prop_op
            brief = render_property_card_db(row_link, intent=s["intent"])
            s["prop_brief"] = brief
            s["stage"] = "show_property_asked_qualify"
            # PRIMERA PREGUNTA según intent
            if s["intent"] == "alquiler":
                s["last_prompt"] = "qual_income"
                return QualifyOut(reply_text=brief + "\n\n" + _ask_income_question())
            else:
                s["last_prompt"] = "qual_disp_venta"
                return QualifyOut(reply_text=brief + "\n\n" + _ask_qualify_prompt("venta"))

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
                return QualifyOut(reply_text="¡Genial! Para la *tasación*, decime el *tipo de operación*: ¿venta o alquiler?")
            s["stage"] = "ask_zone_or_address"
            return QualifyOut(reply_text=_ask_zone_or_address())

        return QualifyOut(reply_text=_say_menu())

    # ========== TASACIÓN 7 PREGUNTAS (sin cambios) ==========
    if stage == "tas_op":
        t = _strip_accents(text)
        if "venta" in t:
            s["tas_op"] = "venta"
        elif "alquiler" in t or "renta" in t or "alquilar" in t:
            s["tas_op"] = "alquiler"
        else:
            return QualifyOut(reply_text="¿Me confirmás el *tipo de operación*? (venta o alquiler)")
        s["stage"] = "tas_prop"
        return QualifyOut(reply_text="Perfecto. ¿Cuál es el *tipo de propiedad*? (ej.: departamento, casa, local, oficina)")

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
        return QualifyOut(reply_text="Anotado. ¿Cuál es la *dirección exacta* del inmueble? (calle y número; si podés, piso/depto)")

    if stage == "tas_dir":
        if not _has_addr_number_strict(text):
            return QualifyOut(reply_text="¿Podés pasarme *calle y número*? Si tenés piso/depto, mejor.")
        s["tas_dir"] = text.strip()
        s["stage"] = "tas_exp"
        return QualifyOut(reply_text="¿La propiedad tiene *expensas*? Si tiene, ¿de cuánto es el *costo mensual* (ARS)? Si no, decime *no tiene*.")

    if stage == "tas_exp":
        t = _strip_accents(text)
        if any(x in t for x in ("no tiene", "sin expensas", "no")):
            s["tas_exp"] = "no tiene"
        else:
            val = _money_from_text(text)
            s["tas_exp"] = f"${val:,}".replace(",", ".") if val else (text.strip() or "no informado")
        s["stage"] = "tas_feat"
        return QualifyOut(reply_text="¿Dispone *balcón, patio, amenities o estudio de factibilidad*? Podés responder con una lista (ej.: “balcón y amenities”) o “no”.")

    if stage == "tas_feat":
        t = _strip_accents(text)
        feats = []
        if "balcon" in t or "balcón" in text.lower(): feats.append("balcón")
        if "patio" in t: feats.append("patio")
        if "amenities" in t: feats.append("amenities")
        if "estudio" in t or "factibilidad" in t: feats.append("estudio factibilidad")
        if t in {"no", "ninguno", "ninguna", "ningunos"}: feats = []
        s["tas_feat"] = ", ".join(feats) if feats else "no"
        s["stage"] = "tas_disp"
        return QualifyOut(reply_text="¡Último dato! ¿Cuál es tu *disponibilidad horaria* aproximada para que te contacte un asesor?")

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
        cierre = "Perfecto, con todos estos datos ya cuento con lo suficiente para derivarte con un asesor, muchisimas gracias por tu tiempo!"
        return QualifyOut(
            reply_text=cierre,
            vendor_push=True,
            vendor_message=resumen,
            closing_text=""
        )
    # ========== FIN TASACIÓN ==========

    # --- DIRECCIÓN / LINK ---
    if stage == "ask_zone_or_address":
        row_link = _try_property_from_link_or_slug(text)
        if row_link:
            intent_infer = _infer_intent_from_row(row_link) or s.get("intent") or "venta"
            if s.get("intent") and s["intent"] != intent_infer:
                user_op = s["intent"]
                _reset(chat_id)
                return QualifyOut(reply_text=_mismatch_msg(user_op, intent_infer))
            s["prop_row"] = row_link
            s["intent"] = s.get("intent") or intent_infer
            brief = render_property_card_db(row_link, intent=s["intent"])
            s["prop_brief"] = brief
            s["stage"] = "show_property_asked_qualify"
            if s["intent"] == "alquiler":
                s["last_prompt"] = "qual_income"
                return QualifyOut(reply_text=brief + "\n\n" + _ask_income_question())
            else:
                s["last_prompt"] = "qual_disp_venta"
                return QualifyOut(reply_text=brief + "\n\n" + _ask_qualify_prompt("venta"))

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
            if s.get("intent") and s["intent"] != intent_infer:
                user_op = s["intent"]
                _reset(chat_id)
                return QualifyOut(reply_text=_mismatch_msg(user_op, intent_infer))
            brief = render_property_card_db(row, intent=intent_infer)
            s["prop_row"] = row
            s["prop_brief"] = brief
            s["intent"] = intent_infer
            s["stage"] = "show_property_asked_qualify"
            if s["intent"] == "alquiler":
                s["last_prompt"] = "qual_income"
                return QualifyOut(reply_text=brief + "\n\n" + _ask_income_question())
            else:
                s["last_prompt"] = "qual_disp_venta"
                return QualifyOut(reply_text=brief + "\n\n" + _ask_qualify_prompt("venta"))

        return QualifyOut(
            reply_text=("No pude identificar la ficha a partir del texto. "
                        "¿Podés confirmarme la *dirección exacta* tal como figura en la publicación?")
        )

       # --- CALIFICACIÓN ---
    if stage == "show_property_asked_qualify":
        intent = s.get("intent", "alquiler")
        nt = _strip_accents(text)

        # ==== ALQUILER: flujo en 2 pasos + disponibilidad ====
        if intent == "alquiler":
            # Paso 1: Ingresos (excluyente)
            if s.get("last_prompt") == "qual_income":
                if _is_no(text):
                    s["stage"] = "done"
                    return QualifyOut(
                        reply_text=("Gracias por la info. Para *alquiler* es un requisito excluyente contar con "
                                    "*ingresos demostrables* que tripliquen el valor del alquiler. "
                                    "Cuando cuentes con esa condición, ¡escribinos por acá y seguimos!"),
                        closing_text=_farewell(),
                    )
                if _is_yes(text) or re.search(r"(ingreso|recibo|demostrable|monotrib|dependencia)", nt):
                    # Avanza al paso 2: garantía (no excluyente)
                    s["last_prompt"] = "qual_guarantee"
                    return QualifyOut(reply_text=_ask_guarantee_question())
                # Si la respuesta es ambigua, repreguntamos
                return QualifyOut(reply_text="¿Podés confirmarme si *contás con ingresos demostrables* que tripliquen el alquiler? Respondé *sí* o *no*.")

            # Paso 2: Garantía (no excluyente) -> luego disponibilidad
            if s.get("last_prompt") == "qual_guarantee":
                garantia = _parse_guarantee_choice(text)
                s["garantia"] = garantia  # registro opcional
                s["last_prompt"] = "qual_disp_alq"
                return QualifyOut(reply_text=_ask_disponibilidad())

            # Paso 3: Disponibilidad (no excluyente)
            if s.get("last_prompt") == "qual_disp_alq":
                s["disp_alquiler"] = text.strip() or "no informado"
                s["stage"] = "ask_handover"
                s.pop("last_prompt", None)
                return QualifyOut(
                    reply_text=("Perfecto 😊 ¿Querés que te contacte un asesor humano por este WhatsApp para avanzar?")
                )

        # ==== VENTAS: solo disponibilidad ====
        if intent == "venta":
            if s.get("last_prompt") != "qual_disp_venta":
                s["last_prompt"] = "qual_disp_venta"
                return QualifyOut(reply_text=_ask_disponibilidad())
            else:
                # Guardamos disponibilidad y avanzamos
                s["disp_venta"] = text.strip() or "no informado"
                s["stage"] = "ask_handover"
                s.pop("last_prompt", None)
                return QualifyOut(
                    reply_text=("Perfecto 😊 ¿Querés que te contacte un asesor humano por este WhatsApp para avanzar?")
                )

    # --- CONTACTO CON ASESOR ---
    if stage == "ask_handover":
        s.pop("last_prompt", None)

        if _is_yes(text):
            s["stage"] = "done"

            # Disponibilidad (si fue informada)
            disp = ""
            if s.get("disp_alquiler"):
                disp = f"Disponibilidad: {s['disp_alquiler']}\n"
            elif s.get("disp_venta"):
                disp = f"Disponibilidad: {s['disp_venta']}\n"

            # Operación y garantía (si corresponde)
            op_line = ""
            if s.get("intent"):
                op_line = f"Operación seleccionada: {s['intent'].capitalize()}\n"

            gar_line = ""
            if s.get("intent") == "alquiler" and s.get("garantia"):
                gar_line = f"Garantía: {s['garantia']}\n"

            vendor_msg = (
                "Lead calificado desde WhatsApp.\n"
                f"Chat: {chat_id}\n"
                f"{op_line}"
                f"{gar_line}"
                f"{disp}"
                f"{s.get('prop_brief','')}\n"
            )

            return QualifyOut(
                reply_text="Perfecto, te derivo con un asesor humano que te contactará por acá. ¡Gracias!",
                vendor_push=True,
                vendor_message=vendor_msg,
                closing_text=_farewell(),
            )

        if _is_no(text):
            s["stage"] = "done"
            return QualifyOut(
                reply_text=("¡Gracias por tu consulta! Quedamos a disposición por cualquier otra duda.\n"
                            "Cuando quieras retomar, escribí *reset* y arrancamos desde cero."),
                closing_text=_farewell(),
            )

        return QualifyOut(
            reply_text="¿Querés que te contacte un asesor humano por este WhatsApp para avanzar? Respondé *sí* o *no*."
        )
