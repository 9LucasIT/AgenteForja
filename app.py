# app.py
import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

import httpx  # queda por compatibilidad del proyecto
from fastapi import FastAPI
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
SITE_URL = os.getenv("SITE_URL", "https://www.veglienzone.com.ar/").strip()

# Preferencia: DATABASE_URL (mysql+pymysql://user:pass@host:port/db)
DATABASE_URL = os.getenv("DATABASE_URL", "") or os.getenv("MYSQL_URL", "")
MYSQL_HOST = os.getenv("MYSQLHOST") or os.getenv("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQLPORT") or os.getenv("MYSQL_PORT") or "3306")
MYSQL_USER = os.getenv("MYSQLUSER") or os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQLPASSWORD") or os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQLDATABASE") or os.getenv("MYSQL_DATABASE")
MYSQL_TABLE = os.getenv("MYSQL_TABLE", "propiedades")  # ← tu tabla principal

# Sesión simple por chatId en memoria
STATE: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="FastAPI WhatsApp Agent (DB)", version="2025-11-03")

# ─────────────────────────────────────────────────────────────
# Entrada/Salida
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# Helpers de texto
# ─────────────────────────────────────────────────────────────
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


def _ask_qualify_prompt(intent: str) -> str:
    if intent == "alquiler":
        return (
            "Para avanzar, ¿contás con *ingresos demostrables* que tripliquen el costo y alguna "
            "*garantía* (preferentemente de CABA: caución FINAER, propietario o garantía propietaria)?"
        )
    else:
        return (
            "Para avanzar, ¿la operación sería *contado* o *financiado*? "
            "¿Tenés prevista alguna *seña* o *reserva*?"
        )


def _farewell() -> str:
    return "Perfecto, quedo atento a tus consultas. ¡Gracias por escribir! 😊"


# ─────────────────────────────────────────────────────────────
# Conexión MySQL (no rompe si falta el driver)
# ─────────────────────────────────────────────────────────────
try:
    import pymysql
    from pymysql.cursors import DictCursor
    PYM_AVAILABLE = True
except Exception:
    PYM_AVAILABLE = False


def _parse_db_url(url: str):
    """
    Admite mysql:// ó mysql+pymysql://
    Devuelve (host, port, user, password, db)
    """
    if not url:
        return None
    u = urlparse(url)
    host = u.hostname
    port = u.port or 3306
    user = u.username
    pwd = u.password
    db = u.path.lstrip("/") if u.path else None
    return host, port, user, pwd, db


def _db_params():
    if DATABASE_URL:
        parsed = _parse_db_url(DATABASE_URL)
        if parsed:
            return {
                "host": parsed[0],
                "port": parsed[1],
                "user": parsed[2],
                "password": parsed[3],
                "database": parsed[4],
            }
    # fallback por variables sueltas
    return {
        "host": MYSQL_HOST,
        "port": MYSQL_PORT,
        "user": MYSQL_USER,
        "password": MYSQL_PASSWORD,
        "database": MYSQL_DB,
    }


def _safe_connect():
    """
    Intenta conectar. Si falla o no hay driver, devuelve None (sin romper el flujo).
    """
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


# ─────────────────────────────────────────────────────────────
# Búsqueda por dirección en BD
# ─────────────────────────────────────────────────────────────
def _build_like_patterns(raw: str) -> List[str]:
    """
    Genera variantes tipo LIKE para mejorar recall.
    """
    text = raw.strip()
    # quitar ' al ' / 'altura'
    text_no_al = re.sub(r"\b(al|altura)\b", "", text, flags=re.I).strip()

    # separar calle y número (si es que hay)
    num_match = re.search(r"\d{1,5}", text)
    number = num_match.group(0) if num_match else ""
    street = re.sub(r"\d{1,5}", "", text).strip()

    pats = []
    # completo
    pats.append(f"%{text}%")
    if text_no_al and text_no_al != text:
        pats.append(f"%{text_no_al}%")
    # solo calle
    if street:
        pats.append(f"%{street}%")
    # calle + número pegados
    if street and number:
        pats.append(f"%{street} {number}%")
        pats.append(f"%{street}%{number}%")
        pats.append(f"%{number}%{street}%")
    # número solo (por si la carga quedó tipo "Depto en Junín al 600")
    if number:
        pats.append(f"%{number}%")
    # evitar duplicados manteniendo orden
    seen = set()
    unique = []
    for p in pats:
        if p not in seen:
            unique.append(p); seen.add(p)
    return unique


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
                # si la tabla no existe u otro error, salimos
                return rows
    return rows


def search_db_by_address(raw_text: str) -> Optional[dict]:
    """
    Busca la mejor coincidencia por dirección en la base MySQL.
    Devuelve un dict con las columnas conocidas o None si no encuentra,
    sin lanzar excepciones hacia arriba.
    """
    conn = _safe_connect()
    if not conn:
        return None
    try:
        patterns = _build_like_patterns(raw_text)
        # 1) tabla principal
        candidates = _fetch_candidates_from_table(conn, MYSQL_TABLE, patterns)
        # 2) fallback a 'propiedad' si vacío
        if not candidates and MYSQL_TABLE != "propiedad":
            candidates = _fetch_candidates_from_table(conn, "propiedad", patterns)

        if not candidates:
            return None

        # ranking fuzzy por dirección
        q_norm = _strip_accents(raw_text)
        best, best_score = None, 0.0
        for r in candidates:
            addr = _strip_accents(r.get("direccion") or "")
            score = SequenceMatcher(None, q_norm, addr).ratio()
            if score > best_score:
                best, best_score = r, score

        return best if best_score >= 0.55 else None
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# Render de ficha desde BD
# ─────────────────────────────────────────────────────────────
def _fmt_money(v) -> str:
    if v in (None, "", "0", 0):
        return "—"
    try:
        v = float(v)
        if v >= 1000:
            return f"USD {int(v):,}".replace(",", ".")
        return f"USD {v}"
    except Exception:
        return str(v)


def render_property_card_db(row: dict, intent: str) -> str:
    """
    row contiene: id, direccion, zona, tipo_propiedad, ambientes,
    dormitorios, cochera, precio_venta, precio_alquiler, total_construido
    """
    addr = row.get("direccion") or "Sin dirección"
    zona = row.get("zona") or "—"
    tprop = row.get("tipo_propiedad") or "Propiedad"
    amb = row.get("ambientes") or 0
    dorm = row.get("dormitorios") or 0
    coch = row.get("cochera")
    coch_txt = "Sí" if str(coch).strip() in {"1", "si", "sí", "true", "True"} else "No"
    m2 = row.get("total_construido") or 0
    cod = row.get("id") or "—"

    if intent == "alquiler":
        price_txt = _fmt_money(row.get("precio_alquiler"))
        op = "Alquiler"
    else:
        price_txt = _fmt_money(row.get("precio_venta"))
        op = "Venta"

    return (
        f"*{tprop}*\n"
        f"{addr} (Zona: {zona})\n\n"
        f"• Operación: {op}\n"
        f"• Valor: {price_txt}\n"
        f"• Sup. construída: {m2} m²\n"
        f"• Amb: {amb} | Dorm: {dorm} | Cochera: {coch_txt}\n"
        f"• Código: {cod}\n\n"
        f"{SITE_URL}"
    )


# ─────────────────────────────────────────────────────────────
# Motor de conversación / estados
# ─────────────────────────────────────────────────────────────
def _reset(chat_id: str):
    STATE[chat_id] = {"stage": "menu"}


def _ensure_session(chat_id: str):
    if chat_id not in STATE:
        _reset(chat_id)


def _wants_reset(t: str) -> bool:
    t = _strip_accents(t)
    return t in {"reset", "reiniciar", "restart"}


def _is_yes(t: str) -> bool:
    t = _strip_accents(t)
    return t in {"si", "sí", "ok", "dale", "claro", "perfecto", "de una", "si, claro"}


def _is_no(t: str) -> bool:
    t = _strip_accents(t)
    return t in {"no", "nop", "no gracias", "nah"}


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


# ─────────────────────────────────────────────────────────────
# Endpoint principal
# ─────────────────────────────────────────────────────────────
@app.post("/qualify", response_model=QualifyOut)
async def qualify(body: QualifyIn) -> QualifyOut:
    chat_id = body.chatId
    text = (body.message or "").strip()

    _ensure_session(chat_id)
    s = STATE[chat_id]

    # RESET
    if _wants_reset(text):
        _reset(chat_id)
        return QualifyOut(reply_text=_say_menu())

    stage = s.get("stage", "menu")

    # ── stage: menu → detectar intención
    if stage == "menu":
        if not text:
            return QualifyOut(reply_text=_say_menu())

        if _is_rental_intent(text) or _is_sale_intent(text) or _is_valuation_intent(text):
            s["intent"] = "alquiler" if _is_rental_intent(text) else "venta" if _is_sale_intent(text) else "tasacion"
            s["stage"] = "ask_zone_or_address" if s["intent"] != "tasacion" else "tasacion_address"
            if s["intent"] == "tasacion":
                return QualifyOut(
                    reply_text=("¡Genial! Para la *tasación*, pasame la *dirección exacta* y el *tipo de propiedad* "
                                "(ej.: departamento 2 amb en Palermo).")
                )
            return QualifyOut(reply_text=_ask_zone_or_address())

        return QualifyOut(reply_text=_say_menu())

    # ── tasación
    if stage == "tasacion_address":
        s["tasacion_input"] = text
        s["stage"] = "tasacion_contact"
        return QualifyOut(
            reply_text=("Gracias. ¿Podrías dejarme un *teléfono o email* para coordinar la visita de tasación? "
                        "También decime *franja horaria* de preferencia.")
        )

    if stage == "tasacion_contact":
        s["tasacion_contact"] = text
        s["stage"] = "done"
        vendor_msg = (
            "Solicitud de TASACIÓN desde WhatsApp\n"
            f"Chat: {chat_id}\n"
            f"Datos: {s.get('tasacion_input','(sin detalle)')}\n"
            f"Contacto: {s.get('tasacion_contact','(sin contacto)')}"
        )
        return QualifyOut(
            reply_text=("¡Perfecto! Derivo tu consulta para coordinar la tasación. "
                        "En breve un asesor te contactará por este WhatsApp."),
            vendor_push=True,
            vendor_message=vendor_msg,
            closing_text=_farewell(),
        )

    # ── stage: ask_zone_or_address
    if stage == "ask_zone_or_address":
        # caso “solo zona/barrio”
        if _is_zone_search(text):
            s["stage"] = "done"
            msg = (
                "Perfecto. Te dejo el link donde están todas nuestras propiedades para que puedas ver si alguna te interesa:\n"
                f"{SITE_URL}\n\n"
                "Cualquier consulta puntual de una ficha me escribís por acá."
            )
            return QualifyOut(reply_text=msg, closing_text=_farewell())

        # Buscar en BD por dirección
        intent = s.get("intent", "alquiler")
        row = search_db_by_address(text)

        if row:
            brief = render_property_card_db(row, intent=intent)
            s["prop_row"] = row
            s["prop_brief"] = brief
            s["stage"] = "show_property_asked_qualify"
            return QualifyOut(reply_text=brief + "\n\n" + _ask_qualify_prompt(intent))

        # No hubo match
        return QualifyOut(
            reply_text=(
                "No pude identificar la ficha a partir del texto. "
                "¿Podés confirmarme la *dirección exacta* tal como figura en la publicación?"
            )
        )

    # ── stage: show_property_asked_qualify
    if stage == "show_property_asked_qualify":
        intent = s.get("intent", "alquiler")
        nt = _strip_accents(text)

        if intent == "alquiler":
            has_income = bool(re.search(r"(ingreso|recibo|demostrable|monotrib|dependencia)", nt))
            has_guarantee = bool(re.search(r"(garantia|caucion|propietari[ao]|finaer)", nt))

            if _is_no(text):
                s["stage"] = "done"
                return QualifyOut(
                    reply_text="Entiendo. Si en otro momento contás con los requisitos, ¡escribinos por acá!",
                    closing_text=_farewell(),
                )

            if has_income and has_guarantee:
                s["stage"] = "ask_handover"
                return QualifyOut(
                    reply_text=(
                        "¡Genial! Con esos datos podés calificar. "
                        "¿Querés que te contacte un asesor humano por este WhatsApp para avanzar?"
                    )
                )

            return QualifyOut(
                reply_text=(
                    "Para avanzar necesito confirmar: ¿tenés *ingresos demostrables* que tripliquen el costo y alguna "
                    "*garantía* (caución FINAER / propietario / garantía propietaria)? Respondé *sí* o contame qué te falta."
                )
            )

        # Venta
        if intent == "venta":
            if _is_no(text):
                s["stage"] = "done"
                return QualifyOut(
                    reply_text="Perfecto, si necesitás ver otras opciones o comparar, escribime por acá.",
                    closing_text=_farewell(),
                )

            talked_money = bool(re.search(r"(contado|financ|credito|hipoteca|se.na|reserva|oferta)", nt))
            if talked_money or _is_yes(text):
                s["stage"] = "ask_handover"
                return QualifyOut(
                    reply_text=("Gracias por la info. ¿Querés que te contacte un asesor para coordinar visita "
                                "y conversar condiciones de compra?")
                )

            return QualifyOut(
                reply_text=("¿La operación sería *contado* o *financiado*? ¿Tenés prevista *seña* o *reserva*?")
            )

    # ── stage: ask_handover
    if stage == "ask_handover":
        if _is_yes(text):
            s["stage"] = "done"
            vendor_msg = f"Lead calificado desde WhatsApp.\nChat: {chat_id}\n{ s.get('prop_brief','') }"
            return QualifyOut(
                reply_text="Perfecto, te derivo con un asesor humano que te contactará por acá. ¡Gracias!",
                vendor_push=True,
                vendor_message=vendor_msg,
                closing_text=_farewell(),
            )
        if _is_no(text):
            s["stage"] = "done"
            return QualifyOut(
                reply_text="¡Sin problema! Si más adelante querés avanzar, escribinos por acá.",
                closing_text=_farewell(),
            )
        return QualifyOut(reply_text="¿Querés que te contacte un asesor humano por este WhatsApp para avanzar? (sí/no)")

    # fallback
    _reset(chat_id)
    return QualifyOut(reply_text=_say_menu())


# ─────────────────────────────────────────────────────────────
# Health & Debug
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug")
def debug():
    params = _db_params()
    return {
        "db_driver_available": PYM_AVAILABLE,
        "db_host_set": bool(params.get("host")),
        "db_name": params.get("database"),
        "table": MYSQL_TABLE,
        "memory_sessions": len(STATE),
    }


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
