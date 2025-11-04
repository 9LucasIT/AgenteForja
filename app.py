# app.py
import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse

import mysql.connector
from fastapi import FastAPI
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
SITE_URL = os.getenv("SITE_URL", "https://www.veglienzone.com.ar/").strip()

# Memoria simple en RAM por chatId
STATE: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="FastAPI WhatsApp Agent", version="2025-11-03")


# ─────────────────────────────────────────────────────────────
# IO Models
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
# Helpers de texto / estado
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


def _ask_qualify_prompt() -> str:
    return (
        "Para avanzar, ¿contás con *ingresos demostrables* que tripliquen el costo y alguna "
        "*garantía* (preferentemente de CABA: seguro de caución FINAER, propietario, o garantía propietaria)?"
    )


def _farewell() -> str:
    return "Perfecto, quedo atento a tus consultas. ¡Gracias por escribir! 😊"


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
    nt = _strip_accents(t)
    keys = [
        "alquiler", "alquilo", "alquilar", "busco alquiler", "estoy buscando alquiler",
        "quiero alquilar", "busco depto en alquiler", "depto en alquiler"
    ]
    return any(k in nt for k in keys) or nt.strip() in {"1", "1-", "1 -", "alquileres"}


def _is_sale_intent(t: str) -> bool:
    nt = _strip_accents(t)
    keys = ["venta", "vender", "compro", "comprar", "busco para comprar", "quiero comprar"]
    return any(k in nt for k in keys) or nt.strip() in {"2", "2-", "2 -", "ventas"}


def _is_valuation_intent(t: str) -> bool:
    nt = _strip_accents(t)
    keys = ["tasacion", "tasación", "tasar", "presupuesto", "avaluo"]
    return any(k in nt for k in keys) or nt.strip() in {"3", "3-", "3 -", "tasaciones"}


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


def _reset(chat_id: str):
    STATE[chat_id] = {"stage": "menu"}


def _ensure_session(chat_id: str):
    if chat_id not in STATE:
        _reset(chat_id)


# ─────────────────────────────────────────────────────────────
# Conexión MySQL (Railway)
# ─────────────────────────────────────────────────────────────
def _parse_database_url(url: str) -> Tuple[str, int, str, str, str]:
    """
    mysql://user:pass@host:port/db  → (host, port, user, pass, db)
    """
    u = urlparse(url)
    return (u.hostname, u.port or 3306, u.username, u.password, u.path.lstrip("/"))


def get_conn():
    # Prioriza DATABASE_URL si existe
    db_url = os.getenv("DATABASE_URL", "").strip()
    if db_url:
        host, port, user, password, database = _parse_database_url(db_url)
    else:
        host = os.getenv("MYSQLHOST", "localhost")
        port = int(os.getenv("MYSQLPORT", "3306"))
        user = os.getenv("MYSQLUSER", "root")
        password = os.getenv("MYSQLPASSWORD", "")
        database = os.getenv("MYSQLDATABASE") or os.getenv("MYSQL_DATABASE") or ""

    return mysql.connector.connect(
        host=host, port=port, user=user, password=password, database=database,
        autocommit=True
    )


# ─────────────────────────────────────────────────────────────
# Búsqueda en BD por dirección (fuzzy)
# ─────────────────────────────────────────────────────────────
def _normalize_address_for_like(text: str) -> str:
    """
    Limpia palabras como 'al/altura' y conserva números si los hay.
    """
    t = text.strip()
    # quitar 'al' / 'altura'
    t = re.sub(r"\b(al|altura)\b", "", t, flags=re.I)
    return t.strip()


def _fetch_candidates_by_like(conn, addr_like: str):
    # LIKE insensible a mayúsculas y acentos usando COLLATE apropiado del server.
    # Si tu instancia es utf8mb4_general_ci por defecto, alcanza con LIKE.
    q = """
        SELECT id, direccion, zona, tipo_propiedad, ambientes, dormitorios, cochera,
               precio_venta, precio_alquiler, total_construido
        FROM propiedades
        WHERE direccion LIKE %s
        LIMIT 20;
    """
    with conn.cursor(dictionary=True) as cur:
        cur.execute(q, (f"%{addr_like}%",))
        return cur.fetchall() or []


def search_property_by_address_db(raw_text: str) -> Optional[dict]:
    """
    Hace 2-3 intentos de LIKE y elige el mejor por similitud.
    """
    addr = raw_text.strip()
    if not addr:
        return None

    addr_like = _normalize_address_for_like(addr)
    conn = get_conn()
    try:
        candidates = _fetch_candidates_by_like(conn, addr_like)

        # si no hay, probamos sólo la parte "calle" (sin número) si existe
        if not candidates:
            m = re.match(r"([^\d]+)", addr_like)  # todo antes del primer número
            if m:
                calle = m.group(1).strip()
                if calle:
                    candidates = _fetch_candidates_by_like(conn, calle)

        if not candidates:
            return None

        # elegir mejor por similitud con direccion
        addr_norm = _strip_accents(addr_like)
        best, best_score = None, 0.0
        for row in candidates:
            dnorm = _strip_accents(row.get("direccion") or "")
            score = SequenceMatcher(None, addr_norm, dnorm).ratio()
            if score > best_score:
                best, best_score = row, score

        return best if best else None
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# Render de ficha a partir de BD
# ─────────────────────────────────────────────────────────────
def _pick_operation_and_price(row: dict) -> Tuple[str, str]:
    """
    Determina Operación y Valor según columnas.
    - Si precio_alquiler tiene texto/no es nulo → Alquiler + ese valor
    - Si precio_venta tiene texto/no es nulo → Venta + ese valor
    - Si ambos existen, prioriza Alquiler (caso raro)
    - Si ninguno, 'Consultar'
    """
    pv = (row.get("precio_venta") or "").strip()
    pa = (row.get("precio_alquiler") or "").strip()

    # Tratar cadenas tipo 'NULL'
    pv = "" if _strip_accents(pv) == "null" else pv
    pa = "" if _strip_accents(pa) == "null" else pa

    if pa:
        return "Alquiler", pa
    if pv:
        return "Venta", pv
    return "—", "Consultar"


def render_property_card_db(row: dict) -> str:
    # Campos base
    titulo = row.get("tipo_propiedad") or "Propiedad"
    direccion = row.get("direccion") or "Sin dirección"
    zona = row.get("zona") or "—"

    ambientes = row.get("ambientes")
    dormitorios = row.get("dormitorios")
    cochera_flag = row.get("cochera")

    # Formateos seguros
    try:
        ambientes_txt = str(int(ambientes)) if ambientes is not None else "0"
    except Exception:
        ambientes_txt = str(ambientes) if ambientes is not None else "0"

    try:
        dorm_txt = str(int(dormitorios)) if dormitorios is not None else "0"
    except Exception:
        dorm_txt = str(dormitorios) if dormitorios is not None else "0"

    try:
        cochera_txt = "Sí" if (int(cochera_flag) if cochera_flag is not None else 0) else "No"
    except Exception:
        cochera_txt = "Sí" if str(cochera_flag).strip() not in {"0", "", "None", "null"} else "No"

    operacion, valor = _pick_operation_and_price(row)

    # Superficie (viene ya con texto “NN metros cuadrados”)
    sup = (row.get("total_construido") or "").strip()
    sup = sup if sup else "—"

    link = SITE_URL

    return (
        f"*{titulo}*\n"
        f"{direccion} (Zona: {zona})\n\n"
        f"• Operación: {operacion}\n"
        f"• Valor: {valor}\n"
        f"• Sup. construida: {sup}\n"
        f"• Amb: {ambientes_txt} | Dorm: {dorm_txt} | Cochera: {cochera_txt}\n\n"
        f"{link}"
    )


# ─────────────────────────────────────────────────────────────
# Motor conversacional
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
            s["stage"] = "ask_zone_or_address"
            return QualifyOut(reply_text=_ask_zone_or_address())

        return QualifyOut(reply_text=_say_menu())

    # ── stage: ask_zone_or_address
    if stage == "ask_zone_or_address":
        # Caso zona/barrio → link general + cierre
        if _is_zone_search(text):
            s["stage"] = "done"
            msg = (
                "Perfecto. Te dejo el link donde están todas nuestras propiedades para que puedas ver si alguna te interesa:\n"
                f"{SITE_URL}\n\n"
                "Cualquier consulta puntual de una ficha me escribís por acá."
            )
            return QualifyOut(reply_text=msg, closing_text=_farewell())

        # Caso dirección → BD
        prop = search_property_by_address_db(text)

        if prop:
            brief = render_property_card_db(prop)
            s["prop_id"] = prop.get("id")
            s["prop_brief"] = brief
            s["stage"] = "show_property_asked_qualify"
            return QualifyOut(reply_text=brief + "\n\n" + _ask_qualify_prompt())

        # Sin match
        return QualifyOut(
            reply_text=(
                "No pude identificar la ficha a partir del texto. "
                "¿Podés confirmarme la *dirección exacta* (calle + número) o reenviarme el *link* de la ficha?"
            )
        )

    # ── stage: show_property_asked_qualify → evaluar requisitos
    if stage == "show_property_asked_qualify":
        nt = _strip_accents(text)
        has_income = bool(re.search(r"(ingreso|recibo|demostrable|monotrib|dependencia)", nt))
        has_guarantee = bool(re.search(r"(garantia|caucion|propietari[ao]|finaer)", nt))

        if _is_no(text):
            s["stage"] = "done"
            return QualifyOut(
                reply_text="Entiendo. Si en otro momento contás con los requisitos, ¡escribinos por acá!",
                closing_text=_farewell(),
            )

        if _is_yes(text) or (has_income and has_guarantee):
            s["stage"] = "ask_handover"
            return QualifyOut(
                reply_text=(
                    "¡Genial! Con esos datos podés calificar. "
                    "¿Querés que te contacte un asesor humano por este WhatsApp para avanzar?"
                )
            )

        # Si todavía no queda claro → repregunta breve
        return QualifyOut(
            reply_text=(
                "Para avanzar necesito confirmar: ¿tenés *n) ingresos demostrables* que tripliquen el costo y alguna "
                "*garantía* (caución FINAER / propietario / garantía propietaria)? Respondé *sí* o contame qué te falta."
            )
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

    # ── fallback
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
    return {
        "SITE_URL": SITE_URL,
        "sessions": len(STATE),
        "db_from": "DATABASE_URL" if os.getenv("DATABASE_URL", "") else "MYSQL* envs",
    }


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
