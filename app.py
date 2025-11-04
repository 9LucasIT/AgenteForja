# app.py
import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

import mysql.connector
from mysql.connector.connection import MySQLConnection
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
SITE_URL = os.getenv("SITE_URL", "https://www.veglienzone.com.ar/").strip()

# Estado muy simple en memoria por chatId
STATE: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="FastAPI WhatsApp Agent (DB)", version="2025-11-03")

# ─────────────────────────────────────────────────────────────
# I/O Models
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
# Utils de texto
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


def _ask_qualify_prompt(op: str) -> str:
    if op == "alquiler":
        return ("Para avanzar con el alquiler, ¿contás con *ingresos demostrables* que tripliquen el costo y alguna "
                "*garantía* (caución FINAER / propietario / garantía propietaria)?")
    else:
        return ("Para avanzar con la venta, ¿la operación sería *contado* o *financiado*? "
                "¿Tenés alguna *seña* o *reserva* prevista? (contame brevemente)")


def _farewell() -> str:
    return "Perfecto, quedo atento a tus consultas. ¡Gracias por escribir! 😊"


# ─────────────────────────────────────────────────────────────
# DB: conexión y helpers
# ─────────────────────────────────────────────────────────────
def _build_mysql_params() -> Dict[str, Any]:
    """
    Soporta:
      - MYSQL_URL = mysql://user:pass@host:port/db
      - o variables sueltas: MYSQLHOST, MYSQLPORT, MYSQLUSER, MYSQLPASSWORD, MYSQL_DATABASE
    """
    url = os.getenv("MYSQL_URL") or os.getenv("MYSQL_PUBLIC_URL")  # por si Railway expone este
    if url:
        p = urlparse(url)
        return {
            "host": p.hostname,
            "port": p.port or 3306,
            "user": p.username,
            "password": p.password,
            "database": (p.path or "/").lstrip("/"),
            "autocommit": True,
        }

    return {
        "host": os.getenv("MYSQLHOST"),
        "port": int(os.getenv("MYSQLPORT", "3306")),
        "user": os.getenv("MYSQLUSER"),
        "password": os.getenv("MYSQLPASSWORD") or os.getenv("MYSQL_ROOT_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE") or os.getenv("MYSQLDATABASE"),
        "autocommit": True,
    }


def _get_conn() -> MySQLConnection:
    params = _build_mysql_params()
    return mysql.connector.connect(**params)


def _fetch_dicts(cursor) -> List[Dict[str, Any]]:
    cols = [c[0] for c in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def _sql_like(s: str) -> str:
    return f"%{s}%"


def _db_search_candidates(raw_text: str, op: Optional[str]) -> List[Dict[str, Any]]:
    """
    Trae candidatos por dirección usando LIKE y pequeñas variantes.
    Filtra por operación si corresponde:
      - alquiler => precio_alquiler > 0
      - venta    => precio_venta > 0
    """
    text = re.sub(r"\b(al|altura)\b", "", raw_text, flags=re.I).strip()
    like1 = _sql_like(text)
    # sólo calle (si no hay número)
    only_street = " ".join(re.findall(r"[A-Za-zÁÉÍÓÚÑáéíóúñ\.]+", text)).strip()
    like2 = _sql_like(only_street) if only_street and only_street != text else None

    where = ["direccion LIKE %s"]
    params = [like1]

    if like2:
        where.append("direccion LIKE %s")
        params.append(like2)

    if op == "alquiler":
        where.append("COALESCE(precio_alquiler,0) > 0")
    elif op == "venta":
        where.append("COALESCE(precio_venta,0) > 0")

    sql = (
        "SELECT id, direccion, zona, tipo_propiedad, ambientes, dormitorios, cochera, "
        "precio_venta, precio_alquiler, total_construido "
        "FROM propiedades "
        f"WHERE {' AND '.join(['(' + ' OR '.join(where[:2]) + ')'] + where[2:])} "
        "LIMIT 40"
        if like2 else
        "SELECT id, direccion, zona, tipo_propiedad, ambientes, dormitorios, cochera, "
        "precio_venta, precio_alquiler, total_construido "
        "FROM propiedades "
        f"WHERE {' AND '.join(where)} "
        "LIMIT 40"
    )

    with _get_conn() as cn:
        cur = cn.cursor()
        cur.execute(sql, params)
        return _fetch_dicts(cur)


def _best_address_match(raw_text: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    q = _strip_accents(raw_text)
    best, best_score = None, 0.0
    for r in candidates:
        addr = _strip_accents(r.get("direccion") or "")
        score = SequenceMatcher(None, q, addr).ratio()
        if score > best_score:
            best, best_score = r, score
    return best if best_score >= 0.55 else None


def _render_card_from_row(r: Dict[str, Any], op: Optional[str]) -> str:
    direccion = r.get("direccion") or "Sin dirección"
    zona = r.get("zona") or "—"
    tipo = r.get("tipo_propiedad") or "Propiedad"
    amb = r.get("ambientes") or 0
    dorm = r.get("dormitorios") or 0
    coch = r.get("cochera")
    coch_txt = "Sí" if (isinstance(coch, (int, float)) and coch > 0) or str(coch).lower() in {"1", "true", "si", "sí"} else "No"
    m2 = r.get("total_construido") or 0

    pv = r.get("precio_venta")
    pa = r.get("precio_alquiler")

    def fmt_money(v):
        if v is None:
            return "—"
        try:
            v = float(v)
            if v >= 1000:
                return f"USD {int(v):,}".replace(",", ".")
            return f"USD {v}"
        except Exception:
            return str(v)

    precio = fmt_money(pa if op == "alquiler" else pv if op == "venta" else (pa or pv))

    lines = [
        f"*{tipo}*",
        f"{direccion} ({zona})",
        "",
        f"• Ambientes: {amb} | Dorm: {dorm} | Sup: {m2} m²",
        f"• Cochera: {coch_txt}",
        f"• Precio: {precio}",
        "",
        SITE_URL,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Intents y helpers de conversación
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
        "alquiler", "alquilo", "alquilar", "quiero alquilar", "busco alquiler",
        "estoy buscando alquiler", "rentar", "rento", "arrendar"
    ]
    return any(k in t for k in keys) or t.strip() in {"1", "1-", "1 -", "alquileres"}


def _is_sale_intent(t: str) -> bool:
    t = _strip_accents(t)
    keys = ["venta", "vender", "comprar", "compro", "quiero comprar"]
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
# Endpoint principal /qualify
# ─────────────────────────────────────────────────────────────
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

    # ── MENU → detectar intención
    if stage == "menu":
        if not text:
            return QualifyOut(reply_text=_say_menu())

        if _is_rental_intent(text):
            s["intent"] = "alquiler"
            s["stage"] = "ask_zone_or_address"
            return QualifyOut(reply_text=_ask_zone_or_address())

        if _is_sale_intent(text):
            s["intent"] = "venta"
            s["stage"] = "ask_zone_or_address"
            return QualifyOut(reply_text=_ask_zone_or_address())

        if _is_valuation_intent(text):
            s["intent"] = "tasacion"
            s["stage"] = "tasacion_address"
            return QualifyOut(
                reply_text=("¡Genial! Para la *tasación*, pasame la *dirección exacta* y el *tipo de propiedad* "
                            "(ej.: departamento 2 amb en Palermo).")
            )

        return QualifyOut(reply_text=_say_menu())

    # ── TASACIONES
    if stage == "tasacion_address":
        # Guardamos los datos y pedimos contacto
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

    # ── PREGUNTA DIRECCIÓN / ZONA para ALQUILER o VENTA
    if stage == "ask_zone_or_address":
        intent = s.get("intent")  # alquiler / venta

        if _is_zone_search(text):
            s["stage"] = "done"
            msg = (
                "Perfecto. Te dejo el link donde están todas nuestras propiedades para que puedas ver si alguna te interesa:\n"
                f"{SITE_URL}\n\n"
                "Cualquier consulta puntual de una ficha me escribís por acá."
            )
            return QualifyOut(reply_text=msg, closing_text=_farewell())

        # Buscar en DB por dirección
        candidates = await run_in_threadpool(_db_search_candidates, text, intent)
        best = _best_address_match(text, candidates)
        if best:
            s["prop_row"] = best
            s["stage"] = "show_property_asked_qualify"
            brief = _render_card_from_row(best, intent)
            return QualifyOut(reply_text=brief + "\n\n" + _ask_qualify_prompt(intent))

        return QualifyOut(
            reply_text=("No pude identificar la ficha a partir del texto. "
                        "¿Podés confirmarme la *dirección exacta* (calle y número) "
                        "o, si querés, mirá el catálogo y reenviame el link de la ficha que te interese:\n"
                        f"{SITE_URL}")
        )

    # ── Mostrar propiedad y calificar (alquiler/venta)
    if stage == "show_property_asked_qualify":
        intent = s.get("intent")
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
                    reply_text=("¡Genial! Con esos datos podés calificar. "
                                "¿Querés que te contacte un asesor humano por este WhatsApp para avanzar?")
                )

            return QualifyOut(
                reply_text=("Para avanzar necesito confirmar: ¿tenés *ingresos demostrables* que tripliquen el costo "
                            "y alguna *garantía* (caución FINAER / propietario / garantía propietaria)? "
                            "Respondé *sí* o contame qué te falta.")
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

    # ── Confirmar derivación a humano
    if stage == "ask_handover":
        if _is_yes(text):
            s["stage"] = "done"
            row = s.get("prop_row", {})
            brief = _render_card_from_row(row, s.get("intent"))
            vendor_msg = f"Lead calificado desde WhatsApp\nChat: {chat_id}\n{brief}"
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
    # NO exponemos credenciales
    return {
        "SITE_URL": SITE_URL,
        "sessions": len(STATE),
        "db_host_set": bool(os.getenv("MYSQL_URL") or os.getenv("MYSQLHOST")),
        "db_name": os.getenv("MYSQL_DATABASE") or os.getenv("MYSQLDATABASE"),
    }


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
