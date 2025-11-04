# app.py
import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

# --- NEW: DB ---
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # reservado p/ futuro
TOKKO_API_KEY = os.getenv("TOKKO_API_KEY", "").strip()
SITE_URL = os.getenv("SITE_URL", "https://www.veglienzone.com.ar/").rstrip("/")

# NEW: conexión DB (Railway) - opcional
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
engine: Optional[Engine] = None
if DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        print("DB: conexión OK")
    except Exception as e:
        print(f"DB: no disponible: {e}")
        engine = None

# Nota: mantenemos memoria simple en RAM por chatId
STATE: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="FastAPI WhatsApp Agent", version="2025-11-02")

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
# Helpers: texto / normalización
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


# ─────────────────────────────────────────────────────────────
# Detectar pistas desde el texto (url / publication_code / reference_code)
# ─────────────────────────────────────────────────────────────
def _extract_tokko_clues_from_text(t: str):
    """
    Devuelve dict con posibles pistas:
      - publication_code: '7256624'
      - reference_code: 'VAP7450253' (u otros como VGA7395075)
      - url: la url capturada
    """
    out = {"publication_code": None, "reference_code": None, "url": None}
    url_match = re.search(r'https?://[^\s>]+', t)
    if url_match:
        out["url"] = url_match.group(0)

    # /p/7256624 -> 7256624
    m_pub = re.search(r'/p/(\d+)', t)
    if m_pub:
        out["publication_code"] = m_pub.group(1)

    # Códigos referencia tipo VAP7378801 / VGA7395075 etc
    m_ref = re.search(r'\b([A-Z]{3}\d{6,})\b', t)
    if m_ref:
        out["reference_code"] = m_ref.group(1)

    return out


# ─────────────────────────────────────────────────────────────
# Tokko API helpers
# ─────────────────────────────────────────────────────────────
def _tokko_headers() -> Dict[str, str]:
    # Tokko usa api_key por query, no por header, pero dejamos estándar
    return {"Accept": "application/json"}


async def _tokko_get(url: str, params: dict) -> dict:
    params = dict(params or {})
    if TOKKO_API_KEY:
        params.setdefault("api_key", TOKKO_API_KEY)

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params, headers=_tokko_headers())
        if r.status_code == 200:
            try:
                return r.json() or {}
            except Exception:
                return {}
        return {}


async def get_tokko_by_publication_code(code: str) -> Optional[dict]:
    """Prueba publication_code, code y como fallback search por el número."""
    if not code:
        return None
    base = "https://api.tokkobroker.com/api/v1/properties/"
    for key in ("publication_code", "code"):
        js = await _tokko_get(base, {key: code, "limit": 1})
        objs = js.get("objects") or []
        if objs:
            return objs[0]
    # Fallback search
    js = await _tokko_get(base, {"search": code, "limit": 1})
    objs = js.get("objects") or []
    return objs[0] if objs else None


async def get_tokko_by_reference_code(ref: str) -> Optional[dict]:
    """Búsqueda por código referencia (VAP*, VGA*, etc) usando search."""
    if not ref:
        return None
    base = "https://api.tokkobroker.com/api/v1/properties/"
    js = await _tokko_get(base, {"search": ref, "limit": 1})
    objs = js.get("objects") or []
    return objs[0] if objs else None


async def get_tokko_by_url(full_url: str) -> Optional[dict]:
    """Si llega URL, probamos /p/<code> y luego search por la URL completa."""
    if not full_url:
        return None
    m = re.search(r'/p/(\d+)', full_url)
    if m:
        found = await get_tokko_by_publication_code(m.group(1))
        if found:
            return found
    base = "https://api.tokkobroker.com/api/v1/properties/"
    js = await _tokko_get(base, {"search": full_url, "limit": 1})
    objs = js.get("objects") or []
    return objs[0] if objs else None


async def search_tokko_by_address(raw_text: str) -> Optional[dict]:
    """
    Búsqueda por dirección robusta con varios intentos y fuzzy match.
    No filtramos por tipo operación para no perder coincidencias.
    """
    text = raw_text.strip()
    text_no_al = re.sub(r"\b(al|altura)\b", "", text, flags=re.I).strip()

    base = "https://api.tokkobroker.com/api/v1/properties/"
    candidates = []

    # 1) consulta completa
    js = await _tokko_get(base, {"search": text, "limit": 20})
    candidates.extend(js.get("objects") or [])

    # 2) sin "al/altura"
    if text_no_al and text_no_al != text:
        js = await _tokko_get(base, {"search": text_no_al, "limit": 20})
        candidates.extend(js.get("objects") or [])

    # 3) si no hay número, probamos solo la calle
    if not re.search(r'\d{2,5}', text):
        calle = " ".join(re.findall(r"[A-Za-zÁÉÍÓÚÑáéíóúñ\.]+", text))
        if calle:
            js = await _tokko_get(base, {"search": calle.strip(), "limit": 20})
            candidates.extend(js.get("objects") or [])

    if not candidates:
        return None

    # Fuzzy
    text_norm = _strip_accents(text_no_al or text)
    best, best_score = None, 0.0
    for p in candidates:
        addr = _strip_accents(p.get("address") or p.get("display_address") or "")
        score = SequenceMatcher(None, text_norm, addr).ratio()
        if score > best_score:
            best, best_score = p, score

    return best if best_score >= 0.55 else None


# ─────────────────────────────────────────────────────────────
# NEW: Fallback a base local (Railway) por dirección
# ─────────────────────────────────────────────────────────────
def _fetch_local_by_address(addr: str, operacion: Optional[str]) -> Optional[dict]:
    """
    Busca en tabla `propiedades` por dirección (LIKE) y opcionalmente por operación ('alquiler'/'venta').
    Espera columnas: codigo_ref, direccion, operacion, moneda, precio, link, surface_cubierta, dormitorios, banos, ambientes
    """
    if not engine or not addr:
        return None
    try:
        like = f"%{addr.strip()}%"
        if operacion in ("alquiler", "venta"):
            q = text(
                "SELECT codigo_ref, direccion, operacion, moneda, precio, link, "
                "surface_cubierta, dormitorios, banos, ambientes "
                "FROM propiedades "
                "WHERE activa=1 AND operacion=:op AND direccion LIKE :dir "
                "ORDER BY updated_at DESC LIMIT 1"
            )
            row = engine.execute(q, {"op": operacion, "dir": like}).fetchone()
        else:
            q = text(
                "SELECT codigo_ref, direccion, operacion, moneda, precio, link, "
                "surface_cubierta, dormitorios, banos, ambientes "
                "FROM propiedades "
                "WHERE activa=1 AND direccion LIKE :dir "
                "ORDER BY updated_at DESC LIMIT 1"
            )
            row = engine.execute(q, {"dir": like}).fetchone()

        if not row:
            return None

        (codigo_ref, direccion, op, moneda, precio, link, sup, dorms, banos, amb) = row
        price_lbl = None
        if precio is not None:
            try:
                price_lbl = f"{moneda or ''} {float(precio):,.0f}".replace(",", ".")
            except Exception:
                price_lbl = f"{moneda or ''} {precio}"

        return {
            "title": f"Ref. {codigo_ref}" if codigo_ref else "Propiedad",
            "address": direccion or "",
            "operation": (op or "").capitalize(),
            "price": price_lbl,
            "surface_covered": sup or 0,
            "bedrooms": dorms or 0,
            "bathrooms": banos or 0,
            "rooms": amb or 0,
            "web_url": link or SITE_URL,
            "code": codigo_ref or "",
        }
    except Exception as e:
        print(f"Local DB lookup error: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Render de ficha breve
# ─────────────────────────────────────────────────────────────
def _fmt_money(v) -> str:
    if v is None:
        return "—"
    try:
        v = float(v)
        if v >= 1000:
            return f"USD {int(v):,}".replace(",", ".")
        return f"USD {v}"
    except Exception:
        return str(v)


def render_property_card(p: dict) -> str:
    title = p.get("title") or p.get("operation") or "Propiedad"
    addr = p.get("address") or p.get("display_address") or "Sin dirección"
    op = (p.get("operation") or p.get("operations", [{}])[0].get("operation_type") if isinstance(p.get("operations"), list) else p.get("operation")) or ""
    op = (op or "").capitalize()
    price = p.get("price") or p.get("price_operation") or p.get("price_from") or p.get("price_usd")
    price_txt = _fmt_money(price)
    m2 = p.get("surface_covered") or p.get("surface_total") or 0
    dorms = p.get("bedrooms") or 0
    baths = p.get("bathrooms") or 0
    amb = p.get("rooms") or 0
    code = p.get("code") or p.get("publication_code") or p.get("slug") or "—"

    link = p.get("web_url") or p.get("url") or SITE_URL

    return (
        f"*{title}*\n"
        f"{addr}\n\n"
        f"• Operación: {op or '—'}\n"
        f"• Valor: {price_txt}\n"
        f"• Sup. cubierta: {m2} m²\n"
        f"• Dorm: {dorms} | Baños: {baths} | Amb: {amb}\n"
        f"• Código: {code}\n\n"
        f"{link}"
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
    keys = ["alquiler", "alquilo", "me gustaria alquilar", "busco alquiler", "estoy buscando alquiler"]
    return any(k in t for k in keys) or t.strip() in {"1", "1-", "1 -", "alquileres"}


def _is_sale_intent(t: str) -> bool:
    t = _strip_accents(t)
    keys = ["venta", "vender", "compro", "comprar"]
    return any(k in t for k in keys) or t.strip() in {"2", "2-", "2 -", "ventas"}


def _is_valuation_intent(t: str) -> bool:
    t = _strip_accents(t)
    keys = ["tasacion", "tasación", "tasar", "3", "3-", "3 -", "tasaciones"]
    return any(k in t for k in keys)


def _is_zone_search(t: str) -> bool:
    """Frases de 'no tengo dirección/link, estoy averiguando por zona/barrio'."""
    nt = _strip_accents(t)
    patterns = [
        r"\bno tengo (la )?direccion\b",
        r"\bno tengo link\b",
        r"\bsolo (zona|barrio)\b",
        r"\bestoy averiguando\b",
        r"\bbusco.*(zona|barrio)\b",
    ]
    return any(re.search(p, nt) for p in patterns)


def _looks_like_address(t: str) -> bool:
    nt = _strip_accents(t)
    return bool(re.search(r"\b(av|avenida|calle|pasaje|pje|ruta|camino|cabildo|santa fe|humboldt|yatay|moreno)\b", nt)) \
           and bool(re.search(r"\b\d{2,5}\b", nt))


# ─────────────────────────────────────────────────────────────
# Ruta principal usada por n8n (Webhook → Qualify)
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

    # Estados
    stage = s.get("stage", "menu")

    # ── stage: menu → detectar intención
    if stage == "menu":
        # primer mensaje o saludo
        if not text:
            return QualifyOut(reply_text=_say_menu())

        if _is_rental_intent(text) or _is_sale_intent(text) or _is_valuation_intent(text):
            s["intent"] = "alquiler" if _is_rental_intent(text) else "venta" if _is_sale_intent(text) else "tasacion"
            s["stage"] = "ask_zone_or_address"
            return QualifyOut(reply_text=_ask_zone_or_address())

        # Saluditos / cualquier cosa ajena → mostrar menú
        return QualifyOut(reply_text=_say_menu())

    # ── stage: ask_zone_or_address
    if stage == "ask_zone_or_address":
        intent = s.get("intent", "alquiler")

        # Si es consulta por zona/barrio → enviar link del sitio y cerrar
        if _is_zone_search(text):
            s["stage"] = "done"
            msg = (
                "Perfecto. Te dejo el link donde están todas nuestras propiedades para que puedas ver si alguna te interesa:\n"
                f"{SITE_URL}\n\n"
                "Cualquier consulta puntual de una ficha me escribís por acá."
            )
            return QualifyOut(reply_text=msg, closing_text=_farewell())

        # Si trae link/dirección/código → intentamos ficha exacta
        clues = _extract_tokko_clues_from_text(text)
        prop = None
        if clues["publication_code"]:
            prop = await get_tokko_by_publication_code(clues["publication_code"])
        if not prop and clues["reference_code"]:
            prop = await get_tokko_by_reference_code(clues["reference_code"])
        if not prop and clues["url"]:
            prop = await get_tokko_by_url(clues["url"])
        if not prop and _looks_like_address(text):
            # NEW: fallback a tu base local si Tokko no devolvió nada
            prop = _fetch_local_by_address(text, "alquiler" if intent == "alquiler" else "venta") or _fetch_local_by_address(text, None)
        if not prop and not clues["url"]:
            # intentar por dirección general en Tokko
            prop = await search_tokko_by_address(text)

        if prop:
            brief = render_property_card(prop)
            s["prop_id"] = prop.get("id")
            s["prop_brief"] = brief

            if intent == "alquiler":
                s["stage"] = "show_property_asked_qualify"
                return QualifyOut(reply_text=brief + "\n\n" + _ask_qualify_prompt())

            if intent == "venta":
                s["stage"] = "venta_visit"
                return QualifyOut(
                    reply_text=brief + "\n\n¿Querés coordinar una *visita* o que un *asesor* te contacte para más detalles?"
                )

            if intent == "tasacion":
                s["stage"] = "tasacion_detalle"
                return QualifyOut(
                    reply_text=brief + "\n\nPara la *tasación*, contame: tipo de propiedad (casa/departamento/cochera), "
                                       "antigüedad aproximada y estado general."
                )

        # No hubo match
        if intent == "alquiler":
            return QualifyOut(
                reply_text=("No pude identificar la ficha a partir del texto. "
                            "¿Podés confirmarme la *dirección exacta* o reenviarme el *link* completo?")
            )
        if intent == "venta":
            return QualifyOut(
                reply_text=("No ubico esa ficha puntual. Si querés, contame *zona/barrio* y *presupuesto* para mostrarte opciones, "
                            "o pasame el *link* de la publicación.")
            )
        # tasación
        return QualifyOut(
            reply_text=("Para tasar necesito la *dirección exacta* y tipo de propiedad. Si tenés el link de la ficha, mejor 😉")
        )

    # ── stage: show_property_asked_qualify → (ALQUILER) evaluar requisitos y derivar
    if stage == "show_property_asked_qualify":
        nt = _strip_accents(text)
        has_income = bool(re.search(r"(ingreso|recibo|demostrable|monotrib|dependencia)", nt))
        has_guarantee = bool(re.search(r"(garantia|caucion|propietari[ao]|finaer)", nt))

        # si dice NO explícito a requisitos
        if _is_no(text):
            STATE[chat_id]["stage"] = "done"
            return QualifyOut(
                reply_text="Entiendo. Si en otro momento contás con los requisitos, ¡escribinos por acá!",
                closing_text=_farewell(),
            )

        if has_income and has_guarantee:
            STATE[chat_id]["stage"] = "ask_handover"
            return QualifyOut(
                reply_text=(
                    "¡Genial! Con esos datos podés calificar. "
                    "¿Querés que te contacte un asesor humano por este WhatsApp para avanzar?"
                )
            )

        # aún no se entiende → repregunta breve
        return QualifyOut(
            reply_text=(
                "Para avanzar necesito confirmar: ¿tenés *ingresos demostrables* que tripliquen el costo y alguna "
                "*garantía* (caución FINAER / propietario / garantía propietaria)? Respondé *sí* o contame qué te falta."
            )
        )

    # ── stage: ask_handover (ALQUILER)
    if stage == "ask_handover":
        if _is_yes(text):
            STATE[chat_id]["stage"] = "done"
            vendor_msg = f"Lead calificado desde WhatsApp.\nChat: {chat_id}\n{ STATE[chat_id].get('prop_brief','') }"
            return QualifyOut(
                reply_text="Perfecto, te derivo con un asesor humano que te contactará por acá. ¡Gracias!",
                vendor_push=True,
                vendor_message=vendor_msg,
                closing_text=_farewell(),
            )
        if _is_no(text):
            STATE[chat_id]["stage"] = "done"
            return QualifyOut(
                reply_text="¡Sin problema! Si más adelante querés avanzar, escribinos por acá.",
                closing_text=_farewell(),
            )
        return QualifyOut(reply_text="¿Querés que te contacte un asesor humano por este WhatsApp para avanzar? (sí/no)")

    # ── stage: venta_visit (VENTAS)
    if stage == "venta_visit":
        if _is_yes(text):
            STATE[chat_id]["stage"] = "done"
            vendor_msg = f"Lead *VENTA* desde WhatsApp.\nChat: {chat_id}\n{ STATE[chat_id].get('prop_brief','') }"
            return QualifyOut(
                reply_text="¡Genial! Te contacto un asesor por este WhatsApp para coordinar visita y detalles.",
                vendor_push=True,
                vendor_message=vendor_msg,
                closing_text=_farewell(),
            )
        if _is_no(text):
            STATE[chat_id]["stage"] = "done"
            return QualifyOut(
                reply_text="¡Perfecto! Cualquier duda o si querés coordinar más adelante, escribinos por acá.",
                closing_text=_farewell(),
            )
        return QualifyOut(reply_text="¿Coordinamos *visita* o preferís que un *asesor* te escriba? (sí/no)")

    # ── stage: tasacion_detalle (TASACIONES)
    if stage == "tasacion_detalle":
        # con cualquier respuesta, derivamos al asesor con el detalle capturado
        STATE[chat_id]["stage"] = "done"
        vendor_msg = (
            "Lead *TASACIÓN* desde WhatsApp.\n"
            f"Chat: {chat_id}\n"
            f"Detalle brindado por el cliente: {text}\n"
            f"{STATE[chat_id].get('prop_brief','')}"
        )
        return QualifyOut(
            reply_text="Gracias. Con esos datos un asesor se pondrá en contacto para avanzar con la tasación.",
            vendor_push=True,
            vendor_message=vendor_msg,
            closing_text=_farewell(),
        )

    # ── stage: done (o desconocido) → menú
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
    # Ojo: no exponemos API keys
    db_ok = False
    try:
        if engine:
            with engine.connect() as c:
                c.execute(text("SELECT 1"))
            db_ok = True
    except Exception:
        db_ok = False

    return {
        "OPENAI_MODEL": OPENAI_MODEL,
        "TOKKO_API_KEY_set": bool(TOKKO_API_KEY),
        "SITE_URL": SITE_URL,
        "memory_sessions": len(STATE),
        "db_ok": db_ok,
    }


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
