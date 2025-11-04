# app.py
import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

# --- DB ---
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOKKO_API_KEY = os.getenv("TOKKO_API_KEY", "").strip()
SITE_URL = os.getenv("SITE_URL", "https://www.veglienzone.com.ar/").rstrip("/")
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
# Helpers texto
# ─────────────────────────────────────────────────────────────
def _strip_accents(s: str) -> str:
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


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
# Claves desde el texto
# ─────────────────────────────────────────────────────────────
def _extract_tokko_clues_from_text(t: str):
    out = {"publication_code": None, "reference_code": None, "url": None}
    url_match = re.search(r'https?://[^\s>]+', t)
    if url_match:
        out["url"] = url_match.group(0)
    m_pub = re.search(r'/p/(\d+)', t)
    if m_pub:
        out["publication_code"] = m_pub.group(1)
    m_ref = re.search(r'\b([A-Z]{3}\d{6,})\b', t)
    if m_ref:
        out["reference_code"] = m_ref.group(1)
    return out


# ─────────────────────────────────────────────────────────────
# Tokko API
# ─────────────────────────────────────────────────────────────
def _tokko_headers() -> Dict[str, str]:
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
    if not code:
        return None
    base = "https://api.tokkobroker.com/api/v1/properties/"
    for key in ("publication_code", "code"):
        js = await _tokko_get(base, {key: code, "limit": 1})
        objs = js.get("objects") or []
        if objs:
            return objs[0]
    js = await _tokko_get(base, {"search": code, "limit": 1})
    objs = js.get("objects") or []
    return objs[0] if objs else None


async def get_tokko_by_reference_code(ref: str) -> Optional[dict]:
    if not ref:
        return None
    base = "https://api.tokkobroker.com/api/v1/properties/"
    js = await _tokko_get(base, {"search": ref, "limit": 1})
    objs = js.get("objects") or []
    return objs[0] if objs else None


async def get_tokko_by_url(full_url: str) -> Optional[dict]:
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
    text = _normalize_spaces(raw_text.strip())
    text_no_al = re.sub(r"\b(al|altura)\b", "", text, flags=re.I).strip()

    base = "https://api.tokkobroker.com/api/v1/properties/"
    candidates = []

    js = await _tokko_get(base, {"search": text, "limit": 20})
    candidates.extend(js.get("objects") or [])
    if text_no_al and text_no_al != text:
        js = await _tokko_get(base, {"search": text_no_al, "limit": 20})
        candidates.extend(js.get("objects") or [])
    if not re.search(r'\d{2,5}', text):
        calle = " ".join(re.findall(r"[A-Za-zÁÉÍÓÚÑáéíóúñ\.]+", text))
        if calle:
            js = await _tokko_get(base, {"search": calle.strip(), "limit": 20})
            candidates.extend(js.get("objects") or [])

    if not candidates:
        return None

    text_norm = _strip_accents(text_no_al or text)
    best, best_score = None, 0.0
    for p in candidates:
        addr = _strip_accents(p.get("address") or p.get("display_address") or "")
        score = SequenceMatcher(None, text_norm, addr).ratio()
        if score > best_score:
            best, best_score = p, score
    return best if best_score >= 0.55 else None


# ─────────────────────────────────────────────────────────────
# Fallback local DB por dirección
# ─────────────────────────────────────────────────────────────
def _like_token(s: str) -> str:
    return f"%{_normalize_spaces(_strip_accents(s))}%"


def _fetch_local_by_address(addr: str, operacion: Optional[str]) -> Optional[dict]:
    if not engine or not addr:
        return None
    try:
        addr = _normalize_spaces(addr)
        addr_no_al = re.sub(r"\b(al|altura)\b", "", addr, flags=re.I).strip()

        # tokens
        street_only = " ".join(re.findall(r"[A-Za-zÁÉÍÓÚÑáéíóúñ\.]+", addr_no_al))
        number = re.search(r"\b(\d{1,6})\b", addr_no_al)
        num = number.group(1) if number else None

        # normalización lado SQL (quita acentos y baja a minúscula)
        def sql_norm(col: str) -> str:
            # reemplazos de acentos seguros en MySQL
            repl = (
                "REPLACE(REPLACE(REPLACE(REPLACE(REPLACE("
                f"LOWER({col}), 'á','a'),'é','e'),'í','i'),'ó','o'),'ú','u')"
            )
            return repl

        where = []
        params = {}

        if street_only:
            where.append(f"{sql_norm('direccion')} LIKE :calle")
            params["calle"] = _like_token(street_only)

        if num:
            where.append(f"{sql_norm('direccion')} LIKE :num")
            params["num"] = f"%{num}%"

        if not where:
            where.append(f"{sql_norm('direccion')} LIKE :fallback")
            params["fallback"] = _like_token(addr_no_al or addr)

        if operacion in ("alquiler", "venta"):
            where.append(f"{sql_norm('operacion')} LIKE :op")
            params["op"] = _like_token(operacion)

        where_sql = " AND ".join(where)

        q = text(
            "SELECT codigo_ref, direccion, operacion, moneda, precio, link, "
            "surface_cubierta, dormitorios, banos, ambientes "
            "FROM propiedades "
            f"WHERE activa=1 AND {where_sql} "
            "ORDER BY updated_at DESC LIMIT 1"
        )
        with engine.begin() as conn:
            row = conn.execute(q, params).fetchone()

        if not row:
            return None

        (codigo_ref, direccion, op, moneda, precio, link, sup, dorms, banos, amb) = row
        price_lbl = None
        if precio is not None:
            try:
                price_lbl = f"{(moneda or '').strip()} {float(precio):,.0f}".replace(",", ".")
            except Exception:
                price_lbl = f"{(moneda or '').strip()} {precio}"

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
# Render tarjeta
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

    # operación Tokko o local
    op = p.get("operation")
    if not op and isinstance(p.get("operations"), list) and p["operations"]:
        op = p["operations"][0].get("operation_type")
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
        f"*{title}*\n{addr}\n\n"
        f"• Operación: {op or '—'}\n"
        f"• Valor: {price_txt}\n"
        f"• Sup. cubierta: {m2} m²\n"
        f"• Dorm: {dorms} | Baños: {baths} | Amb: {amb}\n"
        f"• Código: {code}\n\n{link}"
    )


# ─────────────────────────────────────────────────────────────
# Intenciones y patrones
# ─────────────────────────────────────────────────────────────
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
    return bool(
        re.search(r"\b(alquiler|alquilo|alquilar|busco alquiler|estoy buscando alquiler|quiero alquilar|rentar|renta)\b", nt)
        or nt.strip() in {"1", "1-", "1 -", "alquileres"}
    )


def _is_sale_intent(t: str) -> bool:
    nt = _strip_accents(t)
    return bool(
        re.search(r"\b(venta|vender|comprar|compro|quiero comprar)\b", nt)
        or nt.strip() in {"2", "2-", "2 -", "ventas"}
    )


def _is_valuation_intent(t: str) -> bool:
    nt = _strip_accents(t)
    return bool(
        re.search(r"\b(tasacion|tasación|tasar)\b", nt)
        or nt.strip() in {"3", "3-", "3 -", "tasaciones"}
    )


def _is_zone_search(t: str) -> bool:
    nt = _strip_accents(t)
    patterns = [
        r"\bno tengo (la )?direccion\b",
        r"\bno tengo link\b",
        r"\bsolo (zona|barrio)\b",
        r"\bestoy averiguando\b",
        r"\bbusco.*(zona|barrio)\b",
        r"\bno tengo (dato|propiedad)\b",
    ]
    return any(re.search(p, nt) for p in patterns)


def _looks_like_address(t: str) -> bool:
    nt = _strip_accents(t)
    has_street = bool(re.search(r"\b(av|avenida|calle|pasaje|pje|ruta|camino|cabildo|santa fe|humboldt|yatay|moreno|junin|junun|cabildo)\b", nt))
    has_num = bool(re.search(r"\b\d{2,6}\b", nt))
    return has_street and has_num


# ─────────────────────────────────────────────────────────────
# Endpoint principal
# ─────────────────────────────────────────────────────────────
@app.post("/qualify", response_model=QualifyOut)
async def qualify(body: QualifyIn) -> QualifyOut:
    chat_id = body.chatId
    text = (body.message or "").strip()

    STATE.setdefault(chat_id, {"stage": "menu"})
    s = STATE[chat_id]

    if _wants_reset(text):
        STATE[chat_id] = {"stage": "menu"}
        return QualifyOut(reply_text=_say_menu())

    stage = s.get("stage", "menu")

    # MENU
    if stage == "menu":
        if not text:
            return QualifyOut(reply_text=_say_menu())

        if _is_rental_intent(text) or _is_sale_intent(text) or _is_valuation_intent(text):
            s["intent"] = "alquiler" if _is_rental_intent(text) else "venta" if _is_sale_intent(text) else "tasacion"
            s["stage"] = "ask_zone_or_address"
            return QualifyOut(reply_text=_ask_zone_or_address())

        return QualifyOut(reply_text=_say_menu())

    # PREGUNTA DIRECCIÓN/ZONA
    if stage == "ask_zone_or_address":
        intent = s.get("intent", "alquiler")

        if _is_zone_search(text):
            s["stage"] = "done"
            msg = (
                "Perfecto. Te dejo el link donde están todas nuestras propiedades para que puedas ver si alguna te interesa:\n"
                f"{SITE_URL}\n\n"
                "Cualquier consulta puntual de una ficha me escribís por acá."
            )
            return QualifyOut(reply_text=msg, closing_text=_farewell())

        clues = _extract_tokko_clues_from_text(text)
        prop = None
        if clues["publication_code"]:
            prop = await get_tokko_by_publication_code(clues["publication_code"])
        if not prop and clues["reference_code"]:
            prop = await get_tokko_by_reference_code(clues["reference_code"])
        if not prop and clues["url"]:
            prop = await get_tokko_by_url(clues["url"])

        # Address first: local DB si parece dirección
        if not prop and _looks_like_address(text):
            prop = _fetch_local_by_address(text, "alquiler" if intent == "alquiler" else "venta") or _fetch_local_by_address(text, None)

        if not prop and not clues["url"]:
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
                return QualifyOut(reply_text=brief + "\n\n¿Querés coordinar una *visita* o que un *asesor* te contacte para más detalles?")
            if intent == "tasacion":
                s["stage"] = "tasacion_detalle"
                return QualifyOut(
                    reply_text=brief + "\n\nPara la *tasación*, contame: tipo de propiedad (casa/departamento/cochera), "
                                       "antigüedad aproximada y estado general."
                )

        # No hubo match
        if intent == "alquiler":
            return QualifyOut(reply_text="No pude identificar la ficha a partir del texto. ¿Podés confirmarme la *dirección exacta* o reenviarme el *link* completo?")
        if intent == "venta":
            return QualifyOut(reply_text="No ubico esa ficha puntual. Si querés, pasame la *dirección exacta* o el *link*; también podés decirme zona y presupuesto.")
        return QualifyOut(reply_text="Para tasar necesito la *dirección exacta* y tipo de propiedad. Si tenés el link de la ficha, mejor 😉")

    # ALQUILER → requisitos
    if stage == "show_property_asked_qualify":
        nt = _strip_accents(text)
        has_income = bool(re.search(r"(ingreso|recibo|demostrable|monotrib|dependencia)", nt))
        has_guarantee = bool(re.search(r"(garantia|caucion|propietari[ao]|finaer)", nt))

        if _is_no(text):
            s["stage"] = "done"
            return QualifyOut(reply_text="Entiendo. Si en otro momento contás con los requisitos, ¡escribinos por acá!", closing_text=_farewell())

        if has_income and has_guarantee:
            s["stage"] = "ask_handover"
            return QualifyOut(reply_text="¡Genial! Con esos datos podés calificar. ¿Querés que te contacte un asesor humano por este WhatsApp para avanzar?")

        return QualifyOut(reply_text="Para avanzar necesito confirmar: ¿tenés *ingresos demostrables* que tripliquen el costo y alguna *garantía* (FINAER/propietario/garantía propietaria)? Respondé *sí* o contame qué te falta.")

    # ALQUILER → handoff
    if stage == "ask_handover":
        if _is_yes(text):
            s["stage"] = "done"
            vendor_msg = f"Lead calificado desde WhatsApp.\nChat: {chat_id}\n{s.get('prop_brief','')}"
            return QualifyOut(reply_text="Perfecto, te derivo con un asesor humano que te contactará por acá. ¡Gracias!", vendor_push=True, vendor_message=vendor_msg, closing_text=_farewell())
        if _is_no(text):
            s["stage"] = "done"
            return QualifyOut(reply_text="¡Sin problema! Si más adelante querés avanzar, escribinos por acá.", closing_text=_farewell())
        return QualifyOut(reply_text="¿Querés que te contacte un asesor humano por este WhatsApp para avanzar? (sí/no)")

    # VENTAS
    if stage == "venta_visit":
        if _is_yes(text):
            s["stage"] = "done"
            vendor_msg = f"Lead *VENTA* desde WhatsApp.\nChat: {chat_id}\n{s.get('prop_brief','')}"
            return QualifyOut(reply_text="¡Genial! Te contacto un asesor por este WhatsApp para coordinar visita y detalles.", vendor_push=True, vendor_message=vendor_msg, closing_text=_farewell())
        if _is_no(text):
            s["stage"] = "done"
            return QualifyOut(reply_text="¡Perfecto! Cualquier duda o si querés coordinar más adelante, escribinos por acá.", closing_text=_farewell())
        return QualifyOut(reply_text="¿Coordinamos *visita* o preferís que un *asesor* te escriba? (sí/no)")

    # TASACIONES
    if stage == "tasacion_detalle":
        s["stage"] = "done"
        vendor_msg = f"Lead *TASACIÓN* desde WhatsApp.\nChat: {chat_id}\nDetalle brindado por el cliente: {text}\n{s.get('prop_brief','')}"
        return QualifyOut(reply_text="Gracias. Con esos datos un asesor se pondrá en contacto para avanzar con la tasación.", vendor_push=True, vendor_message=vendor_msg, closing_text=_farewell())

    # fallback
    STATE[chat_id] = {"stage": "menu"}
    return QualifyOut(reply_text=_say_menu())


# ─────────────────────────────────────────────────────────────
# Health / Debug
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug")
def debug():
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
