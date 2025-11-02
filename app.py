# app.py
import os
import re
import time
import json
import math
import httpx
import asyncio
import unicodedata
from difflib import SequenceMatcher
from typing import Optional, Dict, Any

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ==========
# Config
# ==========
TOKKO_API_KEY = os.getenv("TOKKO_API_KEY", "").strip()

GREETING = (
    "Gracias por contactarte con el área comercial de Veglienzone Gestión Inmobiliaria. "
    "¿Cómo podemos ayudarte hoy?\n"
    "1- Alquileres\n2- Ventas\n3- Tasaciones\n\n"
    "Nota: si en cualquier momento escribís *reset*, la conversación se reinicia desde cero."
)

SITE_LINK = "https://www.veglienzone.com.ar/"

# ==========
# App setup
# ==========
app = FastAPI(title="FastAPI Agent - Veglienzone")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========
# In-memory session (simple TTL)
# ==========
_SESS: Dict[str, Dict[str, Any]] = {}
_TTL_SECONDS = 60 * 60 * 6  # 6 horas


def _now() -> float:
    return time.time()


def _get_sess(chat_id: str) -> Dict[str, Any]:
    s = _SESS.get(chat_id)
    if not s or (_now() - s.get("_ts", 0) > _TTL_SECONDS):
        s = {
            "_ts": _now(),
            "stage": "start",              # start | asked_intent | waiting_area_or_address | show_property_asked_qualify | qualifying | after_qualified
            "intention": None,             # alquiler | venta | tasaciones
            "prop_id": None,               # id de propiedad en Tokko si se seleccionó una
            "prop_brief": None,            # resumen de la ficha
            "qualified": False,            # calificado o no
        }
        _SESS[chat_id] = s
    else:
        s["_ts"] = _now()
    return s


def _reset_sess(chat_id: str) -> Dict[str, Any]:
    if chat_id in _SESS:
        del _SESS[chat_id]
    return _get_sess(chat_id)


# ==========
# Utils de texto
# ==========
def _strip_accents(s: str) -> str:
    if not s:
        return ""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower()


def looks_like_no_address(text: str) -> bool:
    """
    Frases para 'no tengo dirección ni link' o 'estoy averiguando por zona/barrio'.
    """
    t = _strip_accents(text)
    patterns = [
        r"\bno tengo (direccion|link)\b",
        r"\bno tengo\b",
        r"\bno (tengo|poseo) (datos|info)\b",
        r"\bsolo (zona|barrio)\b",
        r"\b(estoy|ando) (averiguando|buscando) (por )?(zona|barrio)\b",
        r"\bzona\b",
        r"\bbarrio\b",
        r"\bcentro\b",
    ]
    return any(re.search(p, t) for p in patterns) and not re.search(r"\d{2,5}", t)


def detect_intention(text: str) -> Optional[str]:
    t = _strip_accents(text)
    if re.search(r"\b(alquiler|alquilar|rentar|alquilo)\b", t):
        return "alquiler"
    if re.search(r"\b(venta|vender|vendo)\b", t):
        return "venta"
    if re.search(r"\b(tasaci[oó]n|tasar|valuaci[oó]n)\b", t):
        return "tasaciones"
    return None


# ==========
# Tokko helpers
# ==========
def _tokko_headers():
    return {"X-Authorization": TOKKO_API_KEY} if TOKKO_API_KEY else {}


async def get_tokko_by_publication_code(code: str) -> Optional[dict]:
    """
    Busca ficha por código de publicación (el que aparece en /p/<code> del sitio)
    """
    if not code:
        return None
    url = "https://api.tokkobroker.com/api/v1/properties/"
    params = {"publication_code": code, "limit": 1}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params=params, headers=_tokko_headers())
        if r.status_code == 200:
            js = r.json()
            if js.get("objects"):
                return js["objects"][0]
    return None


async def get_tokko_by_url(full_url: str) -> Optional[dict]:
    """
    Fallback: intenta buscar por URL con 'search' (si Tokko lo indexa)
    """
    if not full_url:
        return None
    url = "https://api.tokkobroker.com/api/v1/properties/"
    q = full_url.split("?")[0]
    params = {"search": q, "limit": 1}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params=params, headers=_tokko_headers())
        if r.status_code == 200:
            js = r.json()
            if js.get("objects"):
                return js["objects"][0]
    return None


async def search_tokko_by_address(raw_text: str, op: Optional[str]) -> Optional[dict]:
    """
    Heurística por dirección con normalización y fuzzy.
    """
    text = raw_text.strip()
    # elimina 'al', 'altura', etc
    text_norm = _strip_accents(re.sub(r"\b(al|altura)\b", "", text, flags=re.I)).strip()

    params = {"search": text_norm, "limit": 15}
    if op == "alquiler":
        params["operation"] = "rent"
    elif op == "venta":
        params["operation"] = "sale"

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get("https://api.tokkobroker.com/api/v1/properties/", params=params, headers=_tokko_headers())
        if r.status_code != 200:
            return None
        data = r.json().get("objects", [])

    if not data:
        return None

    best, best_score = None, 0.0
    for p in data:
        addr = _strip_accents(p.get("address") or p.get("display_address") or "")
        score = SequenceMatcher(None, text_norm, addr).ratio()
        if score > best_score:
            best, best_score = p, score

    return best if best_score >= 0.6 else None


def render_property_card(p: dict) -> str:
    title = p.get("title") or (p.get("property_type") or "Propiedad").title()
    addr = p.get("address") or p.get("display_address") or "Dirección no disponible"
    op = (p.get("operation") or "").capitalize()
    price = p.get("price", "")
    currency = p.get("currency", "")
    price_str = f"{currency} {price}" if price and currency else (str(price) if price else "")
    m2 = p.get("covered_surface")
    rooms = p.get("rooms")
    baths = p.get("bathrooms")
    url = p.get("web_url") or p.get("permalink") or p.get("url") or SITE_LINK

    parts = [
        f"*{title}*",
        f"📍 {addr}",
        f"💼 {op}" if op else None,
        f"💲 {price_str}" if price_str else None,
        f"📏 {m2} m² cub." if m2 else None,
        f"🛏️ {rooms} dorm." if rooms is not None else None,
        f"🛁 {baths} baños" if baths is not None else None,
        f"🔗 {url}",
    ]
    return "\n".join([x for x in parts if x])


# ==========
# Modelos de E/S
# ==========
class QualifyReq(BaseModel):
    chatId: str = Field(..., description="ID de chat de WhatsApp (ej. 5493412...@c.us)")
    message: str = Field(..., description="Texto del usuario (o link)")
    isFromMe: Optional[bool] = False
    senderName: Optional[str] = ""


class QualifyResp(BaseModel):
    reply_text: str
    vendor_push: bool = False
    vendor_message: str = ""
    closing_text: str = ""


# ==========
# Handlers de intentos y estado
# ==========
def _menu() -> str:
    return GREETING


def _ask_area_or_address() -> str:
    return "¿Tenés dirección o link exacto de la propiedad, o estás averiguando por una zona/barrio?"


def _link_only_reply() -> str:
    # solo link del sitio y cierre cordial
    return (
        "Perfecto. Te dejo el link donde están todas nuestras propiedades para que veas si alguna te interesa:\n"
        f"{SITE_LINK}\n\n"
        "Si necesitás algo puntual, escribime por acá."
    )


def _ask_qualify_prompt() -> str:
    return (
        "Para avanzar con esta unidad, ¿contás con *ingresos demostrables* que tripliquen el alquiler y "
        "*garantía apta CABA* (Finaer / seguro de caución o propietaria)?"
    )


def _after_qualify_yes() -> str:
    return "Genial, ¿querés que te contacte un asesor humano por este WhatsApp para coordinar? (sí/no)"


def _after_qualify_no() -> str:
    return (
        "Entiendo. Si preferís, podés explorar el catálogo completo aquí:\n"
        f"{SITE_LINK}\n\n"
        "Quedo atento a cualquier consulta."
    )


def _cordial_close() -> str:
    return "¡Gracias por escribirnos! Cualquier otra consulta, estoy a disposición."


def _looks_yes(t: str) -> bool:
    tt = _strip_accents(t)
    return bool(re.search(r"\b(si|sí|dale|ok|de acuerdo|correcto)\b", tt))


def _looks_no(t: str) -> bool:
    tt = _strip_accents(t)
    return bool(re.search(r"\b(no|nop|no gracias|prefiero que no)\b", tt))


# ==========
# Core routing
# ==========
async def handle_message(chat_id: str, text: str, sender: str) -> QualifyResp:
    s = _get_sess(chat_id)
    t = text.strip()

    # Reset
    if re.search(r"\breset\b", _strip_accents(t)):
        _reset_sess(chat_id)
        return QualifyResp(reply_text=_menu())

    # 1) Si recién empieza o viene de reset → menú
    if s["stage"] == "start":
        s["stage"] = "asked_intent"
        return QualifyResp(reply_text=_menu())

    # 2) Si está eligiendo intención
    if s["stage"] == "asked_intent":
        intent = detect_intention(t)
        if intent:
            s["intention"] = intent
            s["stage"] = "waiting_area_or_address"
            return QualifyResp(reply_text=_ask_area_or_address())
        else:
            # si el usuario escribe algo tipo "busco alquilar", cae igual
            intent = detect_intention(t)
            if intent:
                s["intention"] = intent
                s["stage"] = "waiting_area_or_address"
                return QualifyResp(reply_text=_ask_area_or_address())
            # fallback: reenvía menú
            return QualifyResp(reply_text=_menu())

    # 3) Link explícito → resolver ficha exacta
    url_match = re.search(r'https?://[^\s]+', t)
    if url_match:
        url = url_match.group(0)
        prop = None
        m = re.search(r'/p/(\d+)', url)
        if "veglienzone.com.ar" in url and m:
            prop = await get_tokko_by_publication_code(m.group(1))
        if not prop:
            prop = await get_tokko_by_url(url)

        if prop:
            brief = render_property_card(prop)
            s["prop_id"] = prop.get("id")
            s["prop_brief"] = brief
            s["stage"] = "show_property_asked_qualify"
            return QualifyResp(reply_text=brief + "\n\n" + _ask_qualify_prompt())

        # si no matchea
        return QualifyResp(
            reply_text="No pude identificar la ficha a partir del link. ¿Podés confirmarme la *dirección exacta* o reenviarme el link completo?",
        )

    # 4) Si dice que no tiene dirección → sólo link del sitio
    if looks_like_no_address(t):
        # No modificamos intención; enviamos link y cerramos cordial
        s["stage"] = "after_qualified"  # consideramos cerrado el tramo de búsqueda
        return QualifyResp(reply_text=_link_only_reply())

    # 5) ¿Parece dirección? (calle + número o 'al 2300')
    maybe_addr = re.search(r'[A-Za-zÁÉÍÓÚÑáéíóúñ\.\s]+?\s+\d{2,5}', t) or \
                 re.search(r'[A-Za-zÁÉÍÓÚÑáéíóúñ\.\s]+?\s+al\s+\d{3,5}', t, flags=re.I)

    if s["stage"] in ("waiting_area_or_address", "asked_intent") and maybe_addr:
        prop = await search_tokko_by_address(t, s.get("intention"))
        if prop:
            brief = render_property_card(prop)
            s["prop_id"] = prop.get("id")
            s["prop_brief"] = brief
            s["stage"] = "show_property_asked_qualify"
            return QualifyResp(reply_text=brief + "\n\n" + _ask_qualify_prompt())
        else:
            return QualifyResp(
                reply_text="No encuentro esa dirección todavía. ¿Podés confirmarme *calle y altura* (ej.: *Av. Cabildo 2853*) o mandarme el link de la ficha?",
            )

    # 6) Si ya mostramos una ficha y estamos pidiendo calificar
    if s["stage"] == "show_property_asked_qualify":
        # Si responde que SÍ tiene ingresos + garantía, pasamos a preguntar derivación
        # Aceptamos respuestas libres; si sólo pone "sí", lo tomamos como OK
        if _looks_yes(t) or re.search(r"(ingreso|recibo|monotrib|sueldo).*(garant|finaer|cauci[oó]n|propietaria)", _strip_accents(t)):
            s["qualified"] = True
            s["stage"] = "after_qualified"
            return QualifyResp(reply_text=_after_qualify_yes())
        # Si dice que NO
        if _looks_no(t):
            s["qualified"] = False
            s["stage"] = "after_qualified"
            return QualifyResp(reply_text=_after_qualify_no())
        # Si escribe algo ambiguo, repreguntamos
        return QualifyResp(
            reply_text="¿Contás con *ingresos demostrables* que tripliquen el alquiler y *garantía apta CABA* (Finaer / seguro de caución o propietaria)? (sí/no)"
        )

    # 7) Después de calificar: ofrecer derivar si está calificado
    if s["stage"] == "after_qualified":
        if s.get("qualified"):
            if _looks_yes(t):
                # Acá recién empujamos al asesor humano: vendor_push = True
                vendor_msg = f"Nuevo lead calificado desde WhatsApp.\n\nCliente: {sender}\nChatId: {chat_id}\nIntención: {s.get('intention')}\n\n{(s.get('prop_brief') or '')}"
                # No respondemos nada extraño al cliente, solo confirmamos:
                return QualifyResp(
                    reply_text="Perfecto, te va a escribir un asesor por acá. ¡Gracias!",
                    vendor_push=True,
                    vendor_message=vendor_msg
                )
            if _looks_no(t):
                return QualifyResp(reply_text=_cordial_close())
            # sino, repregunta cortés
            return QualifyResp(reply_text="¿Querés que te contacte un asesor humano por este WhatsApp? (sí/no)")
        else:
            # no calificado → cierre cordial
            return QualifyResp(reply_text=_cordial_close())

    # 8) Fallbacks: si todavía no elegimos intención, forzamos menú
    if not s.get("intention"):
        s["stage"] = "asked_intent"
        return QualifyResp(reply_text=_menu())

    # Si estaba esperando área/dirección pero no entiendo: repregunto
    if s["stage"] == "waiting_area_or_address":
        return QualifyResp(reply_text=_ask_area_or_address())

    # Caso general
    return QualifyResp(reply_text="Perdón, estoy con un inconveniente técnico. ¿Podés repetir tu consulta?")

# ==========
# API
# ==========
@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/debug")
async def debug():
    # Info básica (ocultando secretos)
    env = {
        "TOKKO_API_KEY": "****" if TOKKO_API_KEY else "(missing)",
    }
    return {"env": env}

@app.post("/qualify", response_model=QualifyResp)
async def qualify(payload: QualifyReq = Body(...)):
    """
    Entrada desde n8n:
      - chatId
      - message (puede ser texto o link)
      - isFromMe (ignorado)
      - senderName
    """
    if not payload.chatId or not payload.message:
        return QualifyResp(reply_text="Perdón, llegó un mensaje vacío. ¿Podés repetir?")

    resp = await handle_message(
        chat_id=payload.chatId,
        text=payload.message,
        sender=payload.senderName or ""
    )
    return resp
