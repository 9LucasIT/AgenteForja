# app.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import re
import httpx
import uvicorn

app = FastAPI(title="FastAPI Agent – Veglienzone", version="1.0.0")

# =========================
# Config & constantes
# =========================
CATALOG_LINK = "https://www.veglienzone.com.ar/"
TOKKO_API_KEY = os.getenv("TOKKO_API_KEY", "").strip()
VENDOR_PHONE = os.getenv("VENDOR_PHONE", "").strip()  # por si en el futuro reactivás derivación

MSG_MENU = (
    "Gracias por contactarte con el área comercial de Veglienzone Gestión Inmobiliaria. "
    "¿Cómo podemos ayudarte hoy?\n"
    "1- Alquileres\n"
    "2- Ventas\n"
    "3- Tasaciones\n\n"
    "Nota: si en cualquier momento escribís *reset*, la conversación se reinicia desde cero."
)

MSG_CATALOG = (
    "Perfecto. Te dejo el link donde están todas nuestras propiedades para que veas si alguna te interesa:\n"
    f"{CATALOG_LINK}\n\n"
    "Cualquier consulta me escribís por acá y te ayudo 🙂"
)

MSG_ASK_ADDR_OR_ZONE = (
    "¿Tenés dirección o link exacto de la propiedad, o estás averiguando por una zona/barrio?"
)

MSG_TECH_ISSUE = "Perdón, estoy con un inconveniente técnico. ¿Podés repetir tu consulta?"

# =========================
# Heurísticas (dirección & links)
# =========================
ADDRESS_RE = re.compile(
    r'\b('
    r'Av\.?|Avenida|Calle|Pje\.?|Pasaje|Bvard\.?|Bulevar|Ruta|Camino|Diag\.?|Diagonal|'
    r'[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑa-záéíóúñ\.]+){0,3}'
    r')\s+(\d{1,6})\b'
)

LINK_RE = re.compile(r'(https?://\S+)', re.IGNORECASE)


def looks_like_address(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if LINK_RE.search(t):
        return False
    return bool(ADDRESS_RE.search(t))


def extract_address(text: str) -> Optional[str]:
    m = ADDRESS_RE.search(text or "")
    if not m:
        return None
    return re.sub(r'\s+', ' ', m.group(0)).strip()


def extract_link(text: str) -> Optional[str]:
    m = LINK_RE.search(text or "")
    return m.group(1) if m else None


# =========================
# Modelos de entrada/salida
# =========================
class QualifyIn(BaseModel):
    chatId: Optional[str] = None
    message: Optional[str] = None
    text: Optional[str] = None
    isFromMe: Optional[bool] = None
    senderName: Optional[str] = None

    @property
    def user_text(self) -> str:
        return (self.message or self.text or "").strip()


class BotOut(BaseModel):
    reply_text: str
    closing_text: str = ""
    vendor_push: bool = False
    vendor_message: str = ""


# =========================
# Tokko API (simple client)
# =========================
TOKKO_BASE = "https://www.tokkobroker.com/api/v1"


async def tokko_search_by_address(address: str) -> Optional[Dict[str, Any]]:
    """
    Búsqueda *heurística* por dirección en Tokko.
    NOTA: Tokko tiene varios filtros/recursos; esta función usa un endpoint genérico 'property'
    con parámetro 'search' para resolver direcciones o términos libres.
    Si tu cuenta requiere otros filtros (agency_id, offices, etc.), agregalos aquí.
    """
    if not TOKKO_API_KEY:
        return None

    params = {
        "format": "json",
        "key": TOKKO_API_KEY,
        "search": address,
        "limit": 5,
        "offset": 0,
    }

    url = f"{TOKKO_BASE}/property/"
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            # Estructura esperada: {"objects": [...], "meta": {...}}
            objects = data.get("objects") if isinstance(data, dict) else None
            if not objects:
                return None

            # Elegimos el primer match (o podés rankear por cercanía a la cadena)
            prop = objects[0]
            return {
                "title": prop.get("title") or prop.get("operation", ""),
                "price": prop.get("price"),
                "currency": prop.get("currency", ""),
                "address": prop.get("address") or "",
                "bedrooms": prop.get("bedrooms"),
                "bathrooms": prop.get("bathrooms"),
                "covered_surface": prop.get("covered_surface"),
                "total_surface": prop.get("surface"),
                "has_parking": prop.get("garage"),
                "allows_pets": prop.get("pets_allowed"),
                "url": prop.get("web_url") or prop.get("permalink") or "",
                "raw": prop,
            }
    except Exception:
        return None


def render_property_card(p: Dict[str, Any]) -> str:
    lines = []
    if p.get("title"):
        lines.append(f"*{p['title']}*")
    if p.get("address"):
        lines.append(p["address"])

    # Precio
    price = p.get("price")
    currency = p.get("currency", "")
    if price:
        try:
            price_str = f"{price:,.0f}".replace(",", ".")
        except Exception:
            price_str = str(price)
        lines.append(f"Precio: {currency} {price_str}")

    # Ambientes básicos
    if p.get("bedrooms") is not None:
        lines.append(f"Dormitorios: {p['bedrooms']}")
    if p.get("bathrooms") is not None:
        lines.append(f"Baños: {p['bathrooms']}")
    if p.get("covered_surface") or p.get("total_surface"):
        c = p.get("covered_surface")
        t = p.get("total_surface")
        if c and t:
            lines.append(f"Superficie: {c} m² cubiertos / {t} m² totales")
        elif c:
            lines.append(f"Superficie cubierta: {c} m²")
        elif t:
            lines.append(f"Superficie total: {t} m²")

    # Extras
    if p.get("has_parking") is not None:
        lines.append("Cochera: Sí" if p["has_parking"] else "Cochera: No")
    if p.get("allows_pets") is not None:
        lines.append("Mascotas: Sí" if p["allows_pets"] else "Mascotas: No")

    if p.get("url"):
        lines.append(f"\nFicha completa:\n{p['url']}")

    return "\n".join(lines)


# =========================
# Memoria simple por chat
# =========================
# Si más adelante querés Redis/DB, acá es el punto.
MEMORY: Dict[str, Dict[str, Any]] = {}


def get_state(chat_id: str) -> Dict[str, Any]:
    st = MEMORY.get(chat_id) or {}
    MEMORY[chat_id] = st
    return st


def reset_state(chat_id: str):
    MEMORY.pop(chat_id, None)


# =========================
# Endpoints
# =========================
@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/debug")
async def debug():
    safe_env = {
        "TOKKO_API_KEY": "set" if TOKKO_API_KEY else "missing",
        "VENDOR_PHONE": VENDOR_PHONE or "",
    }
    return {"env": safe_env, "memory_sessions": len(MEMORY)}


@app.post("/api/green/inbound")
async def green_inbound(request: Request):
    """
    Endpoint NOOP para que, si todavía tenés Green apuntando acá,
    nunca rompa. Devuelve 200 OK siempre.
    """
    try:
        _ = await request.json()
    except Exception:
        pass
    return JSONResponse({"status": "ok"}, status_code=200)


@app.post("/qualify", response_model=BotOut)
async def qualify(payload: QualifyIn):
    chat_id = payload.chatId or "unknown"
    text = payload.user_text

    # Normalizaciones
    low = text.lower().strip()

    # RESET
    if low in {"reset", "/reset"}:
        reset_state(chat_id)
        st = get_state(chat_id)
        st["phase"] = "menu"
        return BotOut(reply_text=MSG_MENU)

    # Estado actual
    st = get_state(chat_id)
    phase = st.get("phase")

    # Primera interacción (o sin estado): mostrar menú
    if not phase:
        st["phase"] = "menu"
        return BotOut(reply_text=MSG_MENU)

    # Si estamos en menú: detectar intención alta (alquiler/venta/tasación) y pasar a pedir dirección/zone
    if phase == "menu":
        if any(k in low for k in ["1", "alquiler", "alquilo", "alquilar", "alquileres"]):
            st["operation"] = "alquiler"
            st["phase"] = "ask_address_or_zone"
            return BotOut(reply_text=MSG_ASK_ADDR_OR_ZONE)

        if any(k in low for k in ["2", "venta", "compro", "ventas", "vender"]):
            st["operation"] = "venta"
            st["phase"] = "ask_address_or_zone"
            return BotOut(reply_text=MSG_ASK_ADDR_OR_ZONE)

        if any(k in low for k in ["3", "tasacion", "tasaciones", "tasar"]):
            st["operation"] = "tasacion"
            st["phase"] = "ask_address_or_zone"
            return BotOut(reply_text=MSG_ASK_ADDR_OR_ZONE)

        # no entendido → re-muestro menú
        return BotOut(reply_text=MSG_MENU)

    # Pedimos dirección o zona
    if phase == "ask_address_or_zone":
        lnk = extract_link(text)
        if lnk:
            # Con link: confirmo y quedo listo para responder dudas
            st["last_link"] = lnk
            st["phase"] = "got_reference"  # siguiente paso: preguntas sobre la ficha
            reply = (
                "Recibí el link 👍. Dame un momento mientras reviso la ficha y te respondo.\n\n"
                "Si querés, contame qué dato te interesa (precio, expensas, cochera, mascotas, etc.)."
            )
            return BotOut(reply_text=reply)

        addr = extract_address(text)
        if addr:
            # Intento resolver con Tokko
            p = await tokko_search_by_address(addr)
            if p:
                st["phase"] = "got_reference"
                st["last_property"] = p
                card = render_property_card(p)
                follow = (
                    "\n\n¿Querés que te contacte un asesor humano por este WhatsApp para avanzar?"
                )
                return BotOut(reply_text=f"{card}{follow}")

            # no se encontró → mensaje amable
            st["phase"] = "ask_address_or_zone"
            reply = (
                f"Perfecto, tomé la dirección *{addr}*.\n"
                "No encontré la ficha exacta ahora mismo. Si la tenés, enviame el *link* o el *código* de la propiedad para agilizar. "
                "Si preferís, también puedo pasarte el catálogo general:\n"
                f"{CATALOG_LINK}"
            )
            return BotOut(reply_text=reply)

        # si escribe “zona centro”, “barrio…”, etc. -> catálogo
        if any(k in low for k in ["zona", "barrio", "centro", "macrocentro", "microcentro", "norte", "sur", "oeste", "este"]):
            st["phase"] = "catalog_shared"
            return BotOut(reply_text=MSG_CATALOG)

        # nada reconocible → pregunto de nuevo
        return BotOut(reply_text=MSG_ASK_ADDR_OR_ZONE)

    # Si ya tengo referencia (got_reference): escucho dudas o habilito derivación humana
    if phase == "got_reference":
        if any(k in low for k in ["si", "sí", "quiero", "derivar", "asesor", "contacto", "coordinar", "visita"]):
            # Podés activar vendor_push si ya tenés armado el branch en n8n
            msg = (
                "¡Genial! Aviso a un asesor para que te contacte por este WhatsApp y coordinan los detalles."
            )
            return BotOut(reply_text=msg, vendor_push=True, vendor_message="Lead calificado desde WhatsApp.")

        # sino, respondo estilo “sigo atento”
        return BotOut(reply_text="Perfecto, quedo atento a tus consultas sobre la propiedad 😊")

    # fallback
    return BotOut(reply_text=MSG_TECH_ISSUE)


# =========================
# Run local (opcional)
# =========================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
