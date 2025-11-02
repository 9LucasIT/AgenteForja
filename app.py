# app.py
import os
import re
import time
import json
import logging
from typing import Optional, Dict, Any, List

import requests
from fastapi import FastAPI, Body
from pydantic import BaseModel

# ----------------------------------
# Config
# ----------------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
VENDOR_PHONE = os.getenv("VENDOR_PHONE", "5493412654593")
TOKKO_API_KEY = os.getenv("TOKKO_API_KEY", "")  # <-- ya lo tenés en Railway
SITE_LINK = "https://www.veglienzone.com.ar/"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("veglienzone-agent")

app = FastAPI(title="Veglienzone AI Agent")

# ----------------------------------
# Memoria de sesión simple en memoria
# ----------------------------------
# Nota: es memoria efímera (se reinicia al redeploy). Para producción real,
# conviene Redis o DB. Suficiente para test y costos bajos.
SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_SECONDS = 60 * 60 * 8  # 8h de inactividad


def now() -> float:
    return time.time()


def session(chat_id: str) -> Dict[str, Any]:
    s = SESSIONS.get(chat_id)
    if not s or now() - s.get("_ts", 0) > SESSION_TTL_SECONDS:
        s = {
            "_ts": now(),
            "state": "welcome",
            "op": None,  # 'alquiler' | 'venta'
            "address": None,
            "tokko_hits": [],
            "exact": None,  # ficha exacta
            "qual": {
                "ingresos": None,
                "garantia": None,
                "habitantes": None,
                "mascotas": None,
            },
        }
        SESSIONS[chat_id] = s
    else:
        s["_ts"] = now()
    return s


def reset_session(chat_id: str):
    if chat_id in SESSIONS:
        del SESSIONS[chat_id]
    session(chat_id)


# ----------------------------------
# Utilidades de texto
# ----------------------------------
def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def looks_like_url(text: str) -> bool:
    return bool(re.search(r"https?://", text or "", re.I))


def looks_like_address(text: str) -> bool:
    # Heurística simple: palabra + número (ej. "Av. Cabildo 2853")
    return bool(re.search(r"[a-zA-ZÁÉÍÓÚÑáéíóúñ]+\s+\d{1,6}", text or ""))


def said_no_address(text: str) -> bool:
    t = norm(text)
    return any(
        kw in t
        for kw in [
            "no tengo dirección",
            "no tengo direccion",
            "no tengo link",
            "sin dirección",
            "sin direccion",
            "sin link",
            "busco por zona",
            "busco por barrio",
            "no tengo ni link ni dirección",
            "no tengo ni link ni direccion",
            "no tengo nada",
        ]
    )


def is_yes(text: str) -> bool:
    return norm(text) in {"si", "sí", "ok", "dale", "de una", "si, por favor", "sí, por favor", "quiero"}


def is_no(text: str) -> bool:
    return norm(text) in {"no", "nop", "no gracias", "gracias, no", "por ahora no"}


def extract_op(text: str) -> Optional[str]:
    t = norm(text)
    if re.search(r"\b(1|alquiler|alquilo|alquilar|renta|rent)\b", t):
        return "alquiler"
    if re.search(r"\b(2|venta|compro|comprar|vende|vendo)\b", t):
        return "venta"
    return None


# ----------------------------------
# Tokko API
# ----------------------------------
TOKKO_BASE = "https://api.tokkobroker.com/api/v1/property/"

def tokko_headers():
    return {"Accept": "application/json"}

def tokko_query(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Llama a Tokko /property con el token. Maneja errores de red.
    """
    if not TOKKO_API_KEY:
        raise RuntimeError("Falta TOKKO_API_KEY en variables de entorno.")

    q = {"auth_token": TOKKO_API_KEY, "format": "json"}
    q.update(params or {})
    try:
        r = requests.get(TOKKO_BASE, params=q, headers=tokko_headers(), timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.exception("Tokko error: %s", e)
        return {"objects": []}

def normalize_addr(s: str) -> str:
    s = s or ""
    s = s.replace(".", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def tokko_search_by_address(address: str) -> List[Dict[str, Any]]:
    """
    Devuelve lista de propiedades candidatas. Usa búsqueda amplia por dirección.
    """
    # Estrategia: 1) intentar por "search" (full-text), 2) fallback por "address"
    data = tokko_query({"search": address, "limit": 10})
    objs = data.get("objects") or []

    # orden simple: coincidencias que contengan el número exacto primero
    num = re.findall(r"\d{1,6}", address or "")
    hits = []
    for it in objs:
        addr = (it.get("address") or "") + " " + (it.get("location", {}).get("name") or "")
        score = 0
        if address.lower() in addr.lower():
            score += 3
        if num and any(n in addr for n in num):
            score += 2
        if it.get("code"):
            score += 1
        hits.append((score, it))
    hits.sort(key=lambda x: x[0], reverse=True)
    return [h[1] for h in hits]

def tokko_exact_match(address: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    target = normalize_addr(address)
    for it in candidates:
        addr = normalize_addr(it.get("address") or "")
        if addr == target:
            return it
    # también aceptar coincidencia fuerte “calle + número”
    calle_num = re.findall(r"[a-zA-ZÁÉÍÓÚÑáéíóúñ]+\s+\d{1,6}", address)
    if calle_num:
        pat = normalize_addr(calle_num[0])
        for it in candidates:
            if pat in normalize_addr(it.get("address") or ""):
                return it
    return None

def fmt_price(obj: Dict[str, Any]) -> str:
    currency = (obj.get("price_currency") or obj.get("currency") or "USD").upper()
    price = obj.get("price") or obj.get("price_total") or obj.get("price_list") or 0
    try:
        price = int(float(price))
        price_txt = f"{price:,}".replace(",", ".")
    except Exception:
        price_txt = str(price)
    return f"{currency} {price_txt}"

def property_summary(obj: Dict[str, Any]) -> str:
    code = obj.get("code", "")
    addr = obj.get("address", "Sin dirección")
    loc = obj.get("location", {}).get("name") or ""
    op = (obj.get("operation", {}) or {}).get("name") or obj.get("operation_type") or "Operación"
    price = fmt_price(obj)
    dorm = obj.get("bedrooms", 0)
    baths = obj.get("bathrooms", 0)
    covered = obj.get("covered_surface") or obj.get("surface_covered") or 0
    total = obj.get("surface_total") or obj.get("surface") or 0
    amb = obj.get("ambiences") or obj.get("rooms") or 0
    url = obj.get("web_url") or obj.get("url") or SITE_LINK

    lines = [
        f"• Código: {code}",
        f"• Dirección: {addr}" + (f" ({loc})" if loc else ""),
        f"• Operación: {op}",
        f"• Precio: {price}",
        f"• Dormitorios: {dorm} | Baños: {baths}",
        f"• Sup. cubierta: {covered} m² | Sup. total: {total} m²",
        f"• Ambientes: {amb}",
        f"• Ficha: {url}",
    ]
    return "\n".join(lines)

def list_summary(objs: List[Dict[str, Any]]) -> str:
    out = []
    for i, o in enumerate(objs[:3], 1):
        out.append(f"{i}) {o.get('address','Sin dirección')} – {fmt_price(o)} – {(o.get('operation',{}) or {}).get('name','')}")
    if not out:
        return "No encontré coincidencias para esa dirección."
    return "\n".join(out)


# ----------------------------------
# Esquema de entrada/salida /qualify
# ----------------------------------
class QualifyIn(BaseModel):
    chatId: str
    message: str
    isFromMe: Optional[bool] = False
    senderName: Optional[str] = ""


class QualifyOut(BaseModel):
    reply_text: str = ""
    closing_text: str = ""
    vendor_push: bool = False
    vendor_message: str = ""


# ----------------------------------
# Lógica conversacional
# ----------------------------------
WELCOME = (
    "Gracias por contactarte con el área comercial de Veglienzone Gestión Inmobiliaria. "
    "¿Cómo podemos ayudarte hoy?\n"
    "1- Alquileres\n2- Ventas\n3- Tasaciones\n\n"
    "Nota: si en cualquier momento escribís *reset*, la conversación se reinicia desde cero."
)

ASK_ADDRESS = "¿Tenés dirección o link exacto de la propiedad, o estás averiguando por una zona/barrio?"

def make_vendor_message(chat_id: str, sess: Dict[str, Any]) -> str:
    lines = [
        "✅ *Lead calificado para derivación*",
        f"• Cliente: {chat_id}",
    ]
    if sess.get("op"):
        lines.append(f"• Operación: {sess['op'].title()}")
    if sess.get("exact"):
        lines.append(f"• Propiedad: {sess['exact'].get('address','')}")
        if sess['exact'].get('code'):
            lines.append(f"• Código: {sess['exact']['code']}")
        if sess['exact'].get('web_url'):
            lines.append(f"• Link: {sess['exact']['web_url']}")
    q = sess.get("qual", {})
    if sess.get("op") == "alquiler":
        lines.append(f"• Ingresos demostrables: {q.get('ingresos')}")
        lines.append(f"• Garantía: {q.get('garantia')}")
        lines.append(f"• Habitantes: {q.get('habitantes')}")
        lines.append(f"• Mascotas: {q.get('mascotas')}")
    return "\n".join(lines)


def handle_message(chat_id: str, msg: str) -> QualifyOut:
    out = QualifyOut()
    t = norm(msg)

    # reset
    if t in {"reset", "/reset", "reiniciar"}:
        reset_session(chat_id)
        out.reply_text = WELCOME
        return out

    sess = session(chat_id)

    # Estado inicial
    if sess["state"] == "welcome":
        # detectar si ya aclaró la operación
        op = extract_op(msg)
        if op:
            sess["op"] = op
            sess["state"] = "ask_address"
            out.reply_text = ASK_ADDRESS
            return out
        # aún no
        out.reply_text = WELCOME
        return out

    # Selección de operación
    if sess["state"] in {"ask_op", "ask_address"}:
        # si todavía no definió operación
        if not sess["op"]:
            op = extract_op(msg)
            if not op:
                out.reply_text = WELCOME
                sess["state"] = "welcome"
                return out
            sess["op"] = op
            sess["state"] = "ask_address"
            out.reply_text = ASK_ADDRESS
            return out

        # ya tenemos op, ahora procesar dirección o zona
        if said_no_address(msg):
            # Sólo link y cierro cordialmente
            out.reply_text = (
                "Perfecto, te dejo el link donde están todas nuestras propiedades para que puedas ver si alguna te interesa:\n"
                f"{SITE_LINK}\n\n"
                "Cualquier consulta puntual, escribime por acá 🙌"
            )
            out.closing_text = ""
            sess["state"] = "welcome"  # volvemos al inicio para una próxima consulta
            return out

        if looks_like_url(msg) or looks_like_address(msg):
            # Buscar en Tokko
            candidates = tokko_search_by_address(msg)
            sess["tokko_hits"] = candidates
            exact = tokko_exact_match(msg, candidates)
            if exact:
                sess["exact"] = exact
                # Mostrar SOLO esa ficha
                out.reply_text = property_summary(exact)
                # Iniciar calificación si es alquiler
                if sess["op"] == "alquiler":
                    sess["state"] = "q_ingresos"
                    out.closing_text = "Para avanzar con alquiler, ¿contás con *ingresos demostrables*?"
                else:
                    # Venta: no pedimos requisitos duros, vamos directo a consulta/derivación
                    sess["state"] = "ask_deriv"
                    out.closing_text = "¿Querés que te contacte un asesor humano por este WhatsApp para avanzar?"
                return out
            else:
                # Listar hasta 3 alternativas
                if not candidates:
                    out.reply_text = (
                        "No encontré coincidencias exactas para esa dirección. "
                        "Si querés, mirá nuestro catálogo completo:\n"
                        f"{SITE_LINK}\n\n"
                        "¿Querés decirme otra dirección o barrio?"
                    )
                    sess["state"] = "ask_address"
                    return out
                out.reply_text = "Estas son las coincidencias más cercanas:\n" + list_summary(candidates)
                out.closing_text = (
                    "Si alguna te interesa, decime cuál (1, 2 o 3), o compartime el link/código exacto."
                )
                sess["state"] = "pick_from_list"
                return out

        # Si llegó texto genérico (por zona/barrio pero sin “no tengo dirección”)
        if any(w in t for w in ["zona", "barrio", "centro", "norte", "sur", "oeste", "este"]):
            out.reply_text = (
                "Perfecto, te dejo el link donde están todas nuestras propiedades para que puedas ver si alguna te interesa:\n"
                f"{SITE_LINK}\n\n"
                "Cualquier consulta puntual, escribime por acá 🙌"
            )
            sess["state"] = "welcome"
            return out

        # No entendí → repreguntar
        out.reply_text = ASK_ADDRESS
        return out

    # Selección desde listado (1-3)
    if sess["state"] == "pick_from_list":
        m = re.search(r"\b([1-3])\b", t)
        if not m:
            out.reply_text = "Decime 1, 2 o 3 para ver la ficha completa; o pasame el link/código exacto."
            return out
        idx = int(m.group(1)) - 1
        hits = sess.get("tokko_hits") or []
        if 0 <= idx < len(hits):
            exact = hits[idx]
            sess["exact"] = exact
            out.reply_text = property_summary(exact)
            if sess["op"] == "alquiler":
                sess["state"] = "q_ingresos"
                out.closing_text = "Para avanzar con alquiler, ¿contás con *ingresos demostrables*?"
            else:
                sess["state"] = "ask_deriv"
                out.closing_text = "¿Querés que te contacte un asesor humano por este WhatsApp para avanzar?"
            return out
        out.reply_text = "Número inválido. Elegí 1, 2 o 3; o pasame el link/código exacto."
        return out

    # Calificación para ALQUILER
    if sess["state"] == "q_ingresos":
        sess["qual"]["ingresos"] = "sí" if is_yes(msg) else ("no" if is_no(msg) else msg)
        sess["state"] = "q_garantia"
        out.reply_text = "¿Qué *tipo de garantía* podrías presentar? (Finaer/Seguro de caución, propietaria CABA, u otra)"
        return out

    if sess["state"] == "q_garantia":
        sess["qual"]["garantia"] = msg.strip()
        sess["state"] = "q_habitantes"
        out.reply_text = "¿Cuántas *personas* vivirían en la unidad?"
        return out

    if sess["state"] == "q_habitantes":
        sess["qual"]["habitantes"] = msg.strip()
        sess["state"] = "q_mascotas"
        out.reply_text = "¿Tienen *mascotas*? (sí/no y cuáles)"
        return out

    if sess["state"] == "q_mascotas":
        sess["qual"]["mascotas"] = msg.strip()
        sess["state"] = "ask_deriv"
        out.reply_text = "¡Gracias! Con esos datos puedo ayudarte mejor."
        out.closing_text = "¿Querés que te contacte un asesor humano por este WhatsApp para coordinar?"
        return out

    # Pregunta final de derivación
    if sess["state"] == "ask_deriv":
        if is_yes(msg):
            out.vendor_push = True
            out.vendor_message = make_vendor_message(chat_id, sess)
            out.reply_text = "¡Genial! Ya aviso a un asesor para que te contacte por este WhatsApp 👇"
            out.closing_text = ""
            # luego de derivar, reseteamos al estado inicial
            sess["state"] = "welcome"
            return out
        if is_no(msg):
            out.reply_text = "Perfecto, quedo atento a tus consultas sobre la propiedad. ¡Gracias por escribirnos! 🙂"
            sess["state"] = "welcome"
            return out
        out.reply_text = "¿Te derivo con un asesor por este WhatsApp? (sí/no)"
        return out

    # Fallback
    out.reply_text = "Perdón, estoy con un inconveniente técnico. ¿Podés repetir tu consulta?"
    return out


# ----------------------------------
# Endpoints
# ----------------------------------
@app.get("/health")
def health():
    return {"ok": True, "llm_mode": "openai", "OPENAI_MODEL": OPENAI_MODEL}


@app.post("/qualify", response_model=QualifyOut)
def qualify(payload: QualifyIn = Body(...)):
    """
    Punto que usa tu nodo HTTP "Qualify" en n8n.
    """
    log.info("QUALIFY IN <- %s", payload.dict())
    out = handle_message(payload.chatId, payload.message)
    log.info("QUALIFY OUT -> %s", out.dict())
    return out


# (Opcional) endpoint legacy por si tenés hooks antiguos apuntando acá
@app.post("/api/inbound", response_model=QualifyOut)
def qualify_legacy(payload: QualifyIn = Body(...)):
    return qualify(payload)
