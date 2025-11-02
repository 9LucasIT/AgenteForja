import os
import json
import time
from typing import Dict, Any, Optional, List

import httpx
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# =========================
# Config
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "350"))

TOKKO_API_BASE = os.getenv("TOKKO_API_BASE", "https://api.tokkobroker.com").rstrip("/")
TOKKO_API_KEY  = os.getenv("TOKKO_API_KEY", "")
TOKKO_TIMEOUT  = float(os.getenv("TOKKO_TIMEOUT", "15"))

SITE_URL = "https://www.veglienzone.com.ar/"
VENDOR_PHONE_E164 = "5493412654593"  # no se cambia por ahora

# =========================
# App
# =========================
app = FastAPI(title="WhatsApp Lead Agent – Veglienzone")

# =========================
# Sesiones en memoria (simple)
# =========================
# Estructura por chatId:
# {
#   "phase": str,
#   "operation": "alquiler"|"venta"|None,
#   "zone": str|None,
#   "exact": {"address":..., "link":...} | None,
#   "prop_type": "departamento"|"casa"|... | None,
#   "bedrooms": int|None,
#   "budget": int|None,
#   "rent": {"incomes": bool|None, "guarantee": str|None, "occupants": int|None, "pets": bool|None},
#   "last_updated": ts
# }
SESSIONS: Dict[str, Dict[str, Any]] = {}

def session_for(chat_id: str) -> Dict[str, Any]:
    s = SESSIONS.get(chat_id)
    if not s:
        s = {
            "phase": "menu",  # menu -> ask_operation -> ask_exact_or_zone -> ...
            "operation": None,
            "zone": None,
            "exact": None,
            "prop_type": None,
            "bedrooms": None,
            "budget": None,
            "rent": {"incomes": None, "guarantee": None, "occupants": None, "pets": None},
            "last_updated": time.time(),
        }
        SESSIONS[chat_id] = s
    s["last_updated"] = time.time()
    return s

def reset_session(chat_id: str):
    if chat_id in SESSIONS:
        del SESSIONS[chat_id]

# =========================
# Utilidades NLP simples
# =========================
def norm(text: str) -> str:
    return (text or "").strip().lower()

def contains_any(text: str, terms: List[str]) -> bool:
    t = norm(text)
    return any(term in t for term in terms)

def extract_int(text: str) -> Optional[int]:
    try:
        # busca el primer entero en la frase
        for tok in text.replace("$"," ").replace("."," ").split():
            if tok.isdigit():
                return int(tok)
    except:
        return None
    return None

# =========================
# Cliente OpenAI (opcional)
# =========================
async def llm_intent(text: str) -> Dict[str, Any]:
    """
    Opcional. Si hay OPENAI_API_KEY, pedimos una ayudita para intención.
    Devuelve {"operation": "...", "has_exact": bool, "zone": "..."} cuando puede.
    """
    if not OPENAI_API_KEY:
        return {}
    prompt = f"""
Usuario dice: \"{text}\"

Tarea:
- operation: "alquiler" o "venta" si se entiende, sino vacío.
- has_exact: true si menciona dirección o un link específico, sino false.
- zone: un barrio o zona si se detecta (una palabra o dos), sino vacío.

Responde SOLO JSON válido.
"""
    try:
        import asyncio, json as _json
        from httpx import AsyncClient

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {
            "model": OPENAI_MODEL,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
            "messages": [
                {"role": "system", "content": "Eres un parser breve y exacto."},
                {"role": "user", "content": prompt}
            ]
        }
        async with AsyncClient(timeout=20) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            return _json.loads(content)
    except Exception:
        return {}

# =========================
# Cliente Tokko (en vivo)
# =========================
class TokkoClient:
    def __init__(self, base: str, api_key: str, timeout: float = 15):
        if not api_key:
            raise RuntimeError("Falta TOKKO_API_KEY en variables de entorno.")
        self.base = base
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "X-API-Key": api_key,
            "Accept": "application/json"
        }

    async def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base}{path}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url, headers=self.headers, params=params)
            if resp.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"Tokko GET {path} -> {resp.status_code} {resp.text}")
            return resp.json()

    def normalize(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": p.get("id"),
            "code": p.get("code") or p.get("reference_code"),
            "title": p.get("title") or p.get("type_name"),
            "operation": p.get("operation", {}).get("name") or p.get("operation_type"),
            "address": (p.get("address", {}) or {}).get("full") or p.get("address"),
            "neighborhood": (p.get("location", {}) or {}).get("neighborhood") or p.get("neighborhood"),
            "price": (p.get("price", {}) or {}).get("amount") or p.get("price"),
            "currency": (p.get("price", {}) or {}).get("currency") or p.get("currency"),
            "bedrooms": p.get("bedrooms"),
            "url": p.get("public_url") or p.get("permalink") or p.get("url"),
        }

    async def search(self, *, operation: Optional[str], zone: Optional[str], limit: int = 3) -> List[Dict[str, Any]]:
        # Parámetros genéricos; ajustá con tu cuenta si cambian los nombres
        params: Dict[str, Any] = {
            "available": True,
            "limit": limit,
        }
        if zone:
            params["q"] = zone
        if operation:
            params["operation"] = operation  # "alquiler"|"venta" (si tu API usa "rent"/"sale", adaptalo aquí)
        data = await self._get("/v1/properties", params)
        items = data.get("results") or data.get("data") or data.get("items") or []
        return [self.normalize(p) for p in items]

tokko = TokkoClient(TOKKO_API_BASE, TOKKO_API_KEY, TOKKO_TIMEOUT)

@app.get("/api/tokko/search")
async def api_tokko_search(operation: Optional[str] = None, zone: Optional[str] = None, limit: int = 3):
    props = await tokko.search(operation=operation, zone=zone, limit=limit)
    return {"count": len(props), "items": props}

# =========================
# Modelos de entrada/salida /qualify
# =========================
class QualifyIn(BaseModel):
    chatId: str
    text: str
    isFromMe: Optional[bool] = False

class QualifyOut(BaseModel):
    reply_text: str = ""
    closing_text: str = ""
    vendor_push: bool = False
    vendor_message: str = ""

# =========================
# Lógica conversacional
# =========================
def start_menu_text() -> str:
    return (
        "Gracias por contactarte con el área comercial de Veglienzone Gestión Inmobiliaria. ¿Cómo podemos ayudarte hoy?\n"
        "1- Alquileres\n2- Ventas\n3- Tasaciones\n\n"
        "Nota: si en cualquier momento escribís *reset*, la conversación se reinicia desde cero."
    )

async def build_tokko_list(operation: Optional[str], zone: Optional[str]) -> str:
    try:
        props = await tokko.search(operation=operation, zone=zone, limit=3)
        if not props:
            return "Por ahora no encontré opciones puntuales con ese criterio. Podés ver todo acá: " + SITE_URL
        lines = ["Te dejo algunas opciones ahora mismo:"]
        for p in props:
            price = f"{p.get('currency','')}{p.get('price','')}" if p.get("price") else "Precio consultar"
            code = p.get("code") or p.get("id")
            addr = p.get("address") or p.get("neighborhood") or ""
            url  = p.get("url") or SITE_URL
            lines.append(f"• *{code}* — {addr} — {price}\n{url}")
        return "\n\n".join(lines)
    except Exception:
        return "Puedo compartirte el catálogo completo para que explores: " + SITE_URL

def qualified_for_vendor(s: Dict[str, Any]) -> bool:
    """
    Regla simple:
    - Si es ALQUILER: incomes True y guarantee informada, y al menos zona o exacto.
    - Si es VENTA: con zona o exacto ya se deriva.
    """
    if s.get("operation") == "alquiler":
        r = s.get("rent", {})
        return bool((s.get("zone") or s.get("exact")) and r.get("incomes") is True and r.get("guarantee"))
    if s.get("operation") == "venta":
        return bool(s.get("zone") or s.get("exact"))
    return False

def vendor_summary(s: Dict[str, Any], chat_id: str) -> str:
    op = s.get("operation") or "sin operación"
    zone = s.get("zone") or "-"
    exact = s.get("exact") or {}
    addr = exact.get("address") or exact.get("link") or "-"
    r = s.get("rent", {}) or {}
    bd = s.get("bedrooms")
    budget = s.get("budget")
    parts = [
        "📌 *Lead calificado – Veglienzone*",
        f"• Chat: {chat_id}",
        f"• Operación: {op}",
        f"• Zona: {zone}",
        f"• Exacto: {addr}",
        f"• Tipo: {s.get('prop_type') or '-'}",
        f"• Dormitorios: {bd or '-'}",
        f"• Presupuesto: {budget or '-'}",
    ]
    if op == "alquiler":
        parts += [
            f"• Ingresos demostrables: {r.get('incomes')}",
            f"• Garantía: {r.get('guarantee') or '-'}",
            f"• Habitantes: {r.get('occupants') or '-'}",
            f"• Mascotas: {r.get('pets')}",
        ]
    parts.append(f"• Vendedor destino: {VENDOR_PHONE_E164}")
    return "\n".join(parts)

# =========================
# /qualify (contrato con n8n)
# =========================
@app.post("/qualify", response_model=QualifyOut)
async def qualify(body: QualifyIn):
    chat_id = body.chatId
    text = body.text or ""
    user = norm(text)

    # reset
    if "reset" in user:
        reset_session(chat_id)
        return QualifyOut(reply_text=start_menu_text())

    s = session_for(chat_id)

    # atajos por palabras
    if contains_any(user, ["alquiler", "alquilo", "busco alquilar", "quiero alquilar"]):
        s["operation"] = "alquiler"
        s["phase"] = "ask_exact_or_zone"
    elif contains_any(user, ["venta", "compro", "quiero comprar", "comprar"]):
        s["operation"] = "venta"
        s["phase"] = "ask_exact_or_zone"

    # empujón LLM (opcional)
    if not s.get("operation") and OPENAI_API_KEY:
        info = await llm_intent(text)
        if info.get("operation") in ("alquiler","venta"):
            s["operation"] = info["operation"]
            s["phase"] = "ask_exact_or_zone"
        if info.get("zone"):
            s["zone"] = info["zone"]

    # Fases
    if s["phase"] == "menu":
        # Primer contacto
        if user in ("1","2","3"):
            if user == "1":
                s["operation"] = "alquiler"
            elif user == "2":
                s["operation"] = "venta"
            else:
                return QualifyOut(reply_text="Para tasaciones, te deriva un asesor. ¿Querés que te contacten por este WhatsApp?",
                                  closing_text="")
            s["phase"] = "ask_exact_or_zone"
        else:
            return QualifyOut(reply_text=start_menu_text())

    if s["phase"] == "ask_exact_or_zone":
        # Preguntar si tiene una dirección/link exacto o busca por zona
        if contains_any(user, ["http://","https://","zona prop","zonaprop","link "]):
            s["exact"] = {"link": text.strip()}
            s["phase"] = "collect_more"
            return QualifyOut(reply_text="Perfecto, pasame cualquier consulta puntual de esa ficha o lo que quieras confirmar. 😉")

        # ¿trajo dirección?
        if contains_any(user, ["calle","altura","número","numero","direccion","dirección"]):
            s["exact"] = {"address": text.strip()}
            s["phase"] = "collect_more"
            return QualifyOut(reply_text="¡Genial! Te ayudo con esa dirección. ¿Qué querés saber específicamente?")

        # ¿dijo zona/barrio directamente?
        if contains_any(user, ["zona","barrio","centro","macrocentro","abasto","echesortu","alberdi"]) or s.get("zone"):
            if not s.get("zone"):
                # extrae una palabra de barrio simple si podemos
                tokens = [t for t in user.split() if t.isalpha()]
                s["zone"] = tokens[-1] if tokens else "Centro"
            # Acción solicitada: enviar link + ofrecer lista
            s["phase"] = "offer_tokko"
            msg = (f"Perfecto. Te dejo el link donde están todas nuestras propiedades para ver si alguna te interesa:\n"
                   f"{SITE_URL}\n\n¿Querés que además te envíe *3 opciones* ahora mismo en *{s['zone']}*?")
            return QualifyOut(reply_text=msg)

        # Si todavía no respondió, preguntamos explícito
        return QualifyOut(reply_text=(
            "¿Tenés *dirección o link* exacto de la propiedad, o estás *averiguando por una zona/barrio*?"
        ))

    if s["phase"] == "offer_tokko":
        if contains_any(user, ["si","sí","dale","enviame","mandame","ok"]):
            listado = await build_tokko_list(s.get("operation"), s.get("zone"))
            s["phase"] = "collect_more"
            return QualifyOut(reply_text=listado)
        elif contains_any(user, ["no","después","despues","gracias"]):
            s["phase"] = "collect_more"
            return QualifyOut(reply_text="¡De una! Si ves algo que te guste en el sitio, escribime y lo vemos. ¿Te quedó alguna duda?")
        else:
            return QualifyOut(reply_text="¿Querés que te envíe 3 opciones ahora en esa zona? (Sí/No)")

    if s["phase"] == "collect_more":
        # Preguntas de calificación según operación
        if s.get("operation") == "alquiler":
            r = s["rent"]
            # Ingresos demostrables
            if r["incomes"] is None and contains_any(user, ["ingreso","sueldo","recibo","demostrable","monotributo","recibos"]):
                r["incomes"] = True if contains_any(user, ["tengo","sí","si","cuento"]) else False

            # Garantía
            if r["guarantee"] is None and contains_any(user, ["garant","finaer","propiet","caución","caucion","seguro"]):
                # guardamos texto crudo
                r["guarantee"] = text.strip()

            # Ocupantes
            if r["occupants"] is None and contains_any(user, ["somos","personas","habitamos","vivimos"]):
                val = extract_int(text)
                if val:
                    r["occupants"] = val

            # Mascotas
            if r["pets"] is None and contains_any(user, ["mascota","perro","gato"]):
                r["pets"] = True if contains_any(user, ["sí","si","tengo"]) else False

            # Budget
            if s["budget"] is None and contains_any(user, ["presupuesto","hasta","pago","monto","precio"]):
                val = extract_int(text)
                if val:
                    s["budget"] = val

            # Dormitorios
            if s["bedrooms"] is None and contains_any(user, ["dorm","habitación","habitacion","ambiente","ambientes"]):
                val = extract_int(text)
                if val:
                    s["bedrooms"] = val

            # ¿ya califica?
            if qualified_for_vendor(s):
                msg = vendor_summary(s, chat_id)
                return QualifyOut(
                    reply_text="Con los datos que me pasaste, ya puedo derivarte a un asesor humano para coordinar y ver opciones. ¿Querés que te contacten por este WhatsApp?",
                    vendor_push=True,
                    vendor_message=msg
                )

            # si no, seguimos preguntando
            return QualifyOut(reply_text=(
                "Para avanzar con *alquiler*, necesito confirmar: *ingresos demostrables* (que tripliquen aprox. el alquiler), "
                "*tipo de garantía* (Finaer, caución, propietaria, etc.), *cantidad de habitantes* y si tienen *mascotas*. "
                "Además, si tenés un *presupuesto* y *dormitorios* deseados, genial."
            ))

        elif s.get("operation") == "venta":
            # Para venta, con zona o exacto ya alcanza para derivar.
            if qualified_for_vendor(s):
                msg = vendor_summary(s, chat_id)
                return QualifyOut(
                    reply_text="Perfecto. Te derivo a un asesor humano para que coordinen y te compartan opciones de venta. ¿Querés que te contacten por este WhatsApp?",
                    vendor_push=True,
                    vendor_message=msg
                )
            # caso base, pedimos lo mínimo
            if not s.get("zone") and not s.get("exact"):
                if contains_any(user, ["zona","barrio"]):
                    s["zone"] = text.strip()
                elif contains_any(user, ["calle","direccion","dirección"]):
                    s["exact"] = {"address": text.strip()}
                else:
                    return QualifyOut(reply_text="¿En qué *zona o barrio* estás buscando comprar? (Si tenés dirección exacta, mejor)")
            return QualifyOut(reply_text="¿Tenés alguna preferencia de *tipo de propiedad* o *presupuesto*?")

    # fallback
    return QualifyOut(reply_text="Perdón, estoy con un inconveniente técnico. ¿Podés repetir tu consulta?")

# =========================
# Endpoints de salud y debug
# =========================
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug")
def debug():
    return {
        "llm_mode": "openai" if OPENAI_API_KEY else "rules",
        "OPENAI_MODEL": OPENAI_MODEL,
        "GREEN_ID": os.getenv("GREEN_ID", "****"),
        "N8N_VENDOR_WEBHOOK": os.getenv("N8N_VENDOR_WEBHOOK", "****"),
        "TOKKO_BASE": TOKKO_API_BASE,
        "TOKKO_KEY_set": bool(TOKKO_API_KEY),
    }

# =========================
# NO-OP para Green si está apuntando acá (no reenvía a n8n)
# =========================
@app.post("/api/green/inbound")
async def green_inbound_noop(request: Request):
    try:
        payload = await request.json()
        print("GREEN HOOK HIT ->", json.dumps(payload)[:500])
    except Exception:
        pass
    # No hacemos NADA para no duplicar flujos ni spamear al vendedor
    return {"ok": True}
