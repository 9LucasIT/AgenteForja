# app.py
import os
import time
import json
import re
from typing import Optional, Dict, Any, List

import httpx
from fastapi import FastAPI, Body
from pydantic import BaseModel
from dotenv import load_dotenv

# -----------------------------
# Carga de variables
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GREEN_ID       = os.getenv("GREEN_ID", "").strip()                 # ej: 54934XXXXXXXX@c.us
GREEN_TOKEN    = os.getenv("GREEN_TOKEN", "").strip()
VENDOR_PHONE   = os.getenv("VENDOR_PHONE", "5493412654593").strip()  # sin @c.us
VENDOR_CHATID  = f"{VENDOR_PHONE}@c.us"

DATABASE_URL   = os.getenv("DATABASE_URL", "")  # tu conexión (la misma que ya venías usando)
# Si tenías helpers de DB en otro archivo, podés seguir importándolos. Aquí dejo un stub.

# Endpoint GreenAPI (sendMessage)
GREEN_SEND_URL = f"https://api.green-api.com/waInstance{GREEN_ID}/sendMessage/{GREEN_TOKEN}"
# Nota: si tu GREEN_ID es sólo el número de instancia (ej "7107365363")
# poné: GREEN_SEND_URL = f"https://api.green-api.com/waInstance{GREEN_ID}/sendMessage/{GREEN_TOKEN}"

# -----------------------------
# Cliente HTTP y OpenAI
# -----------------------------
http_client = httpx.AsyncClient(timeout=20)

# OpenAI SDK (simple)
import openai as openai_lib
openai_lib.api_key = OPENAI_API_KEY

# -----------------------------
# Memorias simples en RAM
# -----------------------------
sessions: Dict[str, Dict[str, Any]] = {}  # por chatId
dedup: Dict[str, Dict[str, Any]] = {}     # anti-duplicado: {chatId: {"ts": int, "last": str}}

# -----------------------------
# Utils DB (stub: adaptá a tus helpers reales)
# -----------------------------
def get_listings_for_rent(zone: str) -> List[Dict[str, Any]]:
    """
    Retorna fichas para ALQUILER en 'zone'.
    Reemplazá por tu query real (p.ej. a MySQL usando SQLAlchemy o pymysql).
    Debe devolver al menos: codigo, direccion, precio, dormitorios, cochera (True/False), mascotas ("si"/"no"/"N/D")
    """
    # ⚠️ Ejemplo fijo (stub). Usá tu tabla real.
    data = [
        {"codigo":"A101","direccion":"San Luis 234 (Centro)", "precio":"$120.000","dormitorios":2,"cochera":"sí","mascotas":"N/D"},
        {"codigo":"A205","direccion":"Rioja 1450 (Centro)","precio":"$150.000","dormitorios":3,"cochera":"no","mascotas":"N/D"},
    ]
    zone_l = zone.lower()
    return [x for x in data if zone_l in x["direccion"].lower()]

def get_listing_by_address_or_link(text: str) -> Optional[Dict[str, Any]]:
    """
    Si el cliente pasa dirección exacta o link, buscá la ficha.
    Reemplazá por tu búsqueda real (por dirección, código o URL).
    """
    m = re.search(r"(https?://\S+)", text)
    if m:
        url = m.group(1)
        # Buscar por URL en tu base si guardás portal/link.
        # Stub:
        return {"codigo":"A101","direccion":"San Luis 234 (Centro)","precio":"$120.000","dormitorios":2,"cochera":"sí","mascotas":"N/D","link":url}

    # Buscar por dirección aproximada (stub):
    if "san luis 234" in text.lower():
        return {"codigo":"A101","direccion":"San Luis 234 (Centro)","precio":"$120.000","dormitorios":2,"cochera":"sí","mascotas":"N/D","link":None}

    return None

# -----------------------------
# Mensajería GreenAPI
# -----------------------------
async def send_whatsapp(chat_id: str, message: str) -> None:
    if not message:
        return
    payload = {"chatId": chat_id, "message": message}
    try:
        r = await http_client.post(GREEN_SEND_URL, json=payload)
        r.raise_for_status()
    except Exception as e:
        print("ERROR sending WA:", e, r.text if 'r' in locals() else "")

# -----------------------------
# Pydantic
# -----------------------------
class InboundPayload(BaseModel):
    chatId: str
    message: str
    isFromMe: Optional[bool] = False
    senderName: Optional[str] = None

# -----------------------------
# LLM prompt (una sola vez, IA comercial)
# -----------------------------
SYSTEM_PROMPT = """
Sos un asesor comercial de **Veglienzone Gestión Inmobiliaria**. Tono humano, cordial y directo.
Tareas:
1) Detectar si el usuario habla de ALQUILER o VENTA (aunque lo diga con variantes como "me gustaría alquilar").
2) Si trae link o dirección exacta, responder consultas de esa ficha. Luego, si se muestra interesado, ofrecé derivarlo a un asesor humano.
3) Si NO trae dirección exacta: preguntar si busca por zona/barrio. Cuando diga una zona (ej. "Centro"), envía una lista breve de 2-3 opciones: código, dirección, precio, dormitorios, cochera, mascotas. Luego preguntá si le interesa alguna en particular o si quiere ajustar filtros (dormitorios, precio, mascotas, cochera).
4) Para ALQUILER: si dice una unidad que le interesa, calificá: ingresos demostrables que tripliquen el costo, tipo de garantía (seguro de caución Finaer o garantía propietaria), cantidad de habitantes y si tiene mascotas.
5) Para VENTA: calificá en base a interés y presupuesto, pero no uses m2 como filtro obligatorio.
6) Cuando la calificación esté completa, preguntá si quiere ser derivado con un asesor humano. Si dice que sí, marcá vendor_push=true y armá un vendor_message claro con: nombre (si lo sabemos), operación (alquiler/venta), zona o dirección, requisitos que cumple y teléfono/chatId. NO copies text del cliente; RESUMÍ profesionalmente.
7) Soportá el comando “reset”: saludá y empezá desde cero.
8) Saludo inicial: “Gracias por contactarte con el área comercial de Veglienzone Gestión Inmobiliaria. ¿Cómo podemos ayudarte hoy? 1- Alquileres 2- Ventas 3- Tasaciones”. Aclarar: “Nota: si en cualquier momento escribís reset, la conversación se reinicia desde cero.”

Salida SIEMPRE en JSON estricto:
{
  "reply_text": "texto para el cliente",
  "closing_text": "opcional",
  "vendor_push": false,
  "vendor_message": ""
}
"""

# -----------------------------
# IA helpers
# -----------------------------
async def call_llm(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Llama al modelo y retorna JSON con reply_text, closing_text, vendor_push, vendor_message
    """
    try:
        resp = openai_lib.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=600,
        )
        txt = resp.choices[0].message["content"].strip()
        # Intento parsear JSON
        first_brace = txt.find("{")
        last_brace = txt.rfind("}")
        if first_brace != -1 and last_brace != -1:
            txt = txt[first_brace:last_brace+1]
        data = json.loads(txt)
        # Defaults
        data.setdefault("reply_text","")
        data.setdefault("closing_text","")
        data.setdefault("vendor_push", False)
        data.setdefault("vendor_message","")
        return data
    except Exception as e:
        print("LLM ERROR:", e)
        return {
            "reply_text": "Perdón, estoy con un inconveniente técnico. ¿Podés repetir tu consulta?",
            "closing_text": "",
            "vendor_push": False,
            "vendor_message": ""
        }

def normalize_message(m: str) -> str:
    return (m or "").strip()

def is_reset(m: str) -> bool:
    return normalize_message(m).lower() == "reset"

def has_address_or_link(m: str) -> bool:
    if re.search(r"https?://\S+", m, re.I): 
        return True
    # heurística simple de dirección: calle + número
    if re.search(r"[a-záéíóúñ]+\s+\d{2,5}", m, re.I):
        return True
    return False

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI()

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/api/inbound")
async def inbound(payload: InboundPayload = Body(...)):
    chat_id = payload.chatId
    text    = normalize_message(payload.message or "")
    from_me = bool(payload.isFromMe)

    # 1) Anti-loop: ignoro mensajes “míos”
    if from_me:
        return {"ok": True, "ignored":"from_me"}

    # 1.1) anti-duplicado simple
    now = time.time()
    d = dedup.get(chat_id, {"ts":0,"last":""})
    if d["last"] == text and (now - d["ts"] < 2.0):
        return {"ok": True, "ignored":"duplicate"}
    dedup[chat_id] = {"ts": now, "last": text}

    # 2) reset
    if is_reset(text):
        sessions[chat_id] = {"history":[]}
        await send_whatsapp(chat_id,
            "Gracias por contactarte con el área comercial de Veglienzone Gestión Inmobiliaria. "
            "¿Cómo podemos ayudarte hoy?\n1- Alquileres\n2- Ventas\n3- Tasaciones\n\n"
            "Nota: si en cualquier momento escribís *reset*, la conversación se reinicia desde cero."
        )
        return {"ok": True, "reset": True}

    # 3) session
    sess = sessions.setdefault(chat_id, {"history":[]})

    # 4) Si el usuario mandó dirección/link → intentamos ficha rápida *antes* del LLM
    quick_reply = None
    if has_address_or_link(text):
        ficha = get_listing_by_address_or_link(text)
        if ficha:
            quick_reply = (
                f"¡Genial! Sobre esa propiedad:\n"
                f"- Código: {ficha.get('codigo','N/D')}\n"
                f"- Dirección: {ficha.get('direccion','N/D')}\n"
                f"- Precio: {ficha.get('precio','N/D')}\n"
                f"- Dormitorios: {ficha.get('dormitorios','N/D')} – Cochera: {ficha.get('cochera','N/D')}\n"
                f"{'(Link: '+ficha['link']+')' if ficha.get('link') else ''}\n\n"
                "¿Querés que avancemos con esta unidad o preferís ver opciones similares?"
            )

    # 5) Construyo messages para el LLM
    user_msg = text
    if quick_reply:
        # Añado la “observación” para que la IA mantenga el flujo (derivar, calificar, etc.)
        user_msg += "\n\n[nota_sistema: ya se detectó ficha por dirección/link; continuar flujo con calificación/derivación]"

    messages = [{"role":"system","content":SYSTEM_PROMPT}]
    messages.extend(sess["history"])
    messages.append({"role":"user","content":user_msg})

    llm_out = await call_llm(messages)

    # Si había quick_reply, lo anteponemos
    reply_text = (quick_reply + "\n\n" + llm_out["reply_text"]) if quick_reply else llm_out["reply_text"]
    closing_text = llm_out.get("closing_text","") or ""
    vendor_push  = bool(llm_out.get("vendor_push", False))
    vendor_message = llm_out.get("vendor_message","") or ""

    # 6) Si NO hay dirección y el usuario ya dijo una zona → proponé fichas (desde DB) (heurística simple)
    if not has_address_or_link(text):
        # heurística muy simple para detectar una zona
        m = re.search(r"(zona|barrio|en)\s+([a-záéíóúñ\s]+)$", text.lower())
        if m:
            zone = m.group(2).strip()
            lst = get_listings_for_rent(zone)
            if lst:
                bloque = ["Opciones de alquiler en *{}*:".format(zone.title())]
                for x in lst[:3]:
                    bloque.append(
                        f"- *{x['codigo']}* | {x['direccion']}\n"
                        f"  Precio: {x['precio']} – Dorms: {x['dormitorios']} – Cochera: {x['cochera']} – Mascotas: {x['mascotas']}"
                    )
                bloque.append("\n¿Te interesa alguna en particular o querés ajustar filtros (dormitorios, precio, cochera, mascotas)?")
                reply_text = ("\n".join(bloque)) + ("\n\n" + reply_text if reply_text else "")

    # 7) Enviar respuesta al cliente
    await send_whatsapp(chat_id, reply_text)
    if closing_text:
        await send_whatsapp(chat_id, closing_text)

    # 8) Notificación a vendedor SOLO si vendor_push == True
    if vendor_push and vendor_message:
        await send_whatsapp(VENDOR_CHATID, vendor_message)

    # 9) Actualizo memoria
    sess["history"].append({"role":"user","content":text})
    # Guardamos el reply truncado para no crecer infinito
    sess["history"].append({"role":"assistant","content":reply_text[:1200]})

    return {
        "ok": True,
        "reply_text": reply_text,
        "closing_text": closing_text,
        "vendor_push": vendor_push
    }
