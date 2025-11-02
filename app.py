import os
import re
import json
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

app = FastAPI(title="FastAPI Agent – Veglienzone")

# === ENV (mismos nombres que ya tenés en Railway) ===
GREEN_ID = os.getenv("GREEN_ID", "").strip()
GREEN_TOKEN = os.getenv("GREEN_TOKEN", "").strip()
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "").strip()            # ej: https://nicogalarza.app.n8n.cloud/webhook/veglienzone/vendor-push
VENDOR_PHONE = os.getenv("VENDOR_PHONE", "5493412654593").strip()     # queda igual como pediste
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")               # no dependemos de esto para responder
# Nota: no usamos DB aquí. Es el mismo comportamiento “stateless” que venías usando.

GREEN_BASE = "https://api.green-api.com"

# === Memoria simple en RAM por chatId (se pierde en deploy/restart – mismo comportamiento de antes) ===
STATE = {}  # { chatId: {"mode": "start"|"alquiler"|"venta"|"tiene_link"|"zona"|"calif_alq", ...} }

# ========= Helpers =========

def _msg(txt: str) -> str:
    """Normaliza saltos de línea para WhatsApp (opcional)."""
    return txt.replace("\r\n", "\n").strip()

async def green_send(chat_id: str, message: str) -> None:
    """Envía un texto al usuario por GreenAPI."""
    url = f"{GREEN_BASE}/waInstance{GREEN_ID}/sendMessage/{GREEN_TOKEN}"
    payload = {"chatId": chat_id, "message": message}
    async with httpx.AsyncClient(timeout=20) as client:
        await client.post(url, json=payload)

async def push_to_n8n(chat_id: str, vendor_message: str) -> None:
    """Dispara el webhook de n8n para derivación al vendedor."""
    if not N8N_WEBHOOK_URL:
        return
    payload = {
        "chatId": chat_id,
        "vendor_phone": VENDOR_PHONE,
        "message": vendor_message
    }
    async with httpx.AsyncClient(timeout=20) as client:
        await client.post(N8N_WEBHOOK_URL, json=payload)

def is_link_or_address(text: str) -> bool:
    # muy simple: URL o patrón “calle 123 …”
    if re.search(r"https?://", text, re.I):
        return True
    if re.search(r"\b([A-Za-zÁÉÍÓÚÑáéíóúñ]+\s+\d{2,5})\b", text):
        return True
    return False

def intent_from_text(text: str) -> str:
    t = text.lower()
    # sinónimos mínimos para “alquiler/venta”
    if any(k in t for k in ["alquilo", "alquiler", "alquilar", "quiero alquilar", "busco alquiler"]):
        return "alquiler"
    if any(k in t for k in ["vendo", "venta", "comprar", "compraría", "quiero comprar"]):
        return "venta"
    return "otro"

def greeting() -> str:
    return _msg(
        "Gracias por contactarte con el área comercial de Veglienzone Gestión Inmobiliaria.\n"
        "¿Cómo podemos ayudarte hoy?\n"
        "1- Alquileres\n"
        "2- Ventas\n"
        "3- Tasaciones\n\n"
        "Nota: si en cualquier momento escribís *reset*, la conversación se reinicia desde cero."
    )

# ========= RUTAS TÉCNICAS =========

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug")
def debug():
    # Muestra ENV (enmascarando valores) y estado (conteo)
    masked = {
        "GREEN_ID": GREEN_ID[:4] + "…",
        "GREEN_TOKEN": (GREEN_TOKEN[:6] + "…") if GREEN_TOKEN else "",
        "N8N_WEBHOOK_URL": N8N_WEBHOOK_URL,
        "OPENAI_MODEL": OPENAI_MODEL,
    }
    return {"env": masked, "sessions": len(STATE)}

# ========= WEBHOOK QUE LLAMA GREENAPI =========
# (esto es lo que en tus logs figura como 404: /api/green/inbound)

@app.post("/api/green/inbound")
async def green_inbound(req: Request):
    body = await req.json()
    # Green puede mandar distintos 'typeWebhook'
    type_wh = (body.get("typeWebhook") or "").lower()

    # 1) Guardas cuotas/limites: ignoramos
    if type_wh == "quotaexceeded":
        return JSONResponse({"ignored": "quotaExceeded"}, status_code=200)

    # 2) Solo respondemos a incomingMessageReceived
    if type_wh != "incomingmessagereceived":
        return JSONResponse({"ignored": type_wh}, status_code=200)

    # 3) Parse datos relevantes
    sender = body.get("senderData", {}) or {}
    chat_id = sender.get("chatId") or ""
    is_from_me = False  # Green en este webhook ya viene como entrante, tomamos como false

    msg_data = (body.get("messageData") or {})
    text = ""
    if (msg_data.get("typeMessage") == "textMessage") and msg_data.get("textMessageData"):
        text = msg_data["textMessageData"].get("textMessage") or ""

    if not chat_id:
        return JSONResponse({"error": "no chatId"}, status_code=200)

    # Anti-loop básico
    if is_from_me:
        return JSONResponse({"ignored": "from_me"}, status_code=200)

    # RESET
    if text.strip().lower() == "reset":
        STATE.pop(chat_id, None)
        await green_send(chat_id, greeting())
        return JSONResponse({"ok": True}, status_code=200)

    # Estado actual
    st = STATE.get(chat_id, {"mode": "start"})
    mode = st.get("mode", "start")

    # 0) Arranque si no hay estado
    if mode == "start":
        # ¿Detectamos intención directa?
        it = intent_from_text(text)
        if it == "alquiler":
            STATE[chat_id] = {"mode": "alquiler"}  # siguiente paso
            await green_send(chat_id, _msg("¿Tenés una dirección o link exacto, o estás buscando por una *zona* en particular?"))
            return JSONResponse({"ok": True}, status_code=200)
        elif it == "venta":
            STATE[chat_id] = {"mode": "venta"}
            await green_send(chat_id, _msg("¡Genial! Sobre *ventas*, ¿tenés una dirección o link exacto, o querés consultar por una *zona*?"))
            return JSONResponse({"ok": True}, status_code=200)
        else:
            # saludo
            await green_send(chat_id, greeting())
            return JSONResponse({"ok": True}, status_code=200)

    # 1) ALQUILER: primer bifurcación (link/dirección vs zona)
    if mode == "alquiler":
        if is_link_or_address(text):
            STATE[chat_id] = {"mode": "calif_alq"}
            await green_send(chat_id, _msg(
                "Perfecto. Para *alquiler*, ¿tenés ingresos demostrables que tripliquen el costo, "
                "qué *tipo de garantía* usarías (seguro de caución Finaer o garantía propietaria), "
                "cantidad de *habitantes* y si *tienen mascotas*?"
            ))
            return JSONResponse({"ok": True}, status_code=200)
        else:
            # Consideramos que es zona/barrio
            STATE[chat_id] = {"mode": "zona_alq", "zona": text}
            await green_send(chat_id, _msg(
                "Entendido. Sobre esa *zona*, ¿te interesan *departamentos* o *casas*? "
                "Si querés, decime presupuesto y cantidad de dormitorios."
            ))
            return JSONResponse({"ok": True}, status_code=200)

    # 2) VENTA (simplemente eco de intención)
    if mode == "venta":
        if is_link_or_address(text):
            # venta puntual: pedimos datos básicos y derivamos si quiere
            STATE[chat_id] = {"mode": "venta_detalle"}
            await green_send(chat_id, _msg(
                "Bien. ¿Querés que te pase con un asesor humano para coordinar y avanzamos? (podés responder *sí* para derivarte)"
            ))
            return JSONResponse({"ok": True}, status_code=200)
        else:
            STATE[chat_id] = {"mode": "zona_venta", "zona": text}
            await green_send(chat_id, _msg(
                "Perfecto. ¿Buscás *casa* o *departamento*? Si tenés rango de precio, mejor."
            ))
            return JSONResponse({"ok": True}, status_code=200)

    # 3) Zona alquiler -> pedimos más y luego ofrecemos derivación
    if mode == "zona_alq":
        # Guardamos preferencia y ofrecemos derivar
        STATE[chat_id] = {"mode": "alquiler_oferta"}
        await green_send(chat_id, _msg(
            "Con eso ya puedo buscar opciones. ¿Querés que te envíe algunas por acá y, si te interesa alguna, "
            "te derivo a un asesor humano para coordinar? (Escribí *sí* para derivarte)"))
        return JSONResponse({"ok": True}, status_code=200)

    # 4) Calificación alquiler puntual
    if mode == "calif_alq":
        # ya nos dieron datos; ofrecemos derivación
        vendor_message = _msg(
            "Lead calificado – *Alquiler*.\n"
            f"Datos del cliente: {text}\n"
            "Solicitar seguimiento."
        )
        # Preguntamos antes:
        STATE[chat_id] = {"mode": "confirm_vendor", "vendor_msg": vendor_message}
        await green_send(chat_id, _msg("Con esos datos te puedo derivar a un asesor humano para avanzar. ¿Te contacto por este WhatsApp? (respondé *sí* para derivarte)"))
        return JSONResponse({"ok": True}, status_code=200)

    # 5) Venta detalle o zona venta -> ofrecemos derivar
    if mode in ("venta_detalle", "zona_venta", "alquiler_oferta"):
        if re.search(r"\b(s[ií]|si|okay|dale|ok)\b", text.lower()):
            vendor_message = _msg(
                f"Lead para *{mode}*.\n"
                f"Mensaje del cliente: {text}\n"
                "Solicitar seguimiento."
            )
            await push_to_n8n(chat_id, vendor_message)
            await green_send(chat_id, _msg("¡Listo! En breve te contacta un asesor humano por este WhatsApp."))
            STATE.pop(chat_id, None)
            return JSONResponse({"ok": True}, status_code=200)
        else:
            await green_send(chat_id, _msg("Perfecto, sigo por acá. Cuando quieras derivarte, respondé *sí*."))
            return JSONResponse({"ok": True}, status_code=200)

    # 6) Confirmación de derivación (desde calif_alq)
    if mode == "confirm_vendor":
        if re.search(r"\b(s[ií]|si|dale|ok)\b", text.lower()):
            vendor_message = st.get("vendor_msg", "Lead para seguimiento.")
            await push_to_n8n(chat_id, vendor_message)
            await green_send(chat_id, _msg("¡Listo! En breve te contacta un asesor humano por este WhatsApp."))
            STATE.pop(chat_id, None)
            return JSONResponse({"ok": True}, status_code=200)
        else:
            await green_send(chat_id, _msg("Perfecto, seguimos por acá. Cuando quieras derivarte, respondé *sí*."))
            return JSONResponse({"ok": True}, status_code=200)

    # fallback
    await green_send(chat_id, greeting())
    STATE[chat_id] = {"mode": "start"}
    return JSONResponse({"ok": True}, status_code=200)
