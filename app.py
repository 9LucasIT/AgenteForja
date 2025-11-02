import os
import json
import logging
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("fastapi-agent")

# -----------------------------------------------------------------------------
# Env vars (solo las que necesitamos para que el flujo funcione)
# -----------------------------------------------------------------------------
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "").strip()

# Opcionales (para /debug y consistencia con tu despliegue actual)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GREEN_ID = os.getenv("GREEN_ID", "")
GREEN_TOKEN = os.getenv("GREEN_TOKEN", "")
VENDOR_PHONE = os.getenv("VENDOR_PHONE", "")

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="FastAPI WhatsApp Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _mask(v: Optional[str]) -> str:
    """Enmascara valores sensibles para /debug."""
    if not v:
        return ""
    if len(v) <= 6:
        return "*" * len(v)
    return v[:3] + "*" * (len(v) - 6) + v[-3:]


def _ensure_n8n() -> None:
    if not N8N_WEBHOOK_URL:
        log.error("N8N_WEBHOOK_URL no está configurada en Railway.")
        raise HTTPException(status_code=500, detail="N8N_WEBHOOK_URL no configurada")


def _json(request: Request) -> Dict[str, Any]:
    try:
        return json.loads(request._body) if hasattr(request, "_body") else {}
    except Exception:
        return {}


def _normalize_green_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recibe el JSON crudo de GreenAPI y devuelve el esquema que usa tu n8n:

    {
      "instanceId": 7107365363,
      "typeWebhook": "incomingmessagereceived",
      "idMessage": "...",
      "user_phone": "5493412565812",
      "chatId": "5493412565812@c.us",
      "text": "Hola",
      "isFromMe": false
    }
    """
    # Green llega en raw["body"] o plano (según config). Damos soporte a ambos.
    body = raw.get("body") if isinstance(raw.get("body"), dict) else raw

    # Tipo de evento
    type_webhook = (body.get("typeWebhook") or raw.get("typeWebhook") or "").strip()
    type_webhook = type_webhook.lower().replace(" ", "")

    # Datos de instancia
    instance_data = body.get("instanceData", {})
    instance_id = instance_data.get("idInstance")

    # Identificadores y remitente
    sender_data = body.get("senderData", {}) or {}
    chat_id = sender_data.get("chatId")
    sender = sender_data.get("sender")  # ej: 549xxx@c.us

    # Texto (según tipo)
    text = ""
    msg_data = body.get("messageData") or {}
    if msg_data.get("typeMessage") == "textMessage":
        text = (msg_data.get("textMessageData") or {}).get("textMessage", "") or ""
    # Respaldo: Green a veces manda 'message' o 'text' distinto
    text = text or body.get("message") or body.get("text") or ""

    # isFromMe: si el webhook es de mensajes salientes:
    #  - "Receive webhooks on messages sent from phone" -> tendrían otro tipo,
    #  pero lo más seguro es inferir: si sender = wid propio, podría ser tuyo.
    wid = (instance_data.get("wid") or "").lower()
    is_from_me = False
    if isinstance(sender, str) and wid and sender.lower().startswith(wid.split("@")[0]):
        is_from_me = True

    # user_phone sin sufijo '@c.us'
    user_phone = ""
    if sender and isinstance(sender, str):
        user_phone = sender.split("@")[0]

    # idMessage
    id_message = body.get("idMessage") or raw.get("idMessage")

    normalized = {
        "instanceId": instance_id,
        "typeWebhook": type_webhook,
        "idMessage": id_message,
        "user_phone": user_phone,
        "chatId": chat_id,
        "text": text,
        "isFromMe": bool(is_from_me),
    }

    return normalized


def _post_to_n8n(payload: Dict[str, Any]) -> requests.Response:
    _ensure_n8n()
    log.info(f"Reenviando a n8n → {N8N_WEBHOOK_URL}")
    try:
        r = requests.post(
            N8N_WEBHOOK_URL,
            json=payload,
            timeout=15,
            headers={"Content-Type": "application/json"},
        )
        log.info(f"n8n respondió: {r.status_code}")
        return r
    except requests.RequestException as exc:
        log.exception("Error enviando a n8n")
        raise HTTPException(status_code=502, detail=f"n8n no respondió: {exc}")


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "fastapi-agent"}


@app.get("/health")
@app.get("/healthz")
def health():
    return {"ok": True}


@app.get("/debug")
def debug():
    """Muestra envs claves (enmascaradas) para diagnóstico rápido."""
    return {
        "llm_mode": "openai" if OPENAI_API_KEY else "disabled",
        "OPENAI_MODEL": OPENAI_MODEL,
        "GREEN_ID": _mask(GREEN_ID),
        "GREEN_TOKEN": _mask(GREEN_TOKEN),
        "N8N_WEBHOOK_URL": N8N_WEBHOOK_URL,  # no se enmascara para que lo chequees fácil
        "VENDOR_PHONE": _mask(VENDOR_PHONE),
    }


@app.post("/api/green/inbound")
async def inbound_green(request: Request):
    """
    Webhook de entrada desde GreenAPI.
    - Normaliza los datos para n8n.
    - Ignora mensajes 'from me' (evita loops).
    - Reenvía a n8n y devuelve 200 si todo ok.
    """
    try:
        raw = await request.json()
    except Exception:
        raw = _json(request)  # último recurso
    log.info(f"Webhook Green recibido: {json.dumps(raw)[:1000]}")

    # Casos de cuota excedida u otros que no deben disparar flujo
    if (raw.get("typeWebhook") or "").lower() == "quotaexceeded" or \
       (raw.get("body", {}).get("typeWebhook") or "").lower() == "quotaexceeded":
        log.warning("Evento quotaExceeded recibido. Ignorando.")
        return {"ignored": True, "reason": "quotaExceeded"}

    normalized = _normalize_green_payload(raw)
    log.info(f"Payload normalizado: {normalized}")

    # Anti-loop: si el mensaje es 'mío', no proceso
    if normalized.get("isFromMe"):
        log.info("Mensaje enviado por mí (isFromMe=True). Ignorando.")
        return {"ignored": True, "reason": "from_me"}

    # Si no viene texto ni chat, no podemos hacer nada
    if not normalized.get("chatId"):
        log.warning("Falta chatId en payload normalizado. Ignorando.")
        return {"ignored": True, "reason": "missing_chat_id"}

    # Reenviar a n8n
    r = _post_to_n8n(normalized)

    # Si n8n no responde 2xx, devolvemos 502 para detectarlo rápido
    if r.status_code // 100 != 2:
        raise HTTPException(
            status_code=502,
            detail=f"n8n devolvió {r.status_code}: {r.text[:200]}",
        )

    return {"ok": True, "forwarded": True}


# ----------------------------------------------------------------------------- 
# (Opcional) Endpoint simple para forzar pruebas manuales desde Postman
# -----------------------------------------------------------------------------
@app.post("/api/test/echo")
async def test_echo(payload: Dict[str, Any]):
    log.info(f"/api/test/echo: {payload}")
    return {"ok": True, "echo": payload}
