import os
import json
import logging
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

# ----------------------------
# Config & logger
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("agent")

N8N_VENDOR_WEBHOOK = os.getenv("N8N_VENDOR_WEBHOOK")  # p.ej: https://nicogalarza.app.n8n.cloud/webhook/veglienzone/vendor-push
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # no se usa aquí, pero lo dejo para continuidad
PORT = int(os.getenv("PORT", "8000"))

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="fastapi-agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Utils
# ----------------------------
def normalize_green_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Devuelve un dict 'plano' para que n8n lo entienda siempre igual."""
    body = raw.get("body") or raw  # a veces n8n te lo encierra dentro de "body"
    t = (body.get("typeWebhook") or "").lower()

    # Defaults
    result: Dict[str, Any] = {
        "typeWebhook": t,
        "instanceId": body.get("instanceData", {}).get("idInstance"),
        "idMessage": body.get("idMessage") or "",
        "chatId": "",
        "user_phone": "",
        "text": "",
        "isFromMe": False,
        "raw": body,
    }

    # Casos de cuota excedida (Green manda esto a veces)
    if t == "quotaexceeded":
        return result

    # incomingMessageReceived
    sender = body.get("senderData") or {}
    result["chatId"] = sender.get("chatId", "")
    result["user_phone"] = (sender.get("sender") or "").replace("@c.us", "")
    msg = body.get("messageData") or {}
    if (msg.get("typeMessage") == "textMessage") and msg.get("textMessageData"):
        result["text"] = msg["textMessageData"].get("textMessage", "")

    # Green a veces no manda isFromMe (dejamos False)
    return result


async def post_to_n8n(payload: Dict[str, Any]) -> Optional[int]:
    """Reenvía a n8n el evento ya normalizado."""
    if not N8N_VENDOR_WEBHOOK:
        log.warning("N8N_VENDOR_WEBHOOK no configurado; no se reenviará a n8n")
        return None

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                N8N_VENDOR_WEBHOOK,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        log.info("n8n response: %s %s", r.status_code, r.text[:300])
        return r.status_code
    except Exception as e:
        log.exception("Error posteando a n8n: %s", e)
        return None


async def handle_green(request: Request) -> JSONResponse:
    """Manejador único para los webhooks; retorna 200 siempre que lo podamos leer."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    log.info("GREEN HOOK HIT -> %s", json.dumps(body)[:1000])

    normalized = normalize_green_payload(body)

    # Reenvía a n8n (no bloquea la respuesta a Green; igual esperamos rápido)
    await post_to_n8n(normalized)

    # Green sólo necesita 200
    return JSONResponse({"ok": True})

# ----------------------------
# Rutas (ambas para evitar 404)
# ----------------------------
@app.get("/health")
@app.get("/healthz")
async def health():
    return {"ok": True}

@app.post("/green/inbound")
async def green_inbound_no_prefix(request: Request):
    return await handle_green(request)

@app.post("/api/green/inbound")
async def green_inbound_with_prefix(request: Request):
    return await handle_green(request)

# ----------------------------
# Main (local)
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
