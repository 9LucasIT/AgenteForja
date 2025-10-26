import os
import random
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# ── DB ─────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")
# Railway a veces expone "mysql://"; SQLAlchemy con PyMySQL usa "mysql+pymysql://"
if DATABASE_URL.startswith("mysql://"):
    DATABASE_URL = DATABASE_URL.replace("mysql://", "mysql+pymysql://")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ── APP ────────────────────────────────────────────────────────
app = FastAPI()


# ── Helpers de conversación ───────────────────────────────────
def human_prefix():
    opciones = [
        "Perfecto 👍", "Genial, gracias por el dato.",
        "Buenísimo 😄", "Súper, así avanzamos.", "Perfecto, ya me ubico."
    ]
    return random.choice(opciones)

def fmt_monto(n):
    if not n and n != 0:
        return "N/D"
    s = str(int(n))
    out = ""
    while s:
        out = ("." + s[-3:] + out) if out else s[-3:]
        s = s[:-3]
    return f"${out}"

def yn(v):
    return "Sí" if v is True else ("No" if v is False else "N/D")

def human_summary(slots: dict):
    oper = slots.get("operacion") or "operación a definir"
    zona = slots.get("zona") or slots.get("direccion_exacta") or "zona a definir"
    pmin = slots.get("presupuesto_min")
    pmax = slots.get("presupuesto_max")
    if pmin or pmax:
        presupuesto = f"{fmt_monto(pmin)}–{fmt_monto(pmax)}" if (pmin and pmax) else fmt_monto(pmin or pmax)
    else:
        presupuesto = "sin presupuesto"
    dorm = f'{slots["dormitorios"]} dorm' if slots.get("dormitorios") else "dorm a definir"
    coch = "con cochera" if slots.get("cochera") else "sin cochera"
    pets = "acepta mascotas" if slots.get("mascotas") else "sin info de mascotas"
    return f"{oper.capitalize()} en {zona}. Presupuesto: {presupuesto}. {dorm}, {coch}, {pets}."

def dedupe_question(proposed: str | None, last_q: str | None):
    if not proposed:
        return None
    if not last_q:
        return proposed
    if proposed.strip().lower() == last_q.strip().lower():
        variantes = [
            "Te pido ese dato para continuar: ",
            "Antes de seguir, confirmame: ",
            "Para afinar la búsqueda: ",
        ]
        return f"{random.choice(variantes)}{proposed}"
    return proposed

def next_question(slots: dict):
    # 🔴 Operación (alquiler/venta) es OBLIGATORIA y va PRIMERO
    if not slots.get("operacion"):
        return "¿La búsqueda es para *alquiler* o para *venta*?"
    if not slots.get("zona") and not slots.get("direccion_exacta"):
        return "¿En qué *zona o barrio* te gustaría buscar?"
    if not (slots.get("presupuesto_min") or slots.get("presupuesto_max")):
        return "¿Cuál sería tu *presupuesto aproximado* (mínimo o máximo)?"
    if not slots.get("dormitorios"):
        return "¿Cuántos *dormitorios* te gustaría?"
    if "cochera" not in slots:
        return "¿Necesitás *cochera*?"
    if "mascotas" not in slots:
        return "¿Tenés *mascotas* que debamos contemplar?"
    if not slots.get("direccion_exacta"):
        return "¿Tenés una *dirección exacta*? (calle y número)"
    return None


# ── Parsers simples (sin LLM) ─────────────────────────────────
def parse_into_slots(text_in: str, slots: dict):
    t = text_in.lower()

    # Operación
    if any(w in t for w in ["alquiler", "alquilar", "rentar", "arrendar", "arriendo"]):
        slots["operacion"] = "alquiler"
    if any(w in t for w in ["venta", "vender", "comprar", "compra"]):
        slots["operacion"] = "venta"

    # Zona (ejemplo básico)
    if "abasto" in t:
        slots["zona"] = "Abasto"

    # Presupuesto: números de 4 a 7 dígitos
    import re
    nums = [int(n) for n in re.findall(r"\b\d{4,7}\b", t)]
    if nums:
        val = max(nums) if len(nums) > 1 else nums[0]
        if "presupuesto_min" not in slots:
            slots["presupuesto_min"] = val
        else:
            if val >= slots.get("presupuesto_min", 0):
                slots["presupuesto_max"] = val

    # Dormitorios
    if "studio" in t or "monoamb" in t:
        slots["dormitorios"] = 0
    elif any(w in t for w in ["1 dorm", "1 dormitorio", "un dormitorio"]):
        slots["dormitorios"] = 1
    elif any(w in t for w in ["2 dorm", "2 dormitorio", "dos dormitorio"]):
        slots["dormitorios"] = 2
    elif any(w in t for w in ["3 dorm", "3 dormitorio", "tres dormitorio"]):
        slots["dormitorios"] = 3

    # Cochera
    if "cochera" in t:
        if any(w in t for w in ["si", "sí", "necesito", "quiero"]):
            slots["cochera"] = True
        if any(w in t for w in ["no", "sin"]):
            slots["cochera"] = False

    # Mascotas
    if "mascota" in t or "mascotas" in t:
        if any(w in t for w in ["si", "sí", "tengo"]):
            slots["mascotas"] = True
        if any(w in t for w in ["no", "sin"]):
            slots["mascotas"] = False

    # Dirección (heurística simple)
    if any(w in t for w in ["calle", "av.", "avenida", "pellegrini", "san luis", "mendoza", "boulevard"]):
        slots["direccion_exacta"] = text_in.strip()

    return slots


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/qualify")
async def qualify(request: Request):
    payload = await request.json()
    phone = str(payload.get("user_phone"))
    text_in = (payload.get("text") or "").strip()

    db = SessionLocal()

    # Recupero/creo sesión
    row = db.execute(text("SELECT * FROM chat_session WHERE user_phone=:p"), {"p": phone}).fetchone()
    if row is None:
        db.execute(text("INSERT INTO chat_session (user_phone, status, created_at) VALUES (:p,'active',NOW())"), {"p": phone})
        db.commit()
        row = db.execute(text("SELECT * FROM chat_session WHERE user_phone=:p"), {"p": phone}).fetchone()

    # Slots actuales
    import json
    slots = {}
    if row.slots_json:
        slots = json.loads(row.slots_json)
    last_q = slots.get("last_question")

    # Parseo mensaje → slots
    slots = parse_into_slots(text_in.lower(), slots)

    # Pregunta siguiente (operación primero)
    q = next_question(slots)
    q = dedupe_question(q, last_q)

    # Mensaje humano
    prefix = human_prefix()
    summary = human_summary(slots)
    if q:
        text_out = f"{prefix}\n{summary}\n{q}"
    else:
        text_out = f"{prefix}\n{summary}\nPerfecto, ya tengo todo. Te conecto con un asesor. 🏠"

    # Persisto
    slots["last_question"] = q
    db.execute(
        text("UPDATE chat_session SET slots_json=:s, updated_at=NOW() WHERE user_phone=:p"),
        {"s": json.dumps(slots), "p": phone}
    )
    db.commit()

    vendor_push = q is None
    return JSONResponse({
        "text": text_out,
        "next_question": q,
        "vendor_push": vendor_push,
        "updates": {"slots": slots}
    })
