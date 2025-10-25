import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VENDOR_PHONE = re.sub(r"\D", "", os.getenv("VENDOR_PHONE", ""))  # 549...
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY es requerido")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Compat Railway variables
    MYSQL_USER = os.getenv("MYSQLUSER") or os.getenv("MYSQL_USER") or "root"
    MYSQL_PASS = os.getenv("MYSQLPASSWORD") or os.getenv("MYSQL_PASSWORD") or ""
    MYSQL_HOST = os.getenv("MYSQLHOST") or os.getenv("MYSQL_HOST") or "mysql.railway.internal"
    MYSQL_PORT = os.getenv("MYSQLPORT") or os.getenv("MYSQL_PORT") or "3306"
    MYSQL_DB   = os.getenv("MYSQL_DATABASE") or os.getenv("MYSQLDB") or "railway"
    DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

engine: Engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={"charset": "utf8mb4"},
)

app = FastAPI()

# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------
SLOT_ORDER = ["direccion", "zona", "presupuesto_min", "presupuesto_max", "dormitorios", "cochera", "mascotas"]
YES = {"si", "sí", "claro", "ok", "vale", "dale", "yes", "y", "s"}
NO  = {"no", "nop", "negativo", "n"}

def now():
    return datetime.utcnow()

def fetchone(q: str, p: Dict[str, Any] = None):
    with engine.begin() as conn:
        return conn.execute(text(q), p or {}).mappings().first()

def execute(q: str, p: Dict[str, Any] = None):
    with engine.begin() as conn:
        conn.execute(text(q), p or {})

def upsert_session(user_phone: str, mutator):
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT * FROM chat_session WHERE user_phone=:p"),
            {"p": user_phone},
        ).mappings().first()

        if not row:
            payload = {
                "user_phone": user_phone,
                "last_message_id": None,
                "last_welcome_at": None,
                "cooldown_until": None,
                "status": "active",
                "slots_json": json.dumps({}),
                "created_at": now(),
                "updated_at": now(),
            }
            conn.execute(text("""
                INSERT INTO chat_session
                (user_phone,last_message_id,last_welcome_at,cooldown_until,status,slots_json,created_at,updated_at)
                VALUES (:user_phone,:last_message_id,:last_welcome_at,:cooldown_until,:status,:slots_json,:created_at,:updated_at)
            """), payload)
            row = conn.execute(
                text("SELECT * FROM chat_session WHERE user_phone=:p"),
                {"p": user_phone},
            ).mappings().first()

        slots = {}
        try:
            slots = json.loads(row["slots_json"] or "{}")
        except Exception:
            slots = {}

        new_row = dict(row)
        new_row["slots_json"] = slots
        mutator(new_row)

        new_row["slots_json"] = json.dumps(new_row["slots_json"] or {})
        new_row["updated_at"] = now()

        conn.execute(text("""
            UPDATE chat_session
            SET last_message_id=:last_message_id,
                last_welcome_at=:last_welcome_at,
                cooldown_until=:cooldown_until,
                status=:status,
                slots_json=:slots_json,
                updated_at=:updated_at
            WHERE id=:id
        """), new_row)

        return new_row

def parse_money(texto: str) -> Optional[int]:
    # "60k", "$60.000", "60000"
    t = texto.replace(",", ".")
    m = re.findall(r"([\$]?\s*\d{1,3}(?:[.\s]\d{3})+|\d{4,})", t)
    if not m:
        m2 = re.findall(r"\b(\d{1,3})\s*[kK]\b", t)
        if m2:
            try: return int(m2[0]) * 1000
            except Exception: return None
        return None
    num = re.sub(r"[^\d]", "", m[0])
    try: return int(num)
    except Exception: return None

def parse_yes_no(texto: str) -> Optional[bool]:
    t = texto.strip().lower()
    if t in YES: return True
    if t in NO:  return False
    return None

def safe_int(x, d=None):
    try: return int(x)
    except Exception: return d

def detect_direccion(texto: str) -> Optional[str]:
    """
    Detecta algo tipo 'San Luis 234', 'Mendoza 1500', 'Av. Pellegrini 800'
    Evita confundir con presupuesto (número sin calle antes).
    """
    t = texto.strip()
    # calle + número (la calle debe tener al menos 3 letras)
    m = re.search(r"([a-záéíóúñ\.]{3,}(?:\s+[a-záéíóúñ\.]{2,})*)\s+(\d{1,6})", t, flags=re.I)
    if m:
        calle = m.group(1).strip()
        num   = m.group(2).strip()
        # Evitamos capturar cuando es sólo un número mencionado solo (ej. "60000")
        if not re.fullmatch(r"\d{4,}", t):
            return f"{calle.title()} {num}"
    return None

# ---------------------------------------------------------
# LLM (tono humano + 1 pregunta por turno)
# ---------------------------------------------------------
SYSTEM = """Eres una asesora inmobiliaria muy humana, breve y cálida (rioplatense).
Vamos paso a paso y hacemos UNA sola pregunta por turno.
Slots a completar (en este orden):
1) direccion exacta (si el cliente la da, tomarla y no pidas zona)
2) zona (si no hay direccion)
3) presupuesto_min (ARS)
4) presupuesto_max (ARS)
5) dormitorios (entero)
6) cochera (sí/no)
7) mascotas (¿aceptan mascotas? sí/no)

Reglas:
- Saluda y confirma brevemente lo entendido.
- Nunca hagas dos preguntas en el mismo mensaje.
- No repitas una pregunta si ya fue respondida; si hay duda, pedí una aclaración simple.
- Cuando tengas todos los datos, avisa que ya está y que pronto lo contacta el equipo.
- Tono humano y natural (sin sonar robótica). Puedes usar un emoji suave cada tanto.
"""

def llm_complete(messages: list, temperature: float = 0.5) -> str:
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"model": "gpt-4o-mini", "temperature": temperature, "messages": messages},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return "¿Me contás eso con un poquito más de detalle? Gracias."

# ---------------------------------------------------------
# Core
# ---------------------------------------------------------
class QualifyIn(BaseModel):
    user_phone: str
    message_id: Optional[str] = None
    text: str

def build_confirmation(slots: Dict[str, Any]) -> str:
    chunks = []
    if slots.get("direccion"):
        chunks.append(f"dirección {slots['direccion']}")
    elif slots.get("zona"):
        chunks.append(f"zona {slots['zona']}")
    if slots.get("presupuesto_min") and slots.get("presupuesto_max"):
        chunks.append(f"rango ${slots['presupuesto_min']:,}–${slots['presupuesto_max']:,}".replace(",", "."))
    elif slots.get("presupuesto_min"):
        chunks.append(f"mínimo ${slots['presupuesto_min']:,}".replace(",", "."))
    if slots.get("dormitorios"):
        chunks.append(f"{slots['dormitorios']} dormitorios")
    if slots.get("cochera") is not None:
        chunks.append("con cochera" if slots["cochera"] else "sin cochera")
    if slots.get("mascotas") is not None:
        chunks.append("acepta mascotas" if slots["mascotas"] else "sin mascotas")
    return ("Perfecto, " + ", ".join(chunks) + ".").strip(" ,.") if chunks else ""

def next_missing_slot(slots: Dict[str, Any]) -> Optional[str]:
    for s in SLOT_ORDER:
        if slots.get(s) in (None, "", 0):
            return s
    return None

def question_for(slot: str) -> str:
    return {
        "direccion":        "¿Tenés una dirección exacta? (calle y número)",
        "zona":             "¿En qué zona o barrio te gustaría buscar?",
        "presupuesto_min":  "¿Cuál sería tu presupuesto mínimo aproximado (en ARS)?",
        "presupuesto_max":  "¿Y el presupuesto máximo (en ARS)?",
        "dormitorios":      "¿Cuántos dormitorios te gustaría?",
        "cochera":          "¿Vas a necesitar cochera?",
        "mascotas":         "¿La propiedad debe aceptar mascotas?",
    }.get(slot, "¿Podrías contarme un poco más?")

def try_fill_from_text(slots: Dict[str, Any], texto: str) -> Dict[str, Any]:
    t = texto.strip()

    # direccion (si el cliente dio calle + número, priorizamos esto sobre zona)
    if not slots.get("direccion"):
        d = detect_direccion(t)
        if d:
            slots["direccion"] = d
            # Si hay dirección, zona ya no es obligatoria, pero si la dice la tomamos
    # zona (si no hay dirección)
    if not slots.get("direccion") and not slots.get("zona"):
        z = re.findall(r"[a-záéíóúñ]{4,}(?:\s+[a-záéíóúñ]{2,})?", t, flags=re.I)
        if z:
            zz = z[0].strip().title()
            if zz not in {"Hola", "Presupuesto", "Dormitorios", "Cochera", "Mascotas"}:
                slots["zona"] = zz

    # dinero
    money = parse_money(t)
    if money:
        missing = next_missing_slot(slots)
        if missing in ("presupuesto_min", "presupuesto_max"):
            slots[missing] = money
        else:
            if not slots.get("presupuesto_min"):
                slots["presupuesto_min"] = money
            elif not slots.get("presupuesto_max"):
                slots["presupuesto_max"] = money

    # dormitorios
    if not slots.get("dormitorios"):
        d = re.findall(r"\b([1-6])\b", t)
        if d:
            slots["dormitorios"] = int(d[0])

    # cochera
    if slots.get("cochera") is None:
        yn = parse_yes_no(t.lower())
        if yn is not None:
            slots["cochera"] = yn

    # mascotas
    if slots.get("mascotas") is None:
        yn2 = parse_yes_no(t.lower())
        # sólo setear si escribió explícitamente sí/no en un mensaje que hable de mascotas
        if "mascot" in t.lower() and yn2 is not None:
            slots["mascotas"] = yn2

    # coherencia min/max
    pm, px = slots.get("presupuesto_min"), slots.get("presupuesto_max")
    if pm and px and pm > px:
        slots["presupuesto_min"], slots["presupuesto_max"] = px, pm

    return slots

def is_ready(slots: Dict[str, Any]) -> bool:
    # Dirección o zona (una de las dos) + resto completo
    if not (slots.get("direccion") or slots.get("zona")):
        return False
    for s in ["presupuesto_min", "presupuesto_max", "dormitorios", "cochera", "mascotas"]:
        if slots.get(s) in (None, "", 0):
            return False
    return True

def find_properties(slots: Dict[str, Any]) -> list:
    try:
        with engine.begin() as conn:
            if slots.get("direccion"):
                q = text("""
                    SELECT direccion, zona, precio, dormitorios, cochera
                    FROM propiedades
                    WHERE direccion LIKE :d
                      AND precio BETWEEN :pmin AND :pmax
                      AND dormitorios >= :dorm
                      AND cochera >= :coch
                    ORDER BY precio ASC
                    LIMIT 3
                """)
                rows = conn.execute(q, {
                    "d": f"%{slots['direccion'].split()[0]}%",  # matchea por calle
                    "pmin": safe_int(slots["presupuesto_min"], 0),
                    "pmax": safe_int(slots["presupuesto_max"], 999999999),
                    "dorm": safe_int(slots["dormitorios"], 1),
                    "coch": 1 if slots["cochera"] else 0,
                }).mappings().all()
            else:
                q = text("""
                    SELECT direccion, zona, precio, dormitorios, cochera
                    FROM propiedades
                    WHERE zona LIKE :z
                      AND precio BETWEEN :pmin AND :pmax
                      AND dormitorios >= :dorm
                      AND cochera >= :coch
                    ORDER BY precio ASC
                    LIMIT 3
                """)
                rows = conn.execute(q, {
                    "z": f"%{slots.get('zona','')}%",
                    "pmin": safe_int(slots["presupuesto_min"], 0),
                    "pmax": safe_int(slots["presupuesto_max"], 999999999),
                    "dorm": safe_int(slots["dormitorios"], 1),
                    "coch": 1 if slots["cochera"] else 0,
                }).mappings().all()
            return [dict(r) for r in rows]
    except Exception:
        return []

def format_props(props: list) -> str:
    if not props:
        return "Ahora no veo publicaciones que entren justo en lo que buscás. Si te parece, el equipo comercial te contacta y vemos alternativas cercanas 🙂"
    lines = ["Te paso opciones que encajan:"]
    for p in props:
        price = f"${p['precio']:,}".replace(",", ".")
        coch  = "Sí" if p.get("cochera") else "No"
        lines.append(f"• {p['direccion']} ({p['zona']}) — {price}, {p['dormitorios']} dorm., cochera: {coch}")
    return "\n".join(lines)

def notify_vendor(user_phone: str, slots: Dict[str, Any]):
    try:
        # guardamos direccion/mascotas en notas para no tocar tablas
        notas = []
        if slots.get("direccion"): notas.append(f"direccion={slots['direccion']}")
        if slots.get("mascotas") is not None: notas.append(f"mascotas={'si' if slots['mascotas'] else 'no'}")
        notas_str = "; ".join(notas) if notas else None

        execute("""
            INSERT INTO leads (user_phone,inmueble_interes,dormitorios,cochera,presupuesto_min,presupuesto_max,ventana_tiempo,notas,status,vendor_phone,created_at,updated_at)
            VALUES (:u,:z,:d,:c,:pmin,:pmax,NULL,:nt,'pendiente',:v, :ca,:ua)
        """, {
            "u": user_phone,
            "z": slots.get("zona") or slots.get("direccion"),
            "d": slots.get("dormitorios"),
            "c": 1 if slots.get("cochera") else 0,
            "pmin": slots.get("presupuesto_min"),
            "pmax": slots.get("presupuesto_max"),
            "nt": notas_str,
            "v": VENDOR_PHONE or "",
            "ca": now(),
            "ua": now(),
        })
    except Exception:
        pass

# ---------------------------------------------------------
# API
# ---------------------------------------------------------
@app.get("/healthz")
def healthz():
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception:
        return {"ok": True}

@app.post("/qualify")
def qualify(inp: QualifyIn):
    user_phone = re.sub(r"\D", "", inp.user_phone or "")
    msg = (inp.text or "").strip()
    if not user_phone:
        return {"text": "Necesito un número de contacto para seguir.", "next_question": None, "vendor_push": False, "updates": {"slots": {}}}

    try:
        def mutate(row):
            slots = row["slots_json"] or {}
            slots = try_fill_from_text(slots, msg)

            missing = next_missing_slot(slots)
            last_q = slots.get("_last_q")
            if missing and question_for(missing) == last_q:
                # evitá duplicar la misma pregunta textual
                pass

            row["slots_json"] = slots

        row = upsert_session(user_phone, mutate)
        slots = json.loads(row["slots_json"] or "{}")

        conf = build_confirmation(slots)

        missing = next_missing_slot(slots)
        if missing:
            q = question_for(missing)
            text_out = (conf + ("\n" if conf else "") + q).strip()
            upsert_session(user_phone, lambda r: r["slots_json"].update({"_last_q": q}))
            return {
                "text": text_out,
                "next_question": q,
                "vendor_push": False,
                "updates": {"slots": slots}
            }

        # completo → sugerencias + cierre + notify
        props = find_properties(slots)
        listado = format_props(props)
        closing = "Con eso ya estamos. En breve te escribe alguien del equipo para avanzar. ¡Gracias!"
        notify_vendor(user_phone, slots)

        out = (conf + ("\n" if conf else "") + listado + "\n" + closing).strip()
        return {
            "text": out,
            "next_question": None,
            "vendor_push": True,
            "updates": {"slots": slots}
        }

    except Exception:
        fallback = "Gracias. Para seguir, ¿podés pasarme la dirección exacta (calle y número) o, si no la tenés, el barrio/zona? 🙂"
        return {
            "text": fallback,
            "next_question": "¿Dirección exacta o zona?",
            "vendor_push": False,
            "updates": {"slots": {}}
        }
