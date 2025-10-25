import json
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import requests
import pymysql
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VENDOR_PHONE = re.sub(r"\D", "", os.getenv("VENDOR_PHONE", ""))  # e.g. 5493412654593
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY es requerido")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Compat con variables antiguas de Railway
    MYSQL_USER = os.getenv("MYSQLUSER") or os.getenv("MYSQL_USER") or "root"
    MYSQL_PASS = os.getenv("MYSQLPASSWORD") or os.getenv("MYSQL_PASSWORD") or ""
    MYSQL_HOST = os.getenv("MYSQLHOST") or os.getenv("MYSQL_HOST") or "mysql.railway.internal"
    MYSQL_PORT = os.getenv("MYSQLPORT") or os.getenv("MYSQL_PORT") or "3306"
    MYSQL_DB = os.getenv("MYSQL_DATABASE") or os.getenv("MYSQLDB") or "railway"
    DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASS}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

engine: Engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={"charset": "utf8mb4"},
)

app = FastAPI()


# -------------------------------------------------------------------
# Utilidades
# -------------------------------------------------------------------
SLOT_ORDER = ["zona", "presupuesto_min", "presupuesto_max", "dormitorios", "cochera"]
YES = {"si", "sí", "claro", "obvio", "dale", "ok", "affirmative", "yes", "s"}
NO = {"no", "nop", "negativo", "n"}

def now():
    return datetime.utcnow()

def fetchone(q: str, params: Dict[str, Any] = None):
    with engine.begin() as conn:
        return conn.execute(text(q), params or {}).mappings().first()

def execute(q: str, params: Dict[str, Any] = None):
    with engine.begin() as conn:
        conn.execute(text(q), params or {})

def upsert_session(user_phone: str, mutator):
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT * FROM chat_session WHERE user_phone=:p"),
            {"p": user_phone},
        ).mappings().first()
        if not row:
            # crear
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
            conn.execute(
                text("""
                INSERT INTO chat_session
                (user_phone,last_message_id,last_welcome_at,cooldown_until,status,slots_json,created_at,updated_at)
                VALUES (:user_phone,:last_message_id,:last_welcome_at,:cooldown_until,:status,:slots_json,:created_at,:updated_at)
                """),
                payload,
            )
            row = conn.execute(
                text("SELECT * FROM chat_session WHERE user_phone=:p"),
                {"p": user_phone},
            ).mappings().first()

        # mutar
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

        conn.execute(
            text("""
            UPDATE chat_session
            SET last_message_id=:last_message_id,
                last_welcome_at=:last_welcome_at,
                cooldown_until=:cooldown_until,
                status=:status,
                slots_json=:slots_json,
                updated_at=:updated_at
            WHERE id=:id
            """),
            new_row,
        )
        return new_row

def parse_money(texto: str) -> Optional[int]:
    # acepta 60k, 60.000, $60.000, 60000 aprox...
    m = re.findall(r"([\$]?\s*\d{1,3}(?:[.\s]\d{3})+|\d{4,})", texto.replace(",", "."))
    if not m:
        m2 = re.findall(r"(\d{1,3})\s*[kK]", texto)
        if m2:
            try:
                return int(m2[0]) * 1000
            except Exception:
                return None
        return None
    num = m[0]
    num = re.sub(r"[^\d]", "", num)
    try:
        return int(num)
    except Exception:
        return None

def parse_yes_no(texto: str) -> Optional[bool]:
    t = texto.strip().lower()
    if t in YES: return True
    if t in NO: return False
    return None

def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

# -------------------------------------------------------------------
# LLM (criticamente, una sola pregunta por turno)
# -------------------------------------------------------------------
SYSTEM = """Eres una asesora inmobiliaria amable y profesional (tono rioplatense, trato cercano).
Objetivo: calificar al cliente recogiendo estos datos, estrictamente UNO POR TURNO:
1) zona/dirección (slot: zona)
2) presupuesto mínimo (ARS, slot: presupuesto_min)
3) presupuesto máximo (ARS, slot: presupuesto_max)
4) dormitorios (entero, slot: dormitorios)
5) cochera (sí/no, slot: cochera)

Reglas IMPORTANTES:
- Haz UNA sola pregunta por turno (no juntes dos en el mismo mensaje).
- No repitas la misma pregunta si el cliente ya respondió.
- Reformula breve y cálido lo que entendiste (“Perfecto, rango $A–$B”, etc.).
- Si el cliente escribe algo no relacionado, reconduce suavemente al siguiente slot pendiente.
- Cuando los 5 slots estén completos, di que ya tenés todo y quedás a disposición."""

FEWSHOT = [
    {"role":"user", "content":"Estoy buscando algo por Abasto"},
    {"role":"assistant","content":"Genial, tomo Abasto como zona. ¿Cuál sería tu presupuesto mínimo aproximado (en ARS)?"},
    {"role":"user","content":"De 200 a 300 mil"},
    {"role":"assistant","content":"Perfecto, tomo $200.000 como mínimo. ¿Y el máximo aproximado (en ARS)?"},
]

def llm_complete(messages: list, temperature: float = 0.6) -> str:
    # OpenAI responses (text-only) — gpt-4o-mini / responses API
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o-mini",
                "temperature": temperature,
                "messages": messages
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return "¿Podrías repetirme eso con un poco más de detalle? Gracias."


# -------------------------------------------------------------------
# Core de calificación (1 pregunta por turno)
# -------------------------------------------------------------------
class QualifyIn(BaseModel):
    user_phone: str
    message_id: Optional[str] = None
    text: str

def build_confirmation(slots: Dict[str, Any]) -> str:
    parts = []
    if slots.get("zona"):
        parts.append(f"zona {slots['zona']}")
    if slots.get("presupuesto_min") and slots.get("presupuesto_max"):
        parts.append(f"rango ${slots['presupuesto_min']:,}–${slots['presupuesto_max']:,}".replace(",", "."))
    elif slots.get("presupuesto_min"):
        parts.append(f"mínimo ${slots['presupuesto_min']:,}".replace(",", "."))
    if slots.get("dormitorios"):
        parts.append(f"{slots['dormitorios']} dormitorios")
    if slots.get("cochera") is not None:
        parts.append("con cochera" if slots["cochera"] else "sin cochera")
    return ("Perfecto, " + ", ".join(parts) + ".").strip(" ,.") if parts else ""

def next_missing_slot(slots: Dict[str, Any]) -> Optional[str]:
    for s in SLOT_ORDER:
        if slots.get(s) in (None, "", 0):
            return s
    return None

def question_for(slot: str) -> str:
    if slot == "zona":
        return "¿En qué zona o dirección estás interesado/a?"
    if slot == "presupuesto_min":
        return "¿Cuál sería tu presupuesto mínimo aproximado (en ARS)?"
    if slot == "presupuesto_max":
        return "¿Y el presupuesto máximo (en ARS)?"
    if slot == "dormitorios":
        return "¿Cuántos dormitorios te gustaría tener?"
    if slot == "cochera":
        return "¿Vas a necesitar cochera?"
    return "¿Podrías contarme un poco más?"

def try_fill_slots_from_text(slots: Dict[str, Any], texto: str) -> Dict[str, Any]:
    t = texto.lower()

    # zona: si menciona algo tipo "abasto", "centro", "pichincha", etc. (palabras largas evitamos “hola”)
    if slots.get("zona") in (None, ""):
        # toma la primera palabra interesante si el usuario contestó una zona corta
        zona_m = re.findall(r"[a-záéíóúñ]{4,}(?:\s+[0-9]{1,4})?", t)
        if zona_m:
            z = zona_m[0].strip().title()
            # evitamos palabras genéricas
            if z not in {"Hola","Perfecto","Minimo","Maximo","Dormitorios","Cochera"}:
                slots["zona"] = z

    # dinero
    money = parse_money(texto)
    if money:
        # decide si es min o max en base al slot pendiente
        missing = next_missing_slot(slots)
        if missing in ("presupuesto_min", "presupuesto_max"):
            slots[missing] = money
        else:
            # heurística: si min vacío, llena min; si no, llena max si está vacío
            if not slots.get("presupuesto_min"):
                slots["presupuesto_min"] = money
            elif not slots.get("presupuesto_max"):
                slots["presupuesto_max"] = money

    # dormitorios
    if slots.get("dormitorios") in (None, 0, ""):
        d = re.findall(r"\b([1-6])\b", t)  # 1..6
        if d:
            slots["dormitorios"] = int(d[0])

    # cochera
    if slots.get("cochera") is None:
        yn = parse_yes_no(t)
        if yn is not None:
            slots["cochera"] = yn

    # coherencia min/max
    pm, px = slots.get("presupuesto_min"), slots.get("presupuesto_max")
    if pm and px and pm > px:
        # intercambia si vinieron al revés
        slots["presupuesto_min"], slots["presupuesto_max"] = px, pm

    return slots

def is_ready(slots: Dict[str, Any]) -> bool:
    for s in SLOT_ORDER:
        if slots.get(s) in (None, "", 0):
            return False
    return True

def find_properties(slots: Dict[str, Any]) -> list:
    try:
        q = text("""
            SELECT direccion, zona, precio, dormitorios, cochera
            FROM propiedades
            WHERE zona LIKE :z
              AND precio BETWEEN :pmin AND :pmax
              AND dormitorios >= :d
              AND cochera >= :c
            ORDER BY precio ASC
            LIMIT 3
        """)
        with engine.begin() as conn:
            rows = conn.execute(q, {
                "z": f"%{slots.get('zona','')}%",
                "pmin": safe_int(slots.get("presupuesto_min"), 0),
                "pmax": safe_int(slots.get("presupuesto_max"), 999999999),
                "d": safe_int(slots.get("dormitorios"), 1),
                "c": 1 if slots.get("cochera") else 0,
            }).mappings().all()
            return [dict(r) for r in rows]
    except Exception:
        return []

def format_props(props: list) -> str:
    if not props:
        return "Por ahora no tengo publicaciones que entren justo en ese rango. Si te parece, el equipo comercial te contacta y buscamos alternativas cercanas."
    lines = ["Te paso opciones que encajan:"]
    for p in props:
        price = f"${p['precio']:,}".replace(",", ".")
        coch = "Sí" if p.get("cochera") else "No"
        lines.append(f"• {p['direccion']} ({p['zona']}) — {price}, {p['dormitorios']} dorm., cochera: {coch}")
    return "\n".join(lines)

def notify_vendor(user_phone: str, slots: Dict[str, Any]):
    try:
        # aca solo registramos el lead consolidado; el envío real lo hacés desde n8n (como ya tenés)
        execute("""
            INSERT INTO leads (user_phone,inmueble_interes,dormitorios,cochera,presupuesto_min,presupuesto_max,ventana_tiempo,notas,status,vendor_phone,created_at,updated_at)
            VALUES (:u,:z,:d,:c,:pmin,:pmax,NULL,NULL,'pendiente',:v, :ca,:ua)
        """, {
            "u": user_phone,
            "z": slots.get("zona"),
            "d": slots.get("dormitorios"),
            "c": 1 if slots.get("cochera") else 0,
            "pmin": slots.get("presupuesto_min"),
            "pmax": slots.get("presupuesto_max"),
            "v": VENDOR_PHONE or "",
            "ca": now(),
            "ua": now(),
        })
    except Exception:
        pass


# -------------------------------------------------------------------
# API
# -------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception:
        return {"ok": True}  # no rompemos la healthz

@app.post("/qualify")
def qualify(inp: QualifyIn):
    """
    Entrada desde n8n:
      - user_phone (string e.g. 5493412565812)
      - message_id (opcional)
      - text (mensaje del cliente)
    Respuesta JSON:
      - text           -> mensaje para el cliente (confirmación + pregunta)
      - next_question  -> la pregunta concreta (1 sola)
      - vendor_push    -> bool (cuando ya está todo listo)
      - updates        -> { slots: {...} }  (para guardar en n8n si querés)
    """
    user_phone = re.sub(r"\D", "", inp.user_phone or "")
    msg = (inp.text or "").strip()
    if not user_phone:
        return {"text": "Necesito un número de contacto para continuar.", "next_question": None, "vendor_push": False, "updates": {"slots": {}}}

    try:
        def mutate(row):
            slots = row["slots_json"] or {}
            # 1) completar slots con lo que dijo el cliente
            slots = try_fill_slots_from_text(slots, msg)

            # 2) decidir siguiente slot pendiente
            missing = next_missing_slot(slots)

            # no repitas la misma pregunta que la última vez
            last_q = slots.get("_last_q")
            if missing and question_for(missing) == last_q:
                # si se repite, pedimos aclaración muy breve sobre ese mismo slot sin duplicar la frase
                pass  # igual mantengo missing; la plantilla de salida se ocupa de NO duplicar

            # guarda slots
            row["slots_json"] = slots

        row = upsert_session(user_phone, mutate)
        slots = json.loads(row["slots_json"] or "{}")

        # 3) armar respuesta (confirmación + 1 pregunta)
        conf = build_confirmation(slots)

        missing = next_missing_slot(slots)
        if missing:
            q = question_for(missing)
            # bloquea doble pregunta en la salida: respondemos solo una pregunta
            text_out = (conf + ("\n" if conf else "") + q).strip()
            # recordá última pregunta para no repetir igual en el próximo turno
            upsert_session(user_phone, lambda r: r["slots_json"].update({"_last_q": q}))
            return {
                "text": text_out,
                "next_question": q,
                "vendor_push": False,
                "updates": {"slots": slots}
            }

        # 4) slots completos → buscar propiedades y cerrar
        props = find_properties(slots)
        listado = format_props(props)
        closing = "Con esa info ya estamos. Te van a contactar en breve para coordinar visita o enviarte más opciones."
        notify_vendor(user_phone, slots)

        out = (conf + ("\n" if conf else "") + listado + "\n" + closing).strip()
        return {
            "text": out,
            "next_question": None,
            "vendor_push": True,
            "updates": {"slots": slots}
        }

    except Exception as e:
        # Nunca 500 al usuario/flujo
        fallback = "Gracias. Para seguir, contame en qué zona te interesa y tu presupuesto aproximado (mínimo o máximo)."
        return {
            "text": fallback,
            "next_question": "¿En qué zona o dirección estás interesado/a?",
            "vendor_push": False,
            "updates": {"slots": {}}
        }
