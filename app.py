import os, json, re, time
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# =============== CONFIG FASTAPI ===============
app = FastAPI(title="Real Estate Conversational Agent")

# =============== DB ===============
raw_url = os.getenv("DATABASE_URL", "")
if not raw_url:
    raise RuntimeError("Missing DATABASE_URL. In Railway set DATABASE_URL=${{ MySQL.MYSQL_URL }}")
if raw_url.startswith("mysql://"):
    raw_url = "mysql+pymysql://" + raw_url[len("mysql://"):]
engine = create_engine(raw_url, pool_pre_ping=True, pool_recycle=180)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS chat_session (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_phone VARCHAR(32) NOT NULL,
  last_message_id VARCHAR(64),
  slots_json JSON NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uq_chat_session_user (user_phone)
);

CREATE TABLE IF NOT EXISTS leads (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_phone VARCHAR(32) NOT NULL,
  nombre VARCHAR(120) NULL,
  inmueble_interes VARCHAR(255) NULL,
  dormitorios TINYINT NULL,
  cochera TINYINT NULL,
  presupuesto VARCHAR(60) NULL,
  ventana_tiempo VARCHAR(50) NULL,
  contacto VARCHAR(80) NULL,
  status ENUM('pendiente','precalificado','calificado') DEFAULT 'pendiente',
  vendor_phone VARCHAR(32) DEFAULT '5493412654593',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_leads_phone (user_phone)
);

CREATE TABLE IF NOT EXISTS propiedades (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  codigo VARCHAR(32) UNIQUE,
  direccion VARCHAR(255),
  zona VARCHAR(100),
  precio INT,
  dormitorios TINYINT,
  cochera TINYINT
);
"""
SEED_SQL = """
INSERT INTO propiedades (codigo, direccion, zona, precio, dormitorios, cochera) VALUES
('A101','San Luis 234','Centro',120000,2,1),
('B202','Mendoza 1500','Pichincha',145000,3,0),
('C303','Av. Pellegrini 800','Centro',200000,3,1)
ON DUPLICATE KEY UPDATE direccion=VALUES(direccion);
"""

def do_bootstrap():
    with engine.begin() as conn:
        for stmt in BOOTSTRAP_SQL.strip().split(";\n\n"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))
        cnt = conn.execute(text("SELECT COUNT(*) c FROM propiedades")).mappings().one()["c"]
        if cnt == 0:
            conn.execute(text(SEED_SQL))

@app.on_event("startup")
def startup():
    for i in range(1, 8):
        try:
            do_bootstrap(); print("✅ DB bootstrap ok"); return
        except Exception as e:
            print(f"⚠️ bootstrap attempt {i}: {e}")
            if i == 7: raise
            time.sleep(2*i)

# =============== E/S MODELOS ===============
class MsgIn(BaseModel):
    message_id: str
    user_phone: str
    text: str

class MsgOut(BaseModel):
    text: str
    next_question: Optional[str] = None
    vendor_push: bool = False
    updates: Dict[str, Any] = {}

# =============== REGLAS DEL AGENTE ===============
GREETING = re.compile(r"\b(hola|buenas|buen día|buen dia|hey|qué tal|que tal)\b", re.I)
REQUIRED_SLOTS = ["inmueble_interes", "dormitorios", "presupuesto", "ventana_tiempo", "contacto"]

# =============== LLM ===============
# Usamos la API oficial de OpenAI (model configurable por env).
# Podés cambiar OPENAI_MODEL si querés otro (p.ej. gpt-4o-mini).
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# fallback “cliente” súper simple usando requests si no querés instalar SDK
import requests

def openai_chat(messages: List[Dict[str, str]], max_tokens=300, temperature=0.6) -> str:
    """
    Llama a /v1/chat/completions (formato API OpenAI compatible).
    Si preferís el SDK oficial, podés cambiar por openai.ChatCompletion.create(...)
    """
    if not OPENAI_API_KEY:
        # Sin key: devolver mensaje estándar para no romper
        return "Hola 👋 ¿Qué zona o dirección te interesa? Así te ayudo a buscar."
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

def openai_extract_slots(user_text: str, history_summary: str = "") -> Dict[str, Any]:
    """
    Pedimos al LLM extraer estructura (slots) en JSON.
    """
    if not OPENAI_API_KEY:
        return {}
    system = (
        "Sos un extractor de datos para una inmobiliaria. "
        "Devolvé ÚNICAMENTE un JSON con hasta estas claves: "
        "inmueble_interes (string), dormitorios (number), cochera (boolean/null), "
        "presupuesto (string), ventana_tiempo (string), contacto (string). "
        "Si no hay dato, omití la clave."
    )
    user = f"Historial resumido (opcional): {history_summary}\n\nMensaje nuevo: {user_text}"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        # forzamos salida JSON
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 150
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    try:
        data = json.loads(txt)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}

# =============== UTILS DEL AGENTE ===============
def have_all_required(slots: Dict[str, Any]) -> bool:
    return all(bool(slots.get(k)) for k in REQUIRED_SLOTS)

def summarize_for_extractor(conv: List[Dict[str, str]], limit_chars=800) -> str:
    """
    Pequeño resumen “hard cut” para no mandar todo el historial al extractor.
    """
    s = []
    for m in conv[-8:]:
        role = "U:" if m["role"] == "user" else "A:"
        s.append(f"{role} {m['content']}")
    joined = "\n".join(s)
    return joined[-limit_chars:]

def load_props_for_context(db, user_text: str) -> str:
    """
    Trae 2–3 propiedades que parezcan relevantes según la zona/dirección mencionada,
    para que el agente pueda “conversar con conocimiento”.
    """
    lower = user_text.lower()
    looks_like_property = any(w in lower for w in ["calle","av.","avenida","zona","barrio","pellegrini","san luis","centro","pichincha"])
    if not looks_like_property:
        return ""
    res = db.execute(
        text("""
            SELECT direccion, zona, precio, dormitorios, cochera
            FROM propiedades
            WHERE zona LIKE :q OR direccion LIKE :q
            ORDER BY precio ASC LIMIT 3
        """),
        {"q": f"%{user_text}%"}
    ).mappings().all()
    if not res:
        return ""
    lines = []
    for r in res:
        linea = f"- {r['direccion']} ({r['zona']}): {r['dormitorios']} dorm, {'cochera' if r['cochera'] else 'sin cochera'}, ${r['precio']:,}".replace(",", ".")
        lines.append(linea)
    return "Propiedades que pueden interesarle:\n" + "\n".join(lines)

# =============== ENDPOINTS ===============
@app.get("/")
def root():
    return {"ok": True, "service": "agent-llm", "endpoints": ["/healthz", "/qualify", "/docs"]}

@app.get("/healthz")
def healthz():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

@app.post("/qualify", response_model=MsgOut)
def qualify(msg: MsgIn):
    """
    Entrada: {message_id, user_phone, text}
    Salida:  {text, next_question?, vendor_push, updates}
    """
    db = SessionLocal()
    try:
        # === cargar/crear sesión ===
        s = db.execute(text("SELECT * FROM chat_session WHERE user_phone=:p"), {"p": msg.user_phone}).mappings().first()
        if not s:
            db.execute(text("INSERT INTO chat_session (user_phone, slots_json) VALUES (:p, JSON_OBJECT())"), {"p": msg.user_phone})
            db.commit()
            s = {"user_phone": msg.user_phone, "slots_json": {}}

        # normalizar slots_json
        slots_raw = s.get("slots_json")
        slots: Dict[str, Any] = json.loads(slots_raw) if isinstance(slots_raw, str) else (slots_raw or {})
        conversation: List[Dict[str, str]] = slots.get("conversation", [])

        # === RESET cuando saluda (garantiza comenzar por NO) ===
        if GREETING.search(msg.text.lower()):
            conversation = []
            slots.update({k: None for k in REQUIRED_SLOTS})  # limpia claves
            slots["conversation"] = conversation
            db.execute(
                text("UPDATE chat_session SET slots_json=:sj, last_message_id=:mid, updated_at=NOW() WHERE user_phone=:p"),
                {"sj": json.dumps(slots), "mid": msg.message_id, "p": msg.user_phone}
            )
            # igual respondemos algo natural, no plantilla rígida
            props_ctx = load_props_for_context(db, msg.text)
            system = (
                "Sos un agente inmobiliario argentino, cordial y natural. "
                "Vas a empezar la conversación desde cero. "
                "Hacé UNA pregunta a la vez para entender qué busca."
            )
            user = f"El cliente saludó: {msg.text}\n{props_ctx}"
            ai_resp = openai_chat(
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                max_tokens=120
            )
            conversation.append({"role":"user","content": msg.text})
            conversation.append({"role":"assistant","content": ai_resp})
            slots["conversation"] = conversation
            db.execute(
                text("UPDATE chat_session SET slots_json=:sj, last_message_id=:mid, updated_at=NOW() WHERE user_phone=:p"),
                {"sj": json.dumps(slots), "mid": msg.message_id, "p": msg.user_phone}
            )
            db.commit()
            return MsgOut(text=ai_resp, next_question=None, vendor_push=False, updates={"slots": slots})

        # === CONTEXTO para el LLM ===
        props_ctx = load_props_for_context(db, msg.text)
        system_prompt = (
            "Sos un agente inmobiliario argentino, cálido y directo. "
            "Conversá de forma natural, sin checklist rígido. "
            "Tu objetivo es entender y completar: inmueble_interes (zona/dirección), dormitorios, presupuesto, "
            "ventana_tiempo, contacto. "
            "Hacé preguntas de a una, y confirmá brevemente lo entendido. "
            "Si el cliente ya dio todo, indicá que pasarás con la vendedora."
        )

        # actualizamos historial y pedimos respuesta
        conversation.append({"role": "user", "content": msg.text})
        ai_text = openai_chat(
            messages=[{"role":"system","content":system_prompt}] + conversation + (
                [{"role":"system","content":props_ctx}] if props_ctx else []
            ),
            max_tokens=220,
            temperature=0.7
        )
        conversation.append({"role":"assistant","content": ai_text})

        # === extracción de datos estructurados (slots) ===
        summary = summarize_for_extractor(conversation)
        extracted = openai_extract_slots(user_text=msg.text, history_summary=summary)

        # merge “inteligente” a slots
        def merge_slot(key, val):
            if val is None: return
            if key == "dormitorios":
                try:
                    slots[key] = int(val)
                    return
                except Exception:
                    pass
            slots[key] = val

        for k in ["inmueble_interes","dormitorios","cochera","presupuesto","ventana_tiempo","contacto","nombre"]:
            if k in extracted:
                merge_slot(k, extracted[k])

        slots["conversation"] = conversation

        # persistimos
        db.execute(
            text("UPDATE chat_session SET slots_json=:sj, last_message_id=:mid, updated_at=NOW() WHERE user_phone=:p"),
            {"sj": json.dumps(slots), "mid": msg.message_id, "p": msg.user_phone}
        )
        # upsert del lead con lo que tengamos
        db.execute(
            text("""
                INSERT INTO leads (user_phone, nombre, inmueble_interes, dormitorios, cochera, presupuesto, ventana_tiempo, contacto, status)
                VALUES (:p, :nombre, :ii, :dorm, :coch, :pres, :vent, :cont, 'precalificado')
                ON DUPLICATE KEY UPDATE
                  nombre=COALESCE(:nombre, nombre),
                  inmueble_interes=COALESCE(:ii, inmueble_interes),
                  dormitorios=COALESCE(:dorm, dormitorios),
                  cochera=COALESCE(:coch, cochera),
                  presupuesto=COALESCE(:pres, presupuesto),
                  ventana_tiempo=COALESCE(:vent, ventana_tiempo),
                  contacto=COALESCE(:cont, contacto),
                  updated_at=NOW()
            """),
            {
                "p": msg.user_phone,
                "nombre": slots.get("nombre"),
                "ii": slots.get("inmueble_interes"),
                "dorm": slots.get("dormitorios"),
                "coch": slots.get("cochera"),
                "pres": slots.get("presupuesto"),
                "vent": slots.get("ventana_tiempo"),
                "cont": slots.get("contacto"),
            }
        )
        db.commit()

        # ¿ya está todo?
        done = have_all_required(slots)
        if done:
            # cerramos y pasamos
            db.execute(
                text("UPDATE leads SET status='calificado', updated_at=NOW() WHERE user_phone=:p"),
                {"p": msg.user_phone}
            )
            db.commit()
            return MsgOut(
                text=ai_text if ai_text else "¡Perfecto! Con eso ya tengo todo. Le paso tu consulta a la vendedora y te escribe en breve.",
                next_question=None,
                vendor_push=True,
                updates={"status": "calificado", "slots": slots}
            )

        # si falta info, seguimos la charla con el texto del LLM
        return MsgOut(
            text=ai_text if ai_text else "Genial, te ayudo con eso.",
            next_question=None,  # el LLM ya formula la pregunta
            vendor_push=False,
            updates={"slots": slots}
        )
    finally:
        db.close()
