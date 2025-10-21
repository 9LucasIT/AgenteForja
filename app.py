import os, json, re, time
from datetime import datetime
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

app = FastAPI(title="Real Estate Lead Qualifier")

# === CONFIG ===
raw_url = os.getenv("DATABASE_URL", "")
if not raw_url:
    raise RuntimeError("Missing DATABASE_URL. In Railway set DATABASE_URL=${{ MySQL.MYSQL_URL }}")

# Railway entrega "mysql://..." → necesitamos "mysql+pymysql://..."
if raw_url.startswith("mysql://"):
    raw_url = "mysql+pymysql://" + raw_url[len("mysql://"):]

# Conexión robusta para evitar caídas
engine = create_engine(
    raw_url,
    pool_pre_ping=True,   # verifica conexión antes de usarla
    pool_recycle=180,     # evita "MySQL server has gone away"
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# === Bootstrap DB ===
BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS chat_session (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_phone VARCHAR(32) NOT NULL,
  last_message_id VARCHAR(64),
  last_welcome_at DATETIME NULL,
  cooldown_until DATETIME NULL,
  status ENUM('active','paused','human') DEFAULT 'active',
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
  presupuesto_min INT NULL,
  presupuesto_max INT NULL,
  ventana_tiempo VARCHAR(50) NULL,
  notas TEXT NULL,
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
INSERT INTO propiedades (codigo, direccion, zona, precio, dormitorios, cochera)
VALUES
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
        cnt = conn.execute(text("SELECT COUNT(*) AS c FROM propiedades")).mappings().one()["c"]
        if cnt == 0:
            conn.execute(text(SEED_SQL))

# Bootstrap con reintentos (la DB puede demorar)
@app.on_event("startup")
def startup():
    for attempt in range(1, 8):
        try:
            do_bootstrap()
            print("✅ Database bootstrap done")
            return
        except Exception as e:
            print(f"⚠️ Bootstrap error (attempt {attempt}): {e}")
            if attempt == 7:
                raise
            time.sleep(2 * attempt)

# === MODELOS ===
class MsgIn(BaseModel):
    message_id: str
    user_phone: str
    text: str

class MsgOut(BaseModel):
    text: str
    next_question: Optional[str] = None
    vendor_push: bool = False
    updates: dict = {}

# === LÓGICA DEL AGENTE ===
GREETING = re.compile(r"\b(hola|buenas|hey)\b", re.I)
MONEY = re.compile(r"(\d[\d\.]{3,})")
ORDER = ["inmueble_interes","dormitorios","presupuesto","ventana_tiempo","contacto"]

def extract_signals(text_in: str):
    sig = {}
    m = MONEY.search(text_in)
    if m:
        val = int(m.group(1).replace(".", ""))
        sig["presupuesto_min"] = val
    return sig

def next_missing(slots: dict):
    for s in ORDER:
        if not slots.get(s):
            return s
    return None

def build_question(slot: str) -> str:
    preguntas = {
        "inmueble_interes": "¿Por cuál propiedad querés consultar? Podés decirme una dirección o zona.",
        "dormitorios": "¿Cuántos dormitorios te vienen bien?",
        "presupuesto": "¿Tenés un presupuesto aproximado o un rango?",
        "ventana_tiempo": "¿Para cuándo te gustaría mudarte (urgente, 1–3 meses, 6+ meses)?",
        "contacto": "¿Preferís que te contacten por acá o por llamada? ¿En qué horario?"
    }
    return preguntas[slot]

@app.get("/healthz")
def healthz():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

@app.post("/qualify", response_model=MsgOut)
def qualify(msg: MsgIn):
    db = SessionLocal()
    try:
        s = db.execute(
            text("SELECT * FROM chat_session WHERE user_phone=:p"),
            {"p": msg.user_phone}
        ).mappings().first()
        if not s:
            db.execute(
                text("INSERT INTO chat_session (user_phone, slots_json) VALUES (:p, JSON_OBJECT())"),
                {"p": msg.user_phone}
            )
            db.commit()
            s = {"user_phone": msg.user_phone, "slots_json": {}}

        if s.get("last_message_id") == msg.message_id:
            return MsgOut(text="(ya procesado)")

        slots = s.get("slots_json") or {}
        txt = msg.text.strip()
        sig = extract_signals(txt)
        if "presupuesto_min" in sig and not slots.get("presupuesto"):
            slots["presupuesto"] = f"{sig['presupuesto_min']}+"

        sugerencia_txt = ""
        lower = txt.lower()
        looks_like_property = any(w in lower for w in ["calle","av.","avenida","zona","barrio","pellegrini","san luis","centro"])
        if looks_like_property:
            res = db.execute(
                text(
                    """
                    SELECT direccion, zona, precio, dormitorios, cochera
                    FROM propiedades
                    WHERE zona LIKE :q OR direccion LIKE :q
                    ORDER BY precio ASC LIMIT 3
                    """
                ),
                {"q": f"%{txt}%"}
            ).mappings().all()
            if res:
                slots["inmueble_interes"] = res[0]["direccion"]
                sug = []
                for r in res:
                    sug.append(f"• {r['direccion']} ({r['zona']}) — {r['dormitorios']} dorm, {'cochera' if r['cochera'] else 'sin cochera'}, ${r['precio']:,}".replace(",", "."))
                sugerencia_txt = "Mirá, tengo estas que encajan:\n" + "\n".join(sug)

        missing = next_missing(slots)
        if not missing:
            db.execute(
                text("INSERT INTO leads (user_phone, status) VALUES (:p, 'calificado') ON DUPLICATE KEY UPDATE status='calificado', updated_at=NOW()"),
                {"p": msg.user_phone}
            )
            db.execute(
                text("UPDATE chat_session SET slots_json=:sj, last_message_id=:mid, updated_at=NOW() WHERE user_phone=:p"),
                {"sj": json.dumps(slots), "mid": msg.message_id, "p": msg.user_phone}
            )
            db.commit()
            return MsgOut(
                text=(sugerencia_txt + "\n" if sugerencia_txt else "") + "¡Perfecto! Con eso ya tengo todo. Le paso tu consulta a la vendedora y te escribe en breve.",
                vendor_push=True,
                updates={"status": "calificado", "slots": slots}
            )

        question = build_question(missing)
        body = (sugerencia_txt + "\n" if sugerencia_txt else "") + "Genial, te ayudo con eso."

        db.execute(
            text(
                """
                INSERT INTO leads (user_phone, inmueble_interes, presupuesto_min)
                VALUES (:p, :ii, :pm)
                ON DUPLICATE KEY UPDATE
                  inmueble_interes=COALESCE(:ii, inmueble_interes),
                  presupuesto_min=COALESCE(:pm, presupuesto_min),
                  updated_at=NOW()
                """
            ),
            {"p": msg.user_phone, "ii": slots.get("inmueble_interes"), "pm": sig.get("presupuesto_min")}
        )
        db.execute(
            text("UPDATE chat_session SET slots_json=:sj, last_message_id=:mid, updated_at=NOW() WHERE user_phone=:p"),
            {"sj": json.dumps(slots), "mid": msg.message_id, "p": msg.user_phone}
        )
        db.commit()
        return MsgOut(text=body, next_question=question, vendor_push=False, updates={"slots": slots})
    finally:
        db.close()
