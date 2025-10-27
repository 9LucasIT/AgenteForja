from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import random
import os


import re
import unicodedata

def _norm(txt: str) -> str:
    if not txt:
        return ""
    # minúsculas + sin acentos
    txt = unicodedata.normalize("NFD", txt).encode("ascii", "ignore").decode("utf-8")
    return re.sub(r"\s+", " ", txt.lower()).strip()

def detect_operacion(txt: str) -> str | None:
    t = _norm(txt)
    rent_keys = (
        "alquiler", "alquilo", "alquilar", "renta", "rent", "en alquiler"
    )
    sell_keys = (
        "venta", "vendo", "vender", "comprar", "compro", "en venta"
    )
    if any(k in t for k in rent_keys):
        return "alquiler"
    if any(k in t for k in sell_keys):
        return "venta"
    return None


# === CONFIGURACIÓN BASE ===

app = FastAPI()

# URL de tu base de datos Railway
DATABASE_URL = (
    os.getenv("MYSQL_URL")
    or os.getenv("DATABASE_URL")
    or "mysql+pymysql://root:password@localhost/dbname"
)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# === MODELO DE TABLA chat_session ===
class ChatSession(Base):
    __tablename__ = "chat_session"
    id = Column(Integer, primary_key=True, index=True)
    user_phone = Column(String(50), unique=True)
    conversation = Column(Text)
    stage = Column(String(50))
    vendor_push = Column(Boolean, default=False)
    guard_already_sent = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# === MODELO ENTRANTE DE MENSAJE ===
class MessagePayload(BaseModel):
    message_id: str | None = None
    user_phone: str
    text: str

# === FUNCIONES AUXILIARES ===
def get_or_create_session(db, phone):
    session = db.query(ChatSession).filter(ChatSession.user_phone == phone).first()
    if not session:
        session = ChatSession(
            user_phone=phone,
            conversation="",
            stage="inicio",
            vendor_push=False,
            guard_already_sent=False,
        )
        db.add(session)
        db.commit()
        db.refresh(session)
    return session


def generar_respuesta(session, mensaje):
    texto = mensaje.strip().lower()

    # Reinicio
    if texto == "reset":
        session.stage = "inicio"
        session.conversation = ""
        session.vendor_push = False
        session.guard_already_sent = False
        return (
            "Empecemos desde cero 😊\n¿La búsqueda es para *alquiler* o *venta*?",
            False,
        )

    # === ETAPAS DE CONVERSACIÓN ===
    if session.stage == "inicio":
        session.stage = "tipo_operacion"
        return "¿La búsqueda es para *alquiler* o *venta*?", False

    elif session.stage == "tipo_operacion":
        if "alquiler" in texto:
            session.stage = "zona"
            session.conversation += "Operación: Alquiler\n"
            return "Perfecto 👍 ¿En qué zona o barrio te interesa buscar?", False
        elif "venta" in texto:
            session.stage = "zona"
            session.conversation += "Operación: Venta\n"
            return "Excelente 💰 ¿En qué zona o barrio te interesa?", False
        else:
            return "¿Podrías confirmarme si buscás *alquiler* o *venta*?", False

    elif session.stage == "zona":
        session.conversation += f"Zona: {mensaje}\n"
        session.stage = "presupuesto"
        return "¿Cuál es tu presupuesto aproximado (en pesos)?", False

    elif session.stage == "presupuesto":
        session.conversation += f"Presupuesto: {mensaje}\n"
        session.stage = "dormitorios"
        return "¿Cuántos dormitorios necesitás?", False

    elif session.stage == "dormitorios":
        session.conversation += f"Dormitorios: {mensaje}\n"
        session.stage = "mascotas"
        return "¿Tenés mascotas?", False

    elif session.stage == "mascotas":
        session.conversation += f"Mascotas: {mensaje}\n"
        session.stage = "final"
        session.vendor_push = True
        resumen = session.conversation.replace("\n", " | ")
        return (
            f"Perfecto, con eso ya tengo todo 😄.\n"
            f"Te resumo lo que me contaste:\n{resumen}\n"
            f"En unos minutos un asesor se pondrá en contacto contigo.",
            True,
        )

    else:
        return "Podés escribir *reset* para comenzar de nuevo. 😉", False


# === ENDPOINT PRINCIPAL ===
@app.post("/qualify")
async def qualify(payload: MessagePayload):
    db = SessionLocal()
    try:
        session = get_or_create_session(db, payload.user_phone)
        respuesta, push_vendedor = generar_respuesta(session, payload.text)

        # Guardar conversación
        session.conversation += f"\nCliente: {payload.text}\nAgente: {respuesta}\n"
        session.vendor_push = push_vendedor
        db.commit()

        # Tono humano (varía un poco las frases)
        if not push_vendedor:
            saludo = random.choice(
                [
                    "Genial, gracias por contarme eso. ",
                    "Perfecto, te entiendo. ",
                    "Buenísimo, avancemos. ",
                    "Excelente, seguimos. ",
                ]
            )
            texto = saludo + respuesta
        else:
            texto = respuesta

        return {
            "text": texto,
            "next_question": None,
            "vendor_push": push_vendedor,
            "conversation": session.conversation,
        }

    except Exception as e:
        db.rollback()
        return {"error": str(e)}
    finally:
        db.close()


@app.get("/healthz")
async def healthz():
    return {"ok": True}
