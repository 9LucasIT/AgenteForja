from fastapi import FastAPI, Request
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import random
import openai
import os

# Inicialización
app = FastAPI()

# Config DB Railway
DATABASE_URL = os.getenv("MYSQL_URL") or os.getenv("DATABASE_URL") or "mysql+pymysql://root:password@localhost/dbname"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Config OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Modelo BD
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


# Modelo Pydantic
class MessagePayload(BaseModel):
    message_id: str
    user_phone: str
    text: str


# --- Función auxiliar ---
def get_or_create_session(db, phone):
    session = db.query(ChatSession).filter(ChatSession.user_phone == phone).first()
    if not session:
        session = ChatSession(
            user_phone=phone,
            conversation="",
            stage="inicio",
            vendor_push=False,
            guard_already_sent=False
        )
        db.add(session)
        db.commit()
        db.refresh(session)
    return session


# --- Motor de conversación ---
def generar_respuesta(session, mensaje):
    texto = mensaje.strip().lower()
    conversacion = session.conversation or ""

    # Reinicio manual
    if texto == "reset":
        session.stage = "inicio"
        session.conversation = ""
        session.vendor_push = False
        session.guard_already_sent = False
        return (
            "¡Arranquemos de nuevo! 😊\n"
            "Contame: ¿la búsqueda es para **alquiler** o para **venta**?\n"
            "Tip: cuando quieras reiniciar la conversación, escribí *reset* y empezamos de cero. 😉",
            False
        )

    # Etapas de la conversación
    if session.stage == "inicio":
        session.stage = "tipo_operacion"
        return "¿La búsqueda es para **alquiler** o para **venta**?", False

    elif session.stage == "tipo_operacion":
        if "alquiler" in texto:
            tipo = "alquiler"
        elif "venta" in texto:
            tipo = "venta"
        else:
            return "¿Podrías confirmarme si es para **alquiler** o para **venta**?", False
        session.conversation += f"Operación: {tipo}\n"
        session.stage = "zona"
        return "¿En qué zona o dirección exacta estás interesado? (calle y número si lo tenés)", False

    elif session.stage == "zona":
        session.conversation += f"Zona: {mensaje}\n"
        session.stage = "presupuesto_min"
        return "¿Cuál sería tu presupuesto *mínimo* aproximado (en ARS)?", False

    elif session.stage == "presupuesto_min":
        session.conversation += f"Presupuesto mínimo: {mensaje}\n"
        session.stage = "presupuesto_max"
        return "¿Y el presupuesto *máximo* (en ARS)?", False

    elif session.stage == "presupuesto_max":
        session.conversation += f"Presupuesto máximo: {mensaje}\n"
        session.stage = "dormitorios"
        return "¿Cuántos dormitorios te gustaría tener en la propiedad?", False

    elif session.stage == "dormitorios":
        session.conversation += f"Dormitorios: {mensaje}\n"
        session.stage = "cochera"
        return "¿Vas a necesitar cochera?", False

    elif session.stage == "cochera":
        session.conversation += f"Cochera: {mensaje}\n"
        session.stage = "mascotas"
        return "¿Tenés mascotas que debamos contemplar?", False

    elif session.stage == "mascotas":
        session.conversation += f"Mascotas: {mensaje}\n"
        session.stage = "direccion"
        return "¿Tenés una dirección exacta? (calle y número si lo sabés)", False

    elif session.stage == "direccion":
        session.conversation += f"Dirección: {mensaje}\n"
        session.stage = "final"
        session.vendor_push = True
        resumen = session.conversation.replace("\n", " | ")
        return (
            f"Perfecto 👍 Ya tengo todo.\nTe resumo lo que me contaste:\n{resumen}\n"
            "En breve, un asesor te contactará con las mejores opciones. 😊",
            True
        )

    else:
        return "Podés escribirme *reset* para comenzar una nueva búsqueda. 😉", False


# --- Endpoint principal ---
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

        # Humanizar respuestas finales
        if not push_vendedor:
            saludo = random.choice([
                "Genial, gracias por contarme eso. ",
                "Perfecto, te entiendo. ",
                "Buenísimo, avancemos. ",
                "Excelente, seguimos. "
            ])
            texto = saludo + respuesta
        else:
            texto = respuesta

        return {
            "text": texto,
            "next_question": None,
            "vendor_push": push_vendedor,
            "conversation": session.conversation
        }

    except Exception as e:
        db.rollback()
        return {"error": str(e)}
    finally:
        db.close()


@app.get("/healthz")
async def health():
    return {"ok": True}
