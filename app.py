import os, json, datetime as dt
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# --------- Config ----------
DATABASE_URL = os.getenv("DATABASE_URL")  # mysql+pymysql://...
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
VENDOR_PHONE   = os.getenv("VENDOR_PHONE", "5493412654593")  # número del vendedor

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(bind=engine)

app = FastAPI()


# --------- I/O ---------
class MsgIn(BaseModel):
    message_id: str
    user_phone: str
    text: str


class MsgOut(BaseModel):
    text: Optional[str]
    next_question: Optional[str]
    vendor_push: bool
    updates: Dict[str, Any] = {}


# --------- OpenAI robusto (nunca rompe) ---------
def openai_chat(messages: List[Dict[str, str]], max_tokens=300, temperature=0.6) -> str:
    """
    Llama a OpenAI de forma robusta. Si falla, devuelve un texto seguro.
    """
    try:
        if not OPENAI_API_KEY:
            return ("¡Hola! Para ayudarte mejor contame zona/dirección y un presupuesto estimado. "
                    "Después te pregunto dormitorios y cochera 😉")
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post("https://api.openai.com/v1/chat/completions",
                          headers=headers, json=payload, timeout=30)
        if not r.ok:
            print("openai_chat error", r.status_code, r.text[:300])
            if r.status_code in (401, 429, 500, 503):
                return ("Estoy con mucha demanda ahora mismo. ¿Podés contarme zona/dirección "
                        "y tu presupuesto aproximado? Con eso sigo 👌")
            r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("openai_chat exception:", repr(e))
        return ("Genial. Empecemos simple: decime la zona/dirección y un presupuesto estimado; "
                "luego te consulto dormitorios y cochera.")


def openai_extract_slots(user_text: str, history_summary: str = "") -> Dict[str, Any]:
    """
    Extrae slots como JSON. Nunca rompe: si falla devuelve {}.
    Claves esperadas: inmueble_interes (str), dormitorios (number), cochera (bool/null),
                      presupuesto (str), presupuesto_min (number), presupuesto_max (number),
                      ventana_tiempo (str), contacto (str), zona (str)
    """
    try:
        if not OPENAI_API_KEY:
            return {}
        system = (
            "Sos un extractor de datos para una inmobiliaria en español. "
            "Devolvé SOLO JSON con las posibles claves: "
            "zona (string), inmueble_interes (string), dormitorios (number), cochera (boolean/null), "
            "presupuesto (string), presupuesto_min (number), presupuesto_max (number), "
            'ventana_tiempo (string), contacto (string). '
            "Si algo no está, OMITÍ la clave. Nada de texto adicional."
        )
        user = f"Historial resumido: {history_summary}\n\nMensaje nuevo: {user_text}"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "max_tokens": 200
        }
        r = requests.post("https://api.openai.com/v1/chat/completions",
                          headers=headers, json=payload, timeout=20)
        if not r.ok:
            print("openai_extract_slots error", r.status_code, r.text[:300])
            return {}
        txt = r.json()["choices"][0]["message"]["content"]
        return json.loads(txt) if txt else {}
    except Exception as e:
        print("openai_extract_slots exception:", repr(e))
        return {}


# --------- DB helpers ---------
def get_or_create_session(db, user_phone: str) -> Dict[str, Any]:
    row = db.execute(
        text("SELECT * FROM chat_session WHERE user_phone=:u LIMIT 1"), {"u": user_phone}
    ).mappings().first()
    if row:
        return dict(row)
    now = dt.datetime.utcnow()
    db.execute(text("""
        INSERT INTO chat_session (user_phone, status, slots_json, created_at, updated_at)
        VALUES (:u, 'active', '{}', :c, :u2)
    """), {"u": user_phone, "c": now, "u2": now})
    db.commit()
    row = db.execute(
        text("SELECT * FROM chat_session WHERE user_phone=:u LIMIT 1"), {"u": user_phone}
    ).mappings().first()
    return dict(row)


def save_session(db, sess: Dict[str, Any], slots: Dict[str, Any], last_message_id: str):
    now = dt.datetime.utcnow()
    db.execute(text("""
        UPDATE chat_session
        SET slots_json=:s, last_message_id=:mid, updated_at=:u
        WHERE id=:id
    """), {
        "s": json.dumps(slots, ensure_ascii=False),
        "mid": last_message_id,
        "u": now,
        "id": sess["id"]
    })
    db.commit()


# --------- Búsqueda segura en propiedades ---------
def find_props_safe(filters: dict) -> List[Dict[str, Any]]:
    """
    Busca propiedades con filtros opcionales y jamás levanta excepción.
    Filtros: zona (str), dormitorios (int), cochera (bool), presupuesto_min/max (int).
    """
    sql = """
        SELECT codigo, direccion, zona, precio, dormitorios, cochera
        FROM propiedades
        WHERE 1=1
    """
    params: Dict[str, Any] = {}

    try:
        zona = (filters.get("zona") or "").strip()
        if zona:
            sql += " AND LOWER(zona) = LOWER(:zona)"
            params["zona"] = zona

        dorms = filters.get("dormitorios")
        if isinstance(dorms, (int, float)):
            sql += " AND dormitorios >= :dorms"
            params["dorms"] = int(dorms)

        cochera = filters.get("cochera")
        if cochera is True:
            sql += " AND cochera = 1"
        elif cochera is False:
            sql += " AND cochera = 0"

        pmin = filters.get("presupuesto_min")
        pmax = filters.get("presupuesto_max")
        if isinstance(pmin, (int, float)):
            sql += " AND precio >= :pmin"
            params["pmin"] = int(pmin)
        if isinstance(pmax, (int, float)):
            sql += " AND precio <= :pmax"
            params["pmax"] = int(pmax)

        sql += " ORDER BY precio ASC LIMIT 5"

        with engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
            return [dict(r) for r in rows]
    except Exception as e:
        print("find_props_safe exception:", repr(e), "SQL:", sql, "params:", params)
        return []


# --------- Lógica de “agente”: qué preguntar ahora ---------
def plan_next_question(slots: Dict[str, Any]) -> str:
    """
    Decide inteligentemente cuál es la próxima pregunta que pide información faltante.
    """
    if not slots.get("zona") and not slots.get("inmueble_interes"):
        return "¿En qué zona o dirección te gustaría? (por ejemplo: Centro, Pichincha, Abasto…)"
    if slots.get("dormitorios") in (None, "", 0):
        return "¿Cuántos dormitorios te sirven?"
    if slots.get("cochera") not in (True, False):
        return "¿Necesitás cochera?"
    if not slots.get("presupuesto") and not (slots.get("presupuesto_min") or slots.get("presupuesto_max")):
        return "¿Qué presupuesto estás manejando (aprox)?"
    if not slots.get("ventana_tiempo"):
        return "¿Para cuándo te gustaría mudarte o comprar?"
    if not slots.get("contacto"):
        return "¿Me pasás un contacto (mail o preferencia de horario) para que te llame un asesor?"
    return ""  # ya completo


def is_ready_for_vendor(slots: Dict[str, Any]) -> bool:
    need_zone = bool(slots.get("zona") or slots.get("inmueble_interes"))
    need_budget = bool(slots.get("presupuesto") or slots.get("presupuesto_min") or slots.get("presupuesto_max"))
    need_one_spec = bool(slots.get("dormitorios") or slots.get("cochera") in (True, False))
    return need_zone and need_budget and need_one_spec


# --------- Endpoints ---------
@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/qualify", response_model=MsgOut)
def qualify(msg: MsgIn):
    db = SessionLocal()
    try:
        # --- Sesión y slots ---
        sess = get_or_create_session(db, msg.user_phone)

        # normalizar slots_json
        slots_raw = sess.get("slots_json")
        if isinstance(slots_raw, (bytes, bytearray)):
            try:
                slots_raw = slots_raw.decode("utf-8", errors="ignore")
            except Exception:
                slots_raw = "{}"
        if isinstance(slots_raw, str):
            try:
                slots = json.loads(slots_raw) if slots_raw else {}
            except Exception:
                print("Warn: slots_json corrupto, reseteo")
                slots = {}
        elif isinstance(slots_raw, dict):
            slots = slots_raw
        else:
            slots = {}

        if not isinstance(slots, dict):
            slots = {}
        conversation = slots.get("conversation")
        if not isinstance(conversation, list):
            conversation = []
        slots["conversation"] = conversation

        user_text = (msg.text or "").strip()

        # --- Inicio de conversación / saludo inteligente ---
        if len(conversation) == 0:
            # saludo
            assistant_greet = ("¡Hola! ¿Cómo estás? ¿En qué puedo ayudarte hoy en tu búsqueda inmobiliaria?")
            conversation.append({"role": "user", "content": user_text}) if user_text else None
            conversation.append({"role": "assistant", "content": assistant_greet})
            slots["conversation"] = conversation
            save_session(db, sess, slots, msg.message_id)
            return MsgOut(
                text=assistant_greet,
                next_question="Contame zona/dirección y presupuesto estimado 🙂",
                vendor_push=False,
                updates={"slots": slots}
            )

        # --- sumar turno actual al historial ---
        conversation.append({"role": "user", "content": user_text})

        # --- 1) extraer slots del turno ---
        # resumen muy corto para orientar a la extracción
        hist_for_extractor = ""
        try:
            ultimos = [x for x in conversation[-6:]]
            hist_for_extractor = "\n".join([f"{x['role']}: {x['content']}" for x in ultimos])
        except Exception:
            pass

        new_slots = openai_extract_slots(user_text, hist_for_extractor)
        # merge de slots (sin sobreescribir con vacíos)
        for k, v in new_slots.items():
            if v is not None and v != "":
                slots[k] = v

        # --- 2) respuesta conversacional del agente ---
        # contexto para el agente
        facts = []
        if slots.get("zona"):               facts.append(f"zona: {slots['zona']}")
        if slots.get("inmueble_interes"):   facts.append(f"interés: {slots['inmueble_interes']}")
        if slots.get("dormitorios"):        facts.append(f"dormitorios: {slots['dormitorios']}")
        if slots.get("cochera") in (True, False):
            facts.append("cochera: sí" if slots["cochera"] else "cochera: no")
        if slots.get("presupuesto"):
            facts.append(f"presupuesto: {slots['presupuesto']}")
        if slots.get("presupuesto_min") or slots.get("presupuesto_max"):
            facts.append(f"rango presupuesto: {slots.get('presupuesto_min')}–{slots.get('presupuesto_max')}")
        if slots.get("ventana_tiempo"):    facts.append(f"ventana: {slots['ventana_tiempo']}")

        system = (
            "Sos un agente inmobiliario REAL (no chatbot). "
            "Conversá natural, breve y útil. Confirmá lo entendido y pedí lo que falta "
            "para poder ofrecer opciones. No inventes datos ni promesas."
        )
        user_prompt = (
            "Datos conocidos: " + (", ".join(facts) if facts else "ninguno") +
            ".\nMensaje del cliente: " + user_text
        )
        assistant_reply = openai_chat(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user_prompt}]
        )

        # --- 3) sugerir propiedades (NO rompe si falla) ---
        filters = {
            "zona": slots.get("zona") or slots.get("inmueble_interes") or None,
            "dormitorios": slots.get("dormitorios"),
            "cochera": slots.get("cochera"),
            "presupuesto_min": slots.get("presupuesto_min"),
            "presupuesto_max": slots.get("presupuesto_max"),
        }
        props = []
        if any(v not in (None, "", []) for v in filters.values()):
            props = find_props_safe(filters)

        if props:
            listado = "\n".join(
                f"- {p['direccion']} ({p['zona']}) • {p['dormitorios']}d • "
                f"{'cochera' if p['cochera'] else 'sin cochera'} • USD {p['precio']}"
                for p in props
            )
            assistant_reply = (assistant_reply.strip() +
                               "\n\nEstas opciones podrían interesarte:\n" + listado)

        # --- 4) decidir siguiente pregunta y si empujamos a vendedor ---
        next_q = plan_next_question(slots)
        push = is_ready_for_vendor(slots)

        # completar historial
        conversation.append({"role": "assistant", "content": assistant_reply})
        slots["conversation"] = conversation

        # --- 5) si está listo, snapshot a leads (no rompe si ya existe) ---
        if push:
            try:
                db.execute(text("""
                    INSERT INTO leads (
                        user_phone, inmueble_interes, dormitorios, cochera,
                        presupuesto, presupuesto_min, presupuesto_max,
                        ventana_tiempo, contacto, status, vendor_phone, created_at, updated_at
                    )
                    VALUES (
                        :user_phone, :interes, :dorms, :cochera,
                        :presupuesto, :pmin, :pmax,
                        :ventana, :contacto, 'pendiente', :vendor, :c, :u
                    )
                """), {
                    "user_phone": msg.user_phone,
                    "interes": slots.get("inmueble_interes") or slots.get("zona"),
                    "dorms": slots.get("dormitorios"),
                    "cochera": 1 if slots.get("cochera") is True else (0 if slots.get("cochera") is False else None),
                    "presupuesto": slots.get("presupuesto"),
                    "pmin": slots.get("presupuesto_min"),
                    "pmax": slots.get("presupuesto_max"),
                    "ventana": slots.get("ventana_tiempo"),
                    "contacto": slots.get("contacto"),
                    "vendor": VENDOR_PHONE,
                    "c": dt.datetime.utcnow(),
                    "u": dt.datetime.utcnow(),
                })
                db.commit()
            except Exception as e:
                print("insert lead exception:", repr(e))

        # --- guardar sesión y responder ---
        save_session(db, sess, slots, msg.message_id)

        return MsgOut(
            text=assistant_reply,
            next_question=(next_q or None),
            vendor_push=push,
            updates={"slots": slots}
        )

    except Exception as e:
        print("qualify exception:", repr(e))
        # degradación amable
        return MsgOut(
            text=("Gracias. Para avanzar, contame la zona/dirección y un presupuesto estimado; "
                  "después te consulto dormitorios y cochera."),
            next_question="¿En qué zona o dirección te gustaría? 🙂",
            vendor_push=False,
            updates={}
        )
    finally:
        db.close()
