import os
import re
import time
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# ======== LLM opcional ==========
USE_LLM = bool(os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if USE_LLM:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        USE_LLM = False

# ======== DB (SQLAlchemy) ==========
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

def make_engine() -> Optional[Engine]:
    url = os.getenv("DATABASE_URL", "")
    if not url:
        host = os.getenv("MYSQL_HOST", "")
        user = os.getenv("MYSQL_USER", "")
        pw = os.getenv("MYSQL_PASSWORD", "")
        db = os.getenv("MYSQL_DATABASE", "")
        if host and user and db:
            url = f"mysql+pymysql://{user}:{pw}@{host}:3306/{db}"
    if not url:
        return None
    try:
        return create_engine(url, pool_pre_ping=True)
    except Exception as e:
        log.exception("No se pudo crear el engine de DB")
        return None

# ======== Logging ==========
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fastapi-agent")

# ======== FastAPI ==========
app = FastAPI(title="Veglienzone Agent", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

DB: Optional[Engine] = make_engine()

# ========= Memoria y dedupe =========
# estado por chatId
SESSIONS: Dict[str, Dict[str, Any]] = {}
# idMessage ya procesados (TTL simple)
PROCESSED: Dict[str, float] = {}
DEDUPE_TTL_SEC = 300

def dedupe(id_message: Optional[str]) -> bool:
    """True si este id ya fue procesado recientemente."""
    now = time.time()
    # limpia viejos
    for k in list(PROCESSED.keys()):
        if now - PROCESSED[k] > DEDUPE_TTL_SEC:
            del PROCESSED[k]
    if not id_message:
        return False
    if id_message in PROCESSED:
        return True
    PROCESSED[id_message] = now
    return False

# ========= Utilidades =========
URL_REGEX = re.compile(r"(https?://\S+)", re.IGNORECASE)

def has_url(text: str) -> bool:
    return bool(URL_REGEX.search(text or ""))

def looks_like_address(text: str) -> bool:
    # heurística simple: calle + número
    return bool(re.search(r"[a-zA-Záéíóúüñ]{3,}\s+\d{1,5}", (text or ""), re.IGNORECASE))

def to_money(v: Optional[float]) -> str:
    if v is None: return "N/D"
    try:
        return "${:,.0f}".format(float(v)).replace(",", ".")
    except Exception:
        return str(v)

def ensure_session(chat_id: str) -> Dict[str, Any]:
    if chat_id not in SESSIONS:
        SESSIONS[chat_id] = {
            "phase": "start",
            "operation": None,  # 'alquiler' | 'venta'
            "zone": None,
            "address_or_link": None,
            "tipo": None,       # departamento, casa, etc.
            "dorms": None,
            "budget": None,
            "mascotas": None,
            "habitantes": None,
            "ingresos": None,
            "garantia": None,   # 'finaer' | 'propietaria' | 'otro'
            "cochera": None,
            "interested_codes": [],
            "asked_reset_note": False
        }
    return SESSIONS[chat_id]

def reset_session(chat_id: str):
    if chat_id in SESSIONS:
        del SESSIONS[chat_id]

# ========= LLM helpers =========
def llm_classify_intent(texto: str) -> Dict[str, Any]:
    """
    Devuelve dict con:
      operation: 'alquiler'|'venta'|None
      zone: str|None
      tipo: str|None
      address_or_link: str|None
    """
    if not USE_LLM:
        # reglas simples de respaldo
        t = (texto or "").lower()
        op = "alquiler" if "alquil" in t else ("venta" if "venta" in t or "compro" in t or "comprar" in t else None)
        address = texto if (has_url(texto) or looks_like_address(texto)) else None
        zone = None
        # busco palabras indicativas de zona genérica
        zcands = ["centro", "macrocentro", "abasto", "fisherton", "echesortu", "pichincha", "arroyito", "recoleta", "palermo", "belgrano"]
        for z in zcands:
            if z in t:
                zone = z
                break
        tipo = "departamento" if ("depto" in t or "depart" in t) else ("casa" if "casa" in t else None)
        return {"operation": op, "zone": zone, "tipo": tipo, "address_or_link": address}

    prompt = (
        "Extrae del mensaje de un cliente inmobiliario los siguientes campos en JSON:"
        "operation: 'alquiler' | 'venta' | null;"
        "zone: string o null;"
        "tipo: 'departamento'|'casa'|null;"
        "address_or_link: si el texto contiene link o dirección exacta, colócalo; sino null."
        f" Mensaje: ```{texto}```"
    )
    try:
        comp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content":"Eres un analista de intents inmobiliarios."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        raw = comp.choices[0].message.content.strip()
        # intenta parsear
        data = json.loads(raw) if raw.startswith("{") else {}
        # sanea
        for k in ["operation","zone","tipo","address_or_link"]:
            if k not in data: data[k] = None
        return data
    except Exception:
        return {"operation": None, "zone": None, "tipo": None, "address_or_link": None}

# ========= DB search =========
def _first_ok_query(conn, sqls: List[str], params: Dict[str, Any]):
    """Prueba varias SQL y retorna la primera que no falle."""
    last_err = None
    for s in sqls:
        try:
            return conn.execute(text(s), params).fetchmany(3)
        except Exception as e:
            last_err = e
            continue
    if last_err:
        log.warning(f"No pudo ejecutarse ninguna SQL de propiedades: {last_err}")
    return []

def find_properties(zone: Optional[str],
                    operation: Optional[str],
                    tipo: Optional[str]) -> List[Dict[str, Any]]:
    """
    Devuelve hasta 3 propiedades dictadas por zona/operación/tipo.
    No rompe si cambian nombres de columnas más comunes.
    """
    if DB is None:
        return []

    zone_like = f"%{zone}%" if zone else "%"
    op_like = (operation or "%")
    tipo_like = f"%{tipo}%" if tipo else "%"

    sql_candidates = [
        # zona/barrio + operacion + tipo
        """
        SELECT 
            COALESCE(codigo, id, id_propiedad)     AS code,
            COALESCE(direccion, address)           AS address,
            COALESCE(zona, barrio, localidad)      AS zone,
            COALESCE(precio, precio_base)          AS price,
            COALESCE(dormitorios, ambientes)       AS dorms,
            COALESCE(cochera, cocheras, parking)   AS parking
        FROM propiedad
        WHERE (COALESCE(zona, barrio, localidad) LIKE :zone_like)
          AND (LOWER(COALESCE(operacion, estado)) LIKE :op_like)
          AND (LOWER(COALESCE(tipo, tipoinmueble)) LIKE :tipo_like)
        ORDER BY price ASC
        LIMIT 3
        """,
        # sin tipo
        """
        SELECT 
            COALESCE(codigo, id, id_propiedad)     AS code,
            COALESCE(direccion, address)           AS address,
            COALESCE(zona, barrio, localidad)      AS zone,
            COALESCE(precio, precio_base)          AS price,
            COALESCE(dormitorios, ambientes)       AS dorms,
            COALESCE(cochera, cocheras, parking)   AS parking
        FROM propiedad
        WHERE (COALESCE(zona, barrio, localidad) LIKE :zone_like)
          AND (LOWER(COALESCE(operacion, estado)) LIKE :op_like)
        ORDER BY price ASC
        LIMIT 3
        """,
    ]

    try:
        with DB.connect() as conn:
            rows = _first_ok_query(conn, sql_candidates, {
                "zone_like": zone_like,
                "op_like": f"%{op_like.lower()}%",
                "tipo_like": tipo_like.lower()
            })
        props = []
        for r in rows:
            d = dict(r._mapping)
            props.append({
                "code": d.get("code"),
                "address": d.get("address"),
                "zone": d.get("zone"),
                "price": d.get("price"),
                "dorms": d.get("dorms"),
                "parking": d.get("parking"),
            })
        return props
    except Exception as e:
        log.exception("Error buscando propiedades")
        return []

def format_properties(props: List[Dict[str, Any]]) -> str:
    if not props:
        return "No tengo publicaciones para mostrarte en esa zona ahora mismo. ¿Querés que te avise cuando ingrese algo?"
    lines = ["Te dejo algunas opciones en la zona:"]
    for p in props:
        line = f"- Código: {p.get('code','N/D')} | Dirección: {p.get('address','N/D')} | Dormitorios: {p.get('dorms','N/D')} | Precio: {to_money(p.get('price'))}"
        if p.get("parking"):
            line += " | cochera"
        lines.append(line)
    lines.append("")
    lines.append("Si te interesa alguna, decime el **código** y sigo con el detalle. Si preferís ajustar filtros, contame.")
    return "\n".join(lines)

# ========= Mensajería principal (/qualify) =========
@app.post("/qualify")
async def qualify(payload: Dict[str, Any]):
    """
    Entrada desde n8n (usás el nodo HTTP → FastAPI /qualify).
    Espera: { chatId, text, idMessage, user_phone, ... }
    Devuelve:
    {
      "reply_text": "...",
      "closing_text": "",
      "vendor_push": false,
      "vendor_message": ""
    }
    """
    chat_id = payload.get("chatId")
    raw_text = (payload.get("text") or "").strip()
    id_msg = payload.get("idMessage")

    # dedupe
    if dedupe(id_msg):
        return {"reply_text": "", "closing_text": "", "vendor_push": False, "vendor_message": ""}

    if not chat_id:
        raise HTTPException(400, "chatId requerido")

    # RESET
    if raw_text.lower() == "reset":
        reset_session(chat_id)
        return {
            "reply_text": (
                "Listo, reinicié la conversación desde cero. 👌\n\n"
                "Gracias por contactarte con el área comercial de Veglienzone Gestión Inmobiliaria.\n"
                "¿Cómo podemos ayudarte hoy?\n"
                "1- Alquileres\n2- Ventas\n3- Tasaciones\n\n"
                "_Nota: en cualquier momento escribí **reset** para reiniciar._"
            ),
            "closing_text": "",
            "vendor_push": False,
            "vendor_message": ""
        }

    ss = ensure_session(chat_id)

    # Saludo inicial / menú
    if ss["phase"] == "start":
        # intenta entender intención primaria
        intent = llm_classify_intent(raw_text)
        if intent.get("operation"):
            ss["operation"] = intent["operation"]
            ss["zone"] = intent.get("zone")
            ss["tipo"] = intent.get("tipo")
            ss["address_or_link"] = intent.get("address_or_link")
            ss["phase"] = "after_op"
        else:
            # si responde 1/2/3
            lower = raw_text.lower()
            if lower in ["1","alquiler","alquilo","quiero alquilar","me gustaria alquilar"]:
                ss["operation"] = "alquiler"
                ss["phase"] = "after_op"
            elif lower in ["2","venta","compro","quiero comprar","me gustaria comprar"]:
                ss["operation"] = "venta"
                ss["phase"] = "after_op"

        if ss["phase"] == "start":
            ss["phase"] = "start_menu"
            return {
                "reply_text": (
                    "Gracias por contactarte con el área comercial de Veglienzone Gestión Inmobiliaria. "
                    "¿Cómo podemos ayudarte hoy?\n"
                    "1- Alquileres\n2- Ventas\n3- Tasaciones\n\n"
                    "_Nota: en cualquier momento escribí **reset** para reiniciar._"
                ),
                "closing_text": "",
                "vendor_push": False,
                "vendor_message": ""
            }

    # Primera pregunta clave tras elegir operación
    if ss["phase"] in ("after_op","start_menu"):
        ss["phase"] = "await_addr_or_zone"
        return {
            "reply_text": (
                "¿Tenés una **dirección exacta o link** de la propiedad que viste, o preferís que busquemos por **zona**?\n"
                "Podés responder, por ejemplo: *San Luis 234* o *Centro*."
            ),
            "closing_text": "",
            "vendor_push": False,
            "vendor_message": ""
        }

    # Respuesta a exacta/zonas
    if ss["phase"] == "await_addr_or_zone":
        # si vino dirección o link
        if has_url(raw_text) or looks_like_address(raw_text):
            ss["address_or_link"] = raw_text
            ss["phase"] = "qualify_specific"
            # acá podrías buscar por dirección/código y responder ficha concreta
            return {
                "reply_text": (
                    "¡Perfecto! Si es esa unidad específica, ¿qué datos te interesan saber? "
                    "Puedo contarte precio, dorms, expensas, si acepta mascotas, etc. "
                    "Si querés también te puedo proponer opciones parecidas."
                ),
                "closing_text": "",
                "vendor_push": False,
                "vendor_message": ""
            }
        # si vino zona
        intent = llm_classify_intent(raw_text)
        if intent.get("zone"):
            ss["zone"] = intent["zone"]
        else:
            # usa tal cual lo que dijo como zona
            ss["zone"] = raw_text

        # ofrecer opciones si es alquiler y tipo no especificado, asumir dpto
        if ss["operation"] == "alquiler" and not ss["tipo"]:
            ss["tipo"] = "departamento"

        props = find_properties(zone=ss["zone"], operation=ss["operation"], tipo=ss["tipo"])
        ss["phase"] = "qualify_generic"
        reply = format_properties(props)
        reply += "\n\nPara entender mejor tu necesidad:\n"
        if ss["operation"] == "alquiler":
            reply += "¿Tenés **presupuesto** aproximado y **cantidad de dormitorios**? ¿Y vas a necesitar cochera?"
        else:
            reply += "¿Qué **presupuesto** tenés estimado y **cantidad de dormitorios**? ¿Querés que te envíe opciones que coincidan?"
        return {
            "reply_text": reply,
            "closing_text": "",
            "vendor_push": False,
            "vendor_message": ""
        }

    # Calificación genérica (sin dirección exacta)
    if ss["phase"] == "qualify_generic":
        t = raw_text.lower()

        # intenta capturar números
        m_pres = re.search(r"\$?\s?(\d{4,})", t.replace(".", ""))
        if m_pres and not ss["budget"]:
            try:
                ss["budget"] = float(m_pres.group(1))
            except Exception:
                pass

        m_dorm = re.search(r"(\d+)\s*(dorm|habit|amb)", t)
        if m_dorm and not ss["dorms"]:
            ss["dorms"] = int(m_dorm.group(1))

        if "cochera" in t and ("si" in t or "sí" in t):
            ss["cochera"] = True
        elif "cochera" in t and "no" in t:
            ss["cochera"] = False

        # Si ya tengo lo básico, sigo.
        need = []
        if not ss["budget"]: need.append("presupuesto")
        if not ss["dorms"]: need.append("dormitorios")

        if need:
            return {
                "reply_text": f"Para afinar la búsqueda me falta: {', '.join(need)}. ¿Me lo contás?",
                "closing_text": "",
                "vendor_push": False,
                "vendor_message": ""
            }

        # Si es alquiler, pasar al bloque de requisitos
        if ss["operation"] == "alquiler":
            ss["phase"] = "rent_reqs"
            return {
                "reply_text": (
                    "Genial. Para **alquiler** necesito validar unos datos:\n"
                    "1) ¿Tenés **ingresos demostrables** que aproximadamente **tripliquen** el costo?\n"
                    "2) ¿Qué tipo de **garantía** usarías: *seguro de caución Finaer* o *garantía propietaria*?\n"
                    "3) ¿Cuántos **habitantes** serían y **mascotas** sí/no?"
                ),
                "closing_text": "",
                "vendor_push": False,
                "vendor_message": ""
            }
        else:
            # venta
            ss["phase"] = "confirm_vendor_push"
            resumen = (
                f"Venta en {ss['zone'] or 'N/D'} | Tipo: {ss['tipo'] or 'N/D'} | "
                f"Dorms: {ss['dorms']} | Presupuesto: {to_money(ss['budget'])}"
            )
            return {
                "reply_text": (
                    "Perfecto, con esos datos te puedo acompañar y acercarte opciones. "
                    "¿Querés que te derive ahora a un asesor humano para continuar por este WhatsApp?"
                ),
                "closing_text": "",
                "vendor_push": False,
                "vendor_message": resumen
            }

    # Requisitos de alquiler
    if ss["phase"] == "rent_reqs":
        t = raw_text.lower()

        if ss["ingresos"] is None:
            if "si" in t or "sí" in t:
                ss["ingresos"] = True
            elif "no" in t:
                ss["ingresos"] = False

        if ss["garantia"] is None:
            if "finaer" in t or "caucion" in t or "caución" in t:
                ss["garantia"] = "finaer"
            elif "propietar" in t:
                ss["garantia"] = "propietaria"

        if ss["habitantes"] is None:
            m = re.search(r"(\d+)\s*(personas|habitantes?)", t)
            if m: ss["habitantes"] = int(m.group(1))

        if ss["mascotas"] is None:
            if "mascota" in t:
                if "no" in t:
                    ss["mascotas"] = False
                elif "si" in t or "sí" in t:
                    ss["mascotas"] = True

        need = []
        if ss["ingresos"] is None: need.append("ingresos demostrables (sí/no)")
        if ss["garantia"] is None: need.append("garantía (Finaer/propietaria)")
        if ss["habitantes"] is None: need.append("cantidad de habitantes")
        if ss["mascotas"] is None: need.append("mascotas (sí/no)")

        if need:
            return {
                "reply_text": f"Gracias. Me falta confirmar: {', '.join(need)}.",
                "closing_text": "",
                "vendor_push": False,
                "vendor_message": ""
            }

        # Ya está calificado
        ss["phase"] = "confirm_vendor_push"
        return {
            "reply_text": (
                "Con esos datos, ya puedo ayudarte a encontrar opciones. "
                f"Buscaremos {ss['tipo'] or 'propiedades'} en {ss['zone'] or 'la zona que indiques'} con {ss['dorms']} dormitorios "
                f"y un presupuesto de {to_money(ss['budget'])}."
                "\n\n¿Querés que **te derive a un asesor humano** para coordinar detalles?"
            ),
            "closing_text": "",
            "vendor_push": False,
            "vendor_message": ""
        }

    # Confirmación de derivación
    if ss["phase"] == "confirm_vendor_push":
        t = raw_text.lower()
        if any(x in t for x in ["si","sí","dale","ok","de una","por favor","quiero"]):
            ss["phase"] = "done"
            resumen = (
                f"Lead calificado – {ss['operation'] or 'N/D'}\n"
                f"Zona: {ss['zone'] or 'N/D'}\n"
                f"Tipo: {ss['tipo'] or 'N/D'}\n"
                f"Dorms: {ss['dorms'] or 'N/D'} | Cochera: {ss['cochera']}\n"
                f"Presupuesto: {to_money(ss['budget'])}\n"
            )
            if ss["operation"] == "alquiler":
                resumen += (
                    f"Ingresos: {('sí' if ss['ingresos'] else 'no') if ss['ingresos'] is not None else 'N/D'}\n"
                    f"Garantía: {ss['garantia'] or 'N/D'}\n"
                    f"Habitantes: {ss['habitantes'] or 'N/D'} | Mascotas: {('sí' if ss['mascotas'] else 'no') if ss['mascotas'] is not None else 'N/D'}\n"
                )
            if ss["address_or_link"]:
                resumen += f"Referencia: {ss['address_or_link']}\n"

            return {
                "reply_text": (
                    "Perfecto 🙌. Te derivo ahora con un asesor humano que va a seguir por este WhatsApp."
                ),
                "closing_text": "",
                "vendor_push": True,
                "vendor_message": resumen.strip(),
            }
        elif any(x in t for x in ["no","después","luego","mas tarde"]):
            ss["phase"] = "done"
            return {
                "reply_text": "¡Sin problema! Si más tarde querés continuar, escribime **derivar** y lo coordinamos.",
                "closing_text": "",
                "vendor_push": False,
                "vendor_message": ""
            }
        else:
            return {
                "reply_text": "¿Confirmás que querés que te derive a un asesor humano por este WhatsApp? (sí/no)",
                "closing_text": "",
                "vendor_push": False,
                "vendor_message": ""
            }

    # Fallback
    return {
        "reply_text": "¿Podrías repetirme eso con un poquito más de detalle? Así te ayudo mejor 🙂",
        "closing_text": "",
        "vendor_push": False,
        "vendor_message": ""
    }

# ====== Salud y debug ======
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug")
def debug():
    return {
        "USE_LLM": USE_LLM,
        "OPENAI_MODEL": OPENAI_MODEL,
        "DB_ready": DB is not None,
        "sessions": len(SESSIONS),
        "processed_cache": len(PROCESSED)
    }
