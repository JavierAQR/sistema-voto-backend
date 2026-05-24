import os
import base64
import pickle

import cloudinary
import cloudinary.uploader
import cv2
import numpy as np
from deepface import DeepFace
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session
from fastapi import Form, UploadFile, File

from database import SessionLocal, engine, Base
import models

load_dotenv()

# ── Crear tablas ───────────────────────────────────────────────────────────────
Base.metadata.create_all(bind=engine)

# ── Cloudinary ─────────────────────────────────────────────────────────────────
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key    = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure     = True
)

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title   = "API Electoral · Blindaje Biométrico",
    version = "2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DB dependency ──────────────────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ── Helper: subir imagen a Cloudinary ─────────────────────────────────────────
def subir_foto_partido(foto_base64: str, public_id: str) -> str:
    resultado = cloudinary.uploader.upload(
        f"data:image/jpeg;base64,{foto_base64}",
        folder        = "electoral/partidos",
        public_id     = public_id,
        overwrite     = True,
        transformation= [{"width": 400, "height": 400, "crop": "fill"}]
    )
    return resultado["secure_url"]

def subir_foto_ciudadano(foto_base64: str, public_id: str) -> str:
    """Sin transformaciones para preservar el rostro completo."""
    resultado = cloudinary.uploader.upload(
        f"data:image/jpeg;base64,{foto_base64}",
        folder    = "electoral/ciudadanos",
        public_id = public_id,
        overwrite = True
    )
    return resultado["secure_url"]


def generar_embedding_desde_url(foto_url: str):
    """Descarga la foto de Cloudinary y genera el embedding facial."""
    import urllib.request
    import tempfile
    
    # Descargar la imagen temporalmente
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        urllib.request.urlretrieve(foto_url, tmp.name)
        tmp_path = tmp.name
    
    try:
        resultado = DeepFace.represent(
            img_path         = tmp_path,
            model_name       = "Facenet512",
            detector_backend = "opencv",
            enforce_detection= False  
        )
        return resultado[0]["embedding"]
    finally:
        os.remove(tmp_path)

# ── Schemas Pydantic ───────────────────────────────────────────────────────────
class DNIRequest(BaseModel):
    dni: str

class HuellaRequest(BaseModel):
    votante_id: int
    huella_exitosa: bool

class RostroRequest(BaseModel):
    votante_id: int
    rostro_exitoso: bool
    foto_base64: str = None

class VotoRequest(BaseModel):
    votante_id: int
    partido_id: int

class PartidoCreate(BaseModel):
    nombre: str
    siglas: str
    foto_base64: str

class CiudadanoCreate(BaseModel):
    dni: str
    nombre: str
    foto_base64: str

@app.get("/admin/debug/{dni}")
def debug_votante(dni: str, db: Session = Depends(get_db)):
    v = db.query(models.Votante).filter(models.Votante.dni == dni).first()
    if not v:
        return {"error": "no encontrado"}
    return {
        "dni": v.dni,
        "tiene_foto_url": v.foto_url is not None,
        "foto_url": v.foto_url,
        "tiene_embedding": v.face_embedding is not None,
    }

@app.get("/admin/test-distancia/{dni}")
def test_distancia(dni: str, db: Session = Depends(get_db)):
    votante = db.query(models.Votante).filter(models.Votante.dni == dni).first()
    if not votante or not votante.foto_url:
        return {"error": "no encontrado"}
    
    try:
        embedding_oficial = generar_embedding_desde_url(votante.foto_url)
        return {
            "ok": True,
            "embedding_len": len(embedding_oficial),
            "foto_url": votante.foto_url
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ══════════════════════════════════════════════════════════════════════════════
# PANEL ADMIN (sirve el HTML)
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/admin", include_in_schema=False)
def panel_admin():
    return FileResponse("admin_panel.html")

# ══════════════════════════════════════════════════════════════════════════════
# PARTIDOS
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/admin/partidos")
def crear_partido(request: PartidoCreate, db: Session = Depends(get_db)):
    existente = db.query(models.PartidoPolitico).filter(
        models.PartidoPolitico.siglas == request.siglas
    ).first()
    if existente:
        raise HTTPException(status_code=400, detail=f"Las siglas '{request.siglas}' ya están registradas.")

    try:
        foto_url = subir_foto_partido(request.foto_base64, request.siglas.lower())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error subiendo imagen: {str(e)}")

    partido = models.PartidoPolitico(
        nombre   = request.nombre,
        siglas   = request.siglas,
        foto_url = foto_url
    )
    db.add(partido)
    db.commit()
    return {"mensaje": "Partido registrado con éxito"}


@app.get("/partidos")
def listar_partidos(db: Session = Depends(get_db)):
    partidos = db.query(models.PartidoPolitico).all()
    return [
        {"id": p.id, "nombre": p.nombre, "siglas": p.siglas, "foto_url": p.foto_url}
        for p in partidos
    ]


@app.delete("/admin/partidos/{partido_id}")
def eliminar_partido(partido_id: int, db: Session = Depends(get_db)):
    partido = db.query(models.PartidoPolitico).filter(
        models.PartidoPolitico.id == partido_id
    ).first()
    if not partido:
        raise HTTPException(status_code=404, detail="Partido no encontrado.")

    # Intentar borrar de Cloudinary (no crítico si falla)
    try:
        public_id = f"electoral/partidos/{partido.siglas.lower()}"
        cloudinary.uploader.destroy(public_id)
    except Exception:
        pass

    db.delete(partido)
    db.commit()
    return {"mensaje": "Partido eliminado."}

# ══════════════════════════════════════════════════════════════════════════════
# CIUDADANOS (padrón electoral)
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/admin/ciudadanos")
def registrar_ciudadano(request: CiudadanoCreate, db: Session = Depends(get_db)):
    if not request.dni.isdigit() or len(request.dni) != 8:
        raise HTTPException(status_code=400, detail="DNI inválido.")

    try:
        foto_url = subir_foto_ciudadano(request.foto_base64, request.dni)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error subiendo foto: {str(e)}")

    # ── NUEVO: generar embedding desde la foto oficial ─────────
    try:
        embedding = generar_embedding_desde_url(foto_url)
        embedding_bytes = pickle.dumps(embedding)
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo generar embedding para {request.dni}: {e}")
        embedding_bytes = None
    # ──────────────────────────────────────────────────────────

    votante = db.query(models.Votante).filter(
        models.Votante.dni == request.dni
    ).first()

    if votante:
        votante.nombre         = request.nombre
        votante.foto_url       = foto_url
        votante.face_embedding = embedding_bytes 
        votante.rostro_validado = False
        votante.huella_validada = False
    else:
        votante = models.Votante(
            dni             = request.dni,
            nombre          = request.nombre,
            foto_url        = foto_url,
            face_embedding  = embedding_bytes,  
            huella_validada = False,
            rostro_validado = False,
            ha_votado       = False
        )
        db.add(votante)

    db.commit()
    return {"mensaje": f"Ciudadano {request.dni} registrado con éxito.", "foto_url": foto_url}


@app.get("/admin/ciudadanos")
def listar_ciudadanos(db: Session = Depends(get_db)):
    votantes = db.query(models.Votante).all()
    return [
        {
            "id":        v.id,
            "dni":       v.dni,
            "nombre":    v.nombre,
            "foto_url":  v.foto_url,
            "tiene_foto": v.foto_url is not None,
            "ha_votado": v.ha_votado,
        }
        for v in votantes
    ]

@app.get("/admin/verificaciones-faciales")
def listar_verificaciones(db: Session = Depends(get_db)):
    votantes = db.query(models.Votante).all()

    return [
        {
            "id": v.id,
            "dni": v.dni,
            "rostro_validado": v.rostro_validado,
            "huella_validada": v.huella_validada,
            "ha_votado": v.ha_votado,
            "foto_url":  v.foto_url,
        }
        for v in votantes
    ]


@app.delete("/admin/ciudadanos/{votante_id}")
def eliminar_ciudadano(votante_id: int, db: Session = Depends(get_db)):
    votante = db.query(models.Votante).filter(
        models.Votante.id == votante_id
    ).first()
    if not votante:
        raise HTTPException(status_code=404, detail="Ciudadano no encontrado.")

    try:
        cloudinary.uploader.destroy(f"electoral/ciudadanos/{votante.dni}")
    except Exception:
        pass

    db.delete(votante)
    db.commit()
    return {"mensaje": "Ciudadano eliminado."}


@app.get("/admin/votantes")
def total_votantes(db: Session = Depends(get_db)):
    total = db.query(func.count(models.Votante.id)).scalar()
    return {"total": total}

# ══════════════════════════════════════════════════════════════════════════════
# AUTENTICACIÓN / VOTACIÓN
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/")
def raiz():
    return {"mensaje": "Servidor Electoral Activo · v2.0"}


@app.post("/auth/register-dni")
def registrar_dni(request: DNIRequest, db: Session = Depends(get_db)):
    if not request.dni.isdigit() or len(request.dni) != 8:
        raise HTTPException(status_code=400, detail="DNI inválido.")

    votante = db.query(models.Votante).filter(
        models.Votante.dni == request.dni
    ).first()

    # Si no existe en BD significa que el admin no subió la foto aún
    if not votante or not votante.foto_url:
        raise HTTPException(
            status_code=404,
            detail="DNI no registrado. El administrador debe subir la foto primero."
        )

    if votante.ha_votado:
        raise HTTPException(status_code=403, detail="Este DNI ya emitió su voto.")

    votante.huella_validada = False
    votante.rostro_validado = False
    db.commit()
    db.refresh(votante)

    return {
        "mensaje"   : "DNI reconocido",
        "votante_id": votante.id,
        "datos_oficiales": {
            "dni"             : votante.dni,
            "foto_oficial_url": votante.foto_url,  # URL real de Cloudinary
            "nombre_simulado" : "CIUDADANO REGISTRADO"
        }
    }

@app.post("/admin/upload-dni-foto")
async def subir_foto_dni(
    dni: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not dni.isdigit() or len(dni) != 8:
        raise HTTPException(status_code=400, detail="DNI inválido")

    contenido   = await file.read()
    foto_base64 = base64.b64encode(contenido).decode("utf-8")

    try:
        foto_url = subir_foto_ciudadano(foto_base64, dni)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error subiendo foto: {str(e)}")

    # Siempre crear/actualizar el votante con la URL real de Cloudinary
    votante = db.query(models.Votante).filter(
        models.Votante.dni == dni
    ).first()

    if votante:
        votante.foto_url       = foto_url
        votante.face_embedding = None   # resetear embedding para que se regenere
        votante.rostro_validado = False
        votante.huella_validada = False
    else:
        votante = models.Votante(
            dni             = dni,
            foto_url        = foto_url,
            face_embedding  = None,
            huella_validada = False,
            rostro_validado = False,
            ha_votado       = False
        )
        db.add(votante)

    db.commit()
    return {"mensaje": "Foto guardada", "foto_url": foto_url}


@app.post("/auth/verify-face")
def verificar_rostro(request: RostroRequest, db: Session = Depends(get_db)):
    votante = db.query(models.Votante).filter(
        models.Votante.id == request.votante_id
    ).first()
    if not votante:
        raise HTTPException(status_code=404, detail="Votante no encontrado.")
    if not request.foto_base64:
        raise HTTPException(status_code=400, detail="Captura facial vacía.")

    image_data = base64.b64decode(request.foto_base64)
    nparr = np.frombuffer(image_data, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        current = DeepFace.represent(
            img_path         = img,
            model_name       = "Facenet512",
            detector_backend = "opencv",
            enforce_detection= True
        )[0]["embedding"]

        if votante.face_embedding is None:
            # Primera vez — comparar contra la foto oficial de Cloudinary
            embedding_oficial = generar_embedding_desde_url(votante.foto_url)
            distancia = np.linalg.norm(
                np.array(current) - np.array(embedding_oficial)
            )
            print(f"Distancia facial (vs foto oficial) para {votante.dni}: {distancia}")

            if distancia < 10:
                # Guardar embedding de la selfie para futuras sesiones
                votante.face_embedding  = pickle.dumps(current)
                votante.rostro_validado = True
                db.commit()
                return {"mensaje": "Acceso biométrico concedido."}
            else:
                raise HTTPException(status_code=401, detail="Rostro no coincide con la foto oficial.")
        else:
            # Ya tiene embedding — comparar directamente
            stored    = pickle.loads(votante.face_embedding)
            distancia = np.linalg.norm(np.array(current) - np.array(stored))
            print(f"Distancia facial (vs embedding guardado) para {votante.dni}: {distancia}")

            if distancia < 10:
                votante.rostro_validado = True
                db.commit()
                return {"mensaje": "Acceso biométrico concedido."}
            else:
                raise HTTPException(status_code=401, detail="Rostro no coincide.")

    except HTTPException:
        raise
    except Exception as e:
        print("ERROR BIOMETRÍA:", str(e))
        raise HTTPException(status_code=500, detail="Error en reconocimiento facial.")


@app.post("/auth/verify-fingerprint")
def verificar_huella(request: HuellaRequest, db: Session = Depends(get_db)):
    votante = db.query(models.Votante).filter(
        models.Votante.id == request.votante_id
    ).first()
    if not votante:
        raise HTTPException(status_code=404, detail="Votante no encontrado.")

    if request.huella_exitosa:
        votante.huella_validada = True
        db.commit()
        return {"mensaje": "Huella validada con éxito."}
    else:
        raise HTTPException(status_code=401, detail="Fallo en validación de huella.")


@app.post("/voting/cast")
def emitir_voto(request: VotoRequest, db: Session = Depends(get_db)):
    votante = db.query(models.Votante).filter(
        models.Votante.id == request.votante_id
    ).first()
    if not votante:
        raise HTTPException(status_code=404, detail="Votante no encontrado.")
    if votante.ha_votado:
        raise HTTPException(status_code=403, detail="Ya has votado.")
    if not votante.huella_validada or not votante.rostro_validado:
        raise HTTPException(status_code=403, detail="Falta validación multifactor.")

    voto = models.Voto(partido_id=request.partido_id)
    votante.ha_votado = True
    db.add(voto)
    db.commit()
    return {"mensaje": "Voto registrado correctamente."}


@app.get("/admin/results")
def conteo_votos(db: Session = Depends(get_db)):
    resultados = db.query(
        models.PartidoPolitico.nombre,
        models.PartidoPolitico.siglas,
        func.count(models.Voto.id).label("total_votos")
    ).outerjoin(
        models.Voto, models.PartidoPolitico.id == models.Voto.partido_id
    ).group_by(
        models.PartidoPolitico.id
    ).all()

    return {
        "mensaje"   : "Reporte de resultados",
        "resultados": [{"partido": n, "siglas": s, "votos": t} for n, s, t in resultados]
    }
