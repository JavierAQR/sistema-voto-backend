from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from database import SessionLocal, engine, Base
import models
import os
import base64
import cv2
import numpy as np
from deepface import DeepFace
from fastapi.responses import FileResponse

# Creamos las tablas en la base de datos
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="API de Administración Electoral - Blindaje Biométrico",
    version="1.2.0"
)

# --- CONFIGURACIÓN DE ARCHIVOS ESTÁTICOS ---
UPLOAD_DIR = "uploads/dni_fotos"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

PARTIDOS_DIR = "uploads/partidos"
if not os.path.exists(PARTIDOS_DIR):
    os.makedirs(PARTIDOS_DIR)

# Mount para que el celular descargue la foto
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

# Para que la app pueda descargar las fotos de los partidos
app.mount("/static_partidos", StaticFiles(directory=PARTIDOS_DIR), name="static_partidos")

# --- Dependencia: Conexión a la BD ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Esquemas Pydantic ---
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

# --- NUEVO ESQUEMA ---
class PartidoCreate(BaseModel):
    nombre: str
    siglas: str
    foto_base64: str

# --- ENDPOINTS DE PARTIDOS ---
@app.post("/admin/partidos")
def crear_partido(request: PartidoCreate, db: Session = Depends(get_db)):
    try:
        image_data = base64.b64decode(request.foto_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        nombre_archivo = f"{request.siglas.lower()}.jpg"
        ruta_guardado  = os.path.join(PARTIDOS_DIR, nombre_archivo)
        cv2.imwrite(ruta_guardado, img)
        foto_url = f"/static_partidos/{nombre_archivo}"
        # ─────────────────────────────────────────────────────────────────────
 
        # Evitar duplicados por siglas
        existente = db.query(models.PartidoPolitico).filter(
            models.PartidoPolitico.siglas == request.siglas
        ).first()
        if existente:
            raise HTTPException(status_code=400, detail=f"Las siglas '{request.siglas}' ya están registradas.")
 
        nuevo = models.PartidoPolitico(
            nombre   = request.nombre,
            siglas   = request.siglas,
            foto_url = foto_url
        )
        db.add(nuevo)
        db.commit()
        return {"mensaje": "Partido registrado con éxito"}
 
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
@app.delete("/admin/partidos/{partido_id}")
def eliminar_partido(partido_id: int, db: Session = Depends(get_db)):
    partido = db.query(models.PartidoPolitico).filter(
        models.PartidoPolitico.id == partido_id
    ).first()
    if not partido:
        raise HTTPException(status_code=404, detail="Partido no encontrado.")
 
    # Eliminar archivo local si existe
    if partido.foto_url.startswith("/static_partidos/"):
        nombre_archivo = partido.foto_url.replace("/static_partidos/", "")
        ruta = os.path.join(PARTIDOS_DIR, nombre_archivo)
        if os.path.exists(ruta):
            os.remove(ruta)
 
    db.delete(partido)
    db.commit()
    return {"mensaje": "Partido eliminado."}

@app.get("/partidos")
def listar_partidos(db: Session = Depends(get_db)):
    partidos = db.query(models.PartidoPolitico).all()
    # Devolvemos la lista para que el Votante la vea en su pantalla
    return [{"id": p.id, "nombre": p.nombre, "siglas": p.siglas, "foto_url": p.foto_url} for p in partidos]

@app.get("/admin/votantes")
def total_votantes(db: Session = Depends(get_db)):
    total = db.query(func.count(models.Votante.id)).scalar()
    return {"total": total}


 
@app.get("/admin")
def panel_admin():
    return FileResponse("admin_panel.html")


# --- Endpoints ---

@app.get("/")
def leer_raiz():
    return {"mensaje": "Servidor de Votación Activo - Motor Facenet512 Operativo"}

@app.post("/auth/register-dni")
def registrar_dni(request: DNIRequest, db: Session = Depends(get_db)):
    if not request.dni.isdigit() or len(request.dni) != 8:
        raise HTTPException(status_code=400, detail="DNI inválido.")

    foto_oficial_path = f"/static/{request.dni}.jpg"

    # Verificación de archivo físico
    if not os.path.exists(os.path.join(UPLOAD_DIR, f"{request.dni}.jpg")):
        print(f"ALERTA: No existe el archivo {request.dni}.jpg en {UPLOAD_DIR}")

    votante = db.query(models.Votante).filter(
        models.Votante.dni == request.dni
    ).first()

    # Si no existe, se crea
    if not votante:
        votante = models.Votante(
            dni=request.dni,
            huella_validada=False,
            rostro_validado=False,
            ha_votado=False
        )
        db.add(votante)

    # Si ya existe, se reinicia la sesión biométrica
    else:
        # ── NUEVO: bloquear si ya votó ──────────────────────────
        if votante.ha_votado:
            raise HTTPException(
                status_code=403,
                detail="Este DNI ya emitió su voto."
            )
        # ────────────────────────────────────────────────────────
        votante.huella_validada = False
        votante.rostro_validado = False

    db.commit()
    db.refresh(votante)

    return {
        "mensaje": "DNI reconocido por RENIEC",
        "votante_id": votante.id,
        "datos_oficiales": {
            "dni": request.dni,
            "foto_oficial_url": foto_oficial_path,
            "nombre_simulado": "CIUDADANO REGISTRADO"
        }
    }

@app.post("/auth/verify-face")
def verificar_rostro(request: RostroRequest, db: Session = Depends(get_db)):

    import pickle

    votante = db.query(models.Votante).filter(
        models.Votante.id == request.votante_id
    ).first()

    if not votante:
        raise HTTPException(status_code=404, detail="Votante no encontrado.")

    if not request.foto_base64:
        raise HTTPException(status_code=400, detail="Captura facial vacía.")

    # 🔥 convertir imagen
    image_data = base64.b64decode(request.foto_base64)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        # 🔥 obtener embedding del rostro actual
        current = DeepFace.represent(
            img_path=img,
            model_name="Facenet512",
            detector_backend="opencv",
            enforce_detection=True
        )[0]["embedding"]

        # -------------------------
        # CASO 1: PRIMER REGISTRO
        # -------------------------
        if votante.face_embedding is None:

            votante.face_embedding = pickle.dumps(current)
            votante.rostro_validado = True
            db.commit()

            return {"mensaje": "Rostro registrado correctamente."}

        # -------------------------
        # CASO 2: COMPARACIÓN
        # -------------------------
        stored = pickle.loads(votante.face_embedding)

        distancia = np.linalg.norm(
            np.array(current) - np.array(stored)
        )

        match = distancia < 10  # umbral estable para embeddings

        if match:
            votante.rostro_validado = True
            db.commit()
            return {"mensaje": "Acceso biométrico concedido."}
        else:
            raise HTTPException(status_code=401, detail="Rostro no coincide.")

    except Exception as e:
        print("ERROR BIOMETRÍA:", str(e))
        raise HTTPException(status_code=500, detail="Error en reconocimiento facial")
    
@app.post("/auth/verify-fingerprint")
def verificar_huella(request: HuellaRequest, db: Session = Depends(get_db)):
    votante = db.query(models.Votante).filter(models.Votante.id == request.votante_id).first()
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
    votante = db.query(models.Votante).filter(models.Votante.id == request.votante_id).first()
    
    if votante.ha_votado:
        raise HTTPException(status_code=403, detail="Ya has votado.")
    
    if not votante.huella_validada or not votante.rostro_validado:
        raise HTTPException(status_code=403, detail="Falta validación multifactor.")
        
    nuevo_voto = models.Voto(partido_id=request.partido_id)
    votante.ha_votado = True 
    db.add(nuevo_voto)
    db.commit()
    return {"mensaje": "Voto registrado correctamente."}

@app.get("/admin/results")
def conteo_de_votos(db: Session = Depends(get_db)):
    resultados = db.query(
        models.PartidoPolitico.nombre,
        models.PartidoPolitico.siglas,
        func.count(models.Voto.id).label("total_votos")
    ).outerjoin(
        models.Voto, models.PartidoPolitico.id == models.Voto.partido_id
    ).group_by(
        models.PartidoPolitico.id
    ).all()
    
    reporte = [{"partido": n, "siglas": s, "votos": t} for n, s, t in resultados]
    return {"mensaje": "Reporte de resultados", "resultados": reporte}