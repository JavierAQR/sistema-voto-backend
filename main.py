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
        # Decodificamos la imagen enviada desde la galería del Admin
        image_data = base64.b64decode(request.foto_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Guardamos la imagen físicamente
        nombre_archivo = f"{request.siglas.lower()}.jpg"
        ruta_guardado = os.path.join(PARTIDOS_DIR, nombre_archivo)
        cv2.imwrite(ruta_guardado, img)
        
        foto_url = f"/static_partidos/{nombre_archivo}"
        
        # Guardamos en base de datos (asegúrate de tener este modelo en models.py)
        nuevo_partido = models.PartidoPolitico(
            nombre=request.nombre, 
            siglas=request.siglas, 
            foto_url=foto_url
        )
        db.add(nuevo_partido)
        db.commit()
        return {"mensaje": "Partido registrado con éxito"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error guardando partido: {str(e)}")

@app.get("/partidos")
def listar_partidos(db: Session = Depends(get_db)):
    partidos = db.query(models.PartidoPolitico).all()
    # Devolvemos la lista para que el Votante la vea en su pantalla
    return [{"id": p.id, "nombre": p.nombre, "siglas": p.siglas, "foto_url": p.foto_url} for p in partidos]

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
        votante.huella_validada = False
        votante.rostro_validado = False
        # OJO: NO tocamos ha_votado

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
    votante = db.query(models.Votante).filter(models.Votante.id == request.votante_id).first()
    
    if not votante:
        raise HTTPException(status_code=404, detail="Votante no encontrado.")
    
    foto_oficial_path = os.path.join(UPLOAD_DIR, f"{votante.dni}.jpg")
    
    if not os.path.exists(foto_oficial_path):
        raise HTTPException(status_code=404, detail="No hay foto oficial para comparar.")
        
    if not request.foto_base64:
        raise HTTPException(status_code=400, detail="Captura facial vacía.")

    try:
        # 1. Decodificar lo que llega del Emulador o del Samsung
        image_data = base64.b64decode(request.foto_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        img_celular_rotada = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. GUARDAR ORIGINAL (CRÍTICO para ver cómo llega del A32)
        cv2.imwrite("debug_1_ORIGINAL.jpg", img_celular_rotada)

        # 3. ROTAR LA IMAGEN (Corrección de 90 grados)
        #celular
        img_celular_vertical = cv2.rotate(img_celular_rotada, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # emulador 
        # img_celular_vertical = cv2.rotate(img_celular_rotada, cv2.ROTATE_90_CLOCKWISE)

        # 4. GUARDAR CORREGIDA (CRÍTICO para ver si la IA la ve derecha)
        cv2.imwrite("debug_2_CORREGIDA.jpg", img_celular_vertical)
        
        print("--- SISTEMA BIOMÉTRICO: FOTOS DE DEBUG GENERADAS ---")

        # 5. PASAR LA FOTO ENDEREZADA A LA IA
        resultado = DeepFace.verify(
            img1_path = img_celular_vertical,
            img2_path = foto_oficial_path,
            model_name = 'Facenet512',
            detector_backend = 'opencv',
            enforce_detection = True,
            align = True
        )

        distancia = resultado["distance"]
        
        # Umbral en 0.48 (Tolerancia ideal para presentación del proyecto)
        match_final = distancia < 0.48 

        print(f"RESULTADO: DNI {votante.dni} | Distancia: {distancia:.4f} | Match: {match_final}")

        if match_final:
            votante.rostro_validado = True
            db.commit()
            return {"mensaje": "Acceso biométrico concedido."}
        else:
            raise HTTPException(status_code=401, detail="El rostro no coincide.")

    # Manejo correcto de errores
    except HTTPException:
        raise 
    except ValueError:
        raise HTTPException(status_code=400, detail="IA no detecta rostro. Mejore la iluminación.")
    except Exception as e:
        print(f"ERROR CRÍTICO: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del motor biométrico.")

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
    
    if not votante or votante.ha_votado:
        raise HTTPException(status_code=403, detail="Acceso denegado o ya votó.")
    
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