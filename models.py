from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime
from datetime import datetime
from database import Base

class PartidoPolitico(Base):
    __tablename__ = "partidos"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, unique=True, index=True)
    siglas = Column(String, unique=True, index=True)
    foto_url = Column(String)

class Votante(Base):
    __tablename__ = "votantes"

    id = Column(Integer, primary_key=True, index=True)
    dni = Column(String, unique=True, index=True)
    # Estados de validación multifactor
    huella_validada = Column(Boolean, default=False)
    rostro_validado = Column(Boolean, default=False)
    ha_votado = Column(Boolean, default=False)

class Voto(Base):
    __tablename__ = "votos"

    id = Column(Integer, primary_key=True, index=True)
    partido_id = Column(Integer, ForeignKey("partidos.id"))
    fecha_hora = Column(DateTime, default=datetime.utcnow)
    # Nota: Intencionalmente NO incluimos el votante_id para mantener el voto secreto