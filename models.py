from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, LargeBinary
from datetime import datetime
from database import Base


class PartidoPolitico(Base):
    __tablename__ = "partidos"

    id       = Column(Integer, primary_key=True, index=True)
    nombre   = Column(String, unique=True, index=True)
    siglas   = Column(String, unique=True, index=True)
    foto_url = Column(String)   # URL de Cloudinary


class Votante(Base):
    __tablename__ = "votantes"

    id              = Column(Integer, primary_key=True, index=True)
    dni             = Column(String, unique=True, index=True)
    nombre          = Column(String, nullable=True)          # nombre completo
    foto_url        = Column(String, nullable=True)          # URL Cloudinary foto oficial
    huella_validada = Column(Boolean, default=False)
    rostro_validado = Column(Boolean, default=False)
    ha_votado       = Column(Boolean, default=False)
    face_embedding  = Column(LargeBinary, nullable=True)


class Voto(Base):
    __tablename__ = "votos"

    id         = Column(Integer, primary_key=True, index=True)
    partido_id = Column(Integer, ForeignKey("partidos.id"))
    fecha_hora = Column(DateTime, default=datetime.utcnow)
    # Sin votante_id intencional: voto secreto