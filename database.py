from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Creamos un archivo local llamado votacion.db
SQLALCHEMY_DATABASE_URL = "sqlite:///./votacion.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()