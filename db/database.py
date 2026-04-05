"""Cấu hình kết nối DB (PostgreSQL + pgvector)"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from core.config import DB_URI

engine = create_engine(DB_URI)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def get_db():
    """Yield một DB session, tự đóng khi xong."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
