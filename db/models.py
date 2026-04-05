"""Định nghĩa ORM schemas — Bảng leaf_images lưu metadata + feature vector."""

from sqlalchemy import Column, Integer, String
from pgvector.sqlalchemy import Vector

from db.database import Base

# Tổng dimension: shape(10) + color(402) + texture(54) + symmetry(5) + vein(5) = 476
FEATURE_DIM = 476


class LeafImage(Base):
    __tablename__ = "leaf_images"

    image_id = Column(Integer, primary_key=True)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    feature_vector = Column(Vector(FEATURE_DIM))

    def __repr__(self):
        return f"<LeafImage(image_id={self.image_id}, file_name='{self.file_name}')>"
