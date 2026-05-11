"""Định nghĩa ORM schemas — Bảng leaf_images lưu metadata + 4 feature vector riêng biệt."""

from sqlalchemy import Column, Integer, String
from pgvector.sqlalchemy import Vector

from db.database import Base

# Số chiều cho từng nhóm đặc trưng
SHAPE_DIM = 10        # 7 hình học + 3 Hu moments (log-transformed)
COLOR_DIM = 72        # Radial Spatial HSV: 8H × 3S × 3 vòng đồng tâm
TEXTURE_DIM = 38      # 4 GLCM Haralick + 24 Gabor (12 filter × mean+var) + 10 LBP riu2
VENATION_DIM = 5      # density, edge density, thickness, contrast, uniformity


class LeafImage(Base):
    __tablename__ = "leaf_images"

    image_id = Column(Integer, primary_key=True)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)

    shape_vector = Column(Vector(SHAPE_DIM))
    color_vector = Column(Vector(COLOR_DIM))
    texture_vector = Column(Vector(TEXTURE_DIM))
    venation_vector = Column(Vector(VENATION_DIM))

    def __repr__(self):
        return f"<LeafImage(image_id={self.image_id}, file_name='{self.file_name}')>"
