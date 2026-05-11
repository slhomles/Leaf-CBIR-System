"""Pipeline trích xuất đặc trưng — trả về 4 vector riêng biệt."""

from __future__ import annotations

from features.extractors.shape import ShapeExtractor
from features.extractors.color import ColorExtractor
from features.extractors.texture import TextureExtractor
from features.extractors.vein_extractor import VeinExtractor

# Single instance per extractor (state-less, an toàn để dùng chung)
_SHAPE = ShapeExtractor()
_COLOR = ColorExtractor()
_TEXTURE = TextureExtractor()
_VEIN = VeinExtractor()


def extract_all(image_path: str) -> dict[str, list[float]]:
    """
    Trích xuất 4 nhóm đặc trưng từ 1 ảnh.

    Returns:
        dict với keys:
          - "shape":    list 10 float
          - "color":    list 72 float
          - "texture":  list 38 float
          - "venation": list 5 float
    """
    return {
        "shape": list(_SHAPE.extract(image_path).values()),
        "color": list(_COLOR.extract(image_path).values()),
        "texture": list(_TEXTURE.extract(image_path).values()),
        "venation": list(_VEIN.extract(image_path).values()),
    }
