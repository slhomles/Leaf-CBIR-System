"""Pipeline trích xuất đặc trưng — ghép output của 5 extractor thành 1 vector."""

from features.extractors.shape import ShapeExtractor
from features.extractors.color import ColorExtractor
from features.extractors.texture import TextureExtractor
from features.extractors.symmetry_extractor import SymmetryExtractor
from features.extractors.vein_extractor import VeinExtractor


# Thứ tự cố định: shape(10) + color(10) + texture(10) + symmetry(5) + vein(5) = 40
EXTRACTORS = [
    ShapeExtractor(),
    ColorExtractor(),
    TextureExtractor(),
    SymmetryExtractor(),
    VeinExtractor(),
]


def extract_all(image_path: str) -> list[float]:
    """
    Trích xuất toàn bộ đặc trưng từ ảnh và ghép thành 1 vector phẳng.

    Returns:
        list[float] có 40 phần tử.
    """
    vector = []
    for extractor in EXTRACTORS:
        features = extractor.extract(image_path)
        vector.extend(features.values())
    return vector
