"""Abstract Base Class — Khuôn mẫu chung cho các Extractor."""

from abc import ABC, abstractmethod
import cv2
import numpy as np


class BaseExtractor(ABC):
    """
    Lớp cơ sở cho mọi feature extractor.

    Mọi extractor phải implement method `extract(image_path) -> dict`.
    Cung cấp sẵn `_read_image()` hỗ trợ đường dẫn Unicode trên Windows.
    """

    def _read_image(self, image_path: str) -> np.ndarray:
        """Đọc ảnh từ đường dẫn, hỗ trợ Unicode Windows."""
        buf = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        return image

    def _get_leaf_mask(self, image: np.ndarray) -> np.ndarray:
        """Tạo binary mask phân tách lá khỏi nền đen (threshold=10)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        return mask

    @abstractmethod
    def extract(self, image_path: str) -> dict:
        """Trích xuất đặc trưng từ ảnh. Trả về dict {tên_đặc_trưng: giá_trị}."""
        ...
