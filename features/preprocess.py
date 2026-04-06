"""
Tiền xử lý ảnh lá — replicates pipeline từ notebooks/01_data_preprocessing.ipynb.

Pipeline:
  1. Xóa nền bằng Otsu thresholding + morphological closing → nền đen
  2. Resize giữ tỉ lệ + padding đen → 256×256

Dùng trước extract_all() khi ảnh đầu vào chưa được xử lý.
"""

import cv2
import numpy as np

TARGET_SIZE = 256


def preprocess(image_path: str) -> np.ndarray:
    """
    Đọc ảnh thô và áp dụng toàn bộ pipeline tiền xử lý.

    Args:
        image_path: Đường dẫn đến ảnh (hỗ trợ Unicode trên Windows).

    Returns:
        Ảnh BGR 256×256 với lá trên nền đen.

    Raises:
        ValueError: Nếu không đọc được ảnh.
    """
    # Đọc ảnh Unicode-safe (giống BaseExtractor)
    buf = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    return preprocess_array(image)


def preprocess_array(image: np.ndarray) -> np.ndarray:
    """
    Áp dụng pipeline tiền xử lý lên ảnh BGR đã load sẵn.

    Args:
        image: Ảnh BGR (numpy array).

    Returns:
        Ảnh BGR 256×256 với lá trên nền đen.
    """
    # ------------------------------------------------------------------
    # Bước 1: Xóa nền bằng Otsu thresholding
    # ------------------------------------------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # THRESH_BINARY_INV: lá sáng hơn nền → sau invert lá = trắng, nền = đen
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological closing: lấp lỗ hổng bên trong lá
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Giữ màu gốc của lá, nền = đen
    result = cv2.bitwise_and(image, image, mask=thresh)

    # ------------------------------------------------------------------
    # Bước 2: Resize giữ tỉ lệ + padding đen → TARGET_SIZE × TARGET_SIZE
    # ------------------------------------------------------------------
    h, w = result.shape[:2]
    ratio = TARGET_SIZE / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    resized = cv2.resize(result, (new_w, new_h))

    dh = TARGET_SIZE - new_h
    dw = TARGET_SIZE - new_w
    top,    bottom = dh // 2, dh - dh // 2
    left,   right  = dw // 2, dw - dw // 2

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    return padded
