"""Image preprocessing for the leaf CBIR pipeline.

Steps:
  1. Segment the leaf and set the background to black.
  2. Resize with aspect ratio preserved and pad to 256x256.
"""

from __future__ import annotations

import cv2
import numpy as np

TARGET_SIZE = 256


def preprocess(image_path: str) -> np.ndarray:
    """Read an image from disk and apply the full preprocessing pipeline."""
    buf = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    return preprocess_array(image)


def preprocess_array(image: np.ndarray) -> np.ndarray:
    """Apply preprocessing to an already-loaded BGR image."""
    leaf_image, _ = segment_leaf(image)
    return resize_and_pad(leaf_image, TARGET_SIZE)


def segment_leaf(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the BGR leaf image on black background plus its binary mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = _keep_largest_component(mask)

    leaf_image = cv2.bitwise_and(image, image, mask=mask)
    return leaf_image, mask


def resize_and_pad(image: np.ndarray, target_size: int = TARGET_SIZE) -> np.ndarray:
    """Resize an image while preserving aspect ratio, then black-pad to square."""
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    ratio = target_size / max(h, w)
    new_h = max(1, int(round(h * ratio)))
    new_w = max(1, int(round(w * ratio)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    dh = target_size - new_h
    dw = target_size - new_w
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2

    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Remove small disconnected blobs from a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    return clean
