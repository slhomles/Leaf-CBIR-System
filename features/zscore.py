"""
Utility Z-score normalization cho query vector lúc tìm kiếm.

Dùng params đã tính từ scripts/normalize.py (lưu tại data/zscore_params.npz).

Ví dụ:
    from features.zscore import normalize
    from features.pipeline import extract_all

    raw_vec = extract_all("path/to/image.jpg")
    norm_vec = normalize(raw_vec)   # dùng norm_vec để tìm kiếm trong pgvector
"""

import os
import numpy as np

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'zscore_params.npz')

_mean: np.ndarray | None = None
_std:  np.ndarray | None = None


def _load_params() -> tuple[np.ndarray, np.ndarray]:
    """Tải mean và std từ file (lazy load, chỉ đọc 1 lần)."""
    global _mean, _std
    if _mean is None:
        path = os.path.abspath(_PARAMS_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Không tìm thấy Z-score params tại: {path}\n"
                "Hãy chạy: python -X utf8 scripts/normalize.py"
            )
        data = np.load(path)
        _mean = data['mean']
        _std  = data['std']
    return _mean, _std


def normalize(vector: list[float]) -> list[float]:
    """
    Áp dụng Z-score normalization lên vector đặc trưng thô.

    Args:
        vector: list[float] có 40 phần tử (output của extract_all).

    Returns:
        list[float] đã normalize, sẵn sàng để tìm kiếm trong pgvector.
    """
    mean, std = _load_params()
    v = np.array(vector, dtype=np.float64)
    return ((v - mean) / (std + 1e-8)).tolist()


def get_params() -> tuple[np.ndarray, np.ndarray]:
    """Trả về (mean, std) array nếu cần dùng thủ công."""
    return _load_params()
