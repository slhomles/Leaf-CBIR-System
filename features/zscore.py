"""Z-score normalization per-group cho 4 nhóm đặc trưng (shape/color/texture/venation).

Params được lưu tại data/zscore_params.npz dưới dạng 8 mảng:
    shape_mean, shape_std,
    color_mean, color_std,
    texture_mean, texture_std,
    venation_mean, venation_std

Lưu một lần bằng scripts/normalize.py, lazy-load khi gọi normalize_*.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "zscore_params.npz")
_GROUPS = ("shape", "color", "texture", "venation")

_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = None


def _load_params() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    global _cache
    if _cache is not None:
        return _cache

    path = os.path.abspath(_PARAMS_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy Z-score params tại: {path}\n"
            "Hãy chạy: python -X utf8 scripts/normalize.py"
        )

    data = np.load(path)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for group in _GROUPS:
        cache[group] = (data[f"{group}_mean"], data[f"{group}_std"])
    _cache = cache
    return cache


def normalize_group(vector: Iterable[float], group: str) -> list[float]:
    """Áp dụng Z-score cho 1 nhóm đặc trưng cụ thể."""
    if group not in _GROUPS:
        raise ValueError(f"Unknown group '{group}'. Phải là một trong {_GROUPS}.")
    mean, std = _load_params()[group]
    v = np.asarray(list(vector), dtype=np.float64)
    return ((v - mean) / (std + 1e-8)).tolist()


def normalize_all(vectors: dict[str, list[float]]) -> dict[str, list[float]]:
    """Normalize cả 4 nhóm cùng lúc, trả về dict cùng cấu trúc."""
    return {group: normalize_group(vectors[group], group) for group in _GROUPS}


def get_params(group: str) -> tuple[np.ndarray, np.ndarray]:
    if group not in _GROUPS:
        raise ValueError(f"Unknown group '{group}'.")
    return _load_params()[group]


def reset_cache() -> None:
    """Buộc reload params từ disk (dùng sau khi normalize.py chạy lại)."""
    global _cache
    _cache = None
