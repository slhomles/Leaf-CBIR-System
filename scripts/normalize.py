"""Tính Z-score per-group cho 4 nhóm vector và overwrite trong DB.

Chạy SAU ingest.py:
    python -X utf8 scripts/normalize.py

Output: data/zscore_params.npz với 8 mảng (mean & std cho mỗi nhóm).
"""

from __future__ import annotations

import os
import sys

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from db.database import SessionLocal
from db.models import LeafImage

PARAMS_PATH = os.path.join(PROJECT_ROOT, "data", "zscore_params.npz")
BATCH_SIZE = 200

GROUPS = ("shape", "color", "texture", "venation")
COLUMN_MAP = {
    "shape": LeafImage.shape_vector,
    "color": LeafImage.color_vector,
    "texture": LeafImage.texture_vector,
    "venation": LeafImage.venation_vector,
}


def compute_and_apply():
    db = SessionLocal()

    print("Đọc vectors từ DB...")
    records = db.query(
        LeafImage.image_id,
        LeafImage.shape_vector,
        LeafImage.color_vector,
        LeafImage.texture_vector,
        LeafImage.venation_vector,
    ).all()

    if not records:
        print("Không có dữ liệu trong DB. Hãy chạy scripts/ingest.py trước.")
        db.close()
        return

    ids = [r.image_id for r in records]
    matrices: dict[str, np.ndarray] = {
        "shape":    np.array([list(r.shape_vector)    for r in records], dtype=np.float64),
        "color":    np.array([list(r.color_vector)    for r in records], dtype=np.float64),
        "texture":  np.array([list(r.texture_vector)  for r in records], dtype=np.float64),
        "venation": np.array([list(r.venation_vector) for r in records], dtype=np.float64),
    }
    print(f"  Tổng: {len(ids)} ảnh")
    for g in GROUPS:
        print(f"    {g:9s}: shape={matrices[g].shape}")

    # 1. Tính mean/std từng nhóm
    params: dict[str, np.ndarray] = {}
    for g in GROUPS:
        mean = matrices[g].mean(axis=0)
        std = matrices[g].std(axis=0)
        zero_std = int(np.sum(std < 1e-8))
        if zero_std > 0:
            print(f"  Cảnh báo [{g}]: {zero_std} chiều có std ≈ 0 → 0 sau normalize")
        params[f"{g}_mean"] = mean
        params[f"{g}_std"] = std

    # 2. Lưu params
    np.savez(PARAMS_PATH, **params)
    print(f"Params đã lưu: {PARAMS_PATH}")

    # 3. Normalize từng nhóm
    normalized: dict[str, np.ndarray] = {}
    for g in GROUPS:
        normalized[g] = (matrices[g] - params[f"{g}_mean"]) / (params[f"{g}_std"] + 1e-8)

    # 4. Cập nhật DB theo batch
    print("Cập nhật vectors vào DB...")
    for i in tqdm(range(len(ids))):
        update_payload = {COLUMN_MAP[g]: normalized[g][i].tolist() for g in GROUPS}
        db.query(LeafImage).filter(LeafImage.image_id == ids[i]).update(
            update_payload,
            synchronize_session=False,
        )
        if (i + 1) % BATCH_SIZE == 0:
            db.commit()

    db.commit()
    db.close()

    print(f"\nHoàn tất: {len(ids)} ảnh đã được Z-score per-group và lưu vào DB.")
    print("Dùng features.zscore.normalize_all để normalize vector query khi tìm kiếm.")


if __name__ == "__main__":
    compute_and_apply()
