"""Tính trọng số 'cân bằng đóng góp' cho 4 nhóm đặc trưng.

Mỗi nhóm sẽ đóng góp xấp xỉ 25% vào tổng cosine distance trung bình
trên toàn bộ cặp ảnh trong DB:

    w_i = (1 / E[d_i]) / sum_j (1 / E[d_j])

Chạy SAU normalize.py (vector trong DB phải đã z-score):
    python scripts/compute_balanced_weights.py

Output: data/balanced_weights.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.spatial.distance import pdist

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from db.database import SessionLocal
from db.models import LeafImage

OUT_PATH = os.path.join(PROJECT_ROOT, "data", "balanced_weights.json")

GROUPS = ("shape", "color", "texture", "venation")
COLUMN_MAP = {
    "shape": LeafImage.shape_vector,
    "color": LeafImage.color_vector,
    "texture": LeafImage.texture_vector,
    "venation": LeafImage.venation_vector,
}


def compute_and_save():
    db = SessionLocal()

    print("Đọc vectors từ DB...")
    records = db.query(
        LeafImage.image_id,
        LeafImage.shape_vector,
        LeafImage.color_vector,
        LeafImage.texture_vector,
        LeafImage.venation_vector,
    ).all()

    if len(records) < 2:
        print(f"DB chỉ có {len(records)} ảnh — cần ≥ 2 để tính pairwise distance.")
        print("Hãy chạy ingest.py + normalize.py trước.")
        db.close()
        return

    matrices: dict[str, np.ndarray] = {
        "shape":    np.array([list(r.shape_vector)    for r in records], dtype=np.float64),
        "color":    np.array([list(r.color_vector)    for r in records], dtype=np.float64),
        "texture":  np.array([list(r.texture_vector)  for r in records], dtype=np.float64),
        "venation": np.array([list(r.venation_vector) for r in records], dtype=np.float64),
    }
    db.close()

    n = len(records)
    print(f"  Tổng: {n} ảnh")
    for g in GROUPS:
        print(f"    {g:9s}: shape={matrices[g].shape}")

    print("\nTính mean cosine distance per group (~{} cặp/nhóm)...".format(n * (n - 1) // 2))
    mean_d: dict[str, float] = {}
    for g in GROUPS:
        d = pdist(matrices[g], metric="cosine")
        mean_d[g] = float(d.mean())
        print(f"  E[d_{g:9s}] = {mean_d[g]:.6f}")

    inv = {g: 1.0 / mean_d[g] for g in GROUPS}
    total = sum(inv.values())
    weights = {g: inv[g] / total for g in GROUPS}
    contributions = {g: weights[g] * mean_d[g] for g in GROUPS}

    print("\nKết quả:")
    print(f"  {'group':10s} {'weight':>8s}  {'contribution':>14s}")
    for g in GROUPS:
        print(f"  {g:10s} {weights[g]:>8.4f}  {contributions[g]:>14.6f}")
    print(f"  {'(sum)':10s} {sum(weights.values()):>8.4f}")

    payload = {
        **{g: weights[g] for g in GROUPS},
        "_meta": {
            "mean_distances": mean_d,
            "expected_contributions": contributions,
            "n_samples": n,
            "metric": "cosine",
            "formula": "w_i = (1/E[d_i]) / sum_j(1/E[d_j])",
        },
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nĐã lưu: {OUT_PATH}")
    print("db/crud.py sẽ tự động đọc file này làm DEFAULT_WEIGHTS ở lần import kế tiếp.")


if __name__ == "__main__":
    compute_and_save()
