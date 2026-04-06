"""
Tính Z-score normalization params từ toàn bộ vectors trong DB,
lưu params ra file, rồi cập nhật lại tất cả vectors đã normalize.

Chạy sau khi ingest xong:
    python -X utf8 scripts/normalize.py
"""

import os
import sys

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from db.database import SessionLocal
from db.models import LeafImage

PARAMS_PATH = os.path.join(PROJECT_ROOT, 'data', 'zscore_params.npz')
BATCH_SIZE = 200


def compute_and_apply():
    db = SessionLocal()

    # 1. Fetch toàn bộ vectors
    print("Đọc vectors từ DB...")
    records = db.query(LeafImage.image_id, LeafImage.feature_vector).all()

    if not records:
        print("Không có dữ liệu trong DB.")
        db.close()
        return

    ids = [r.image_id for r in records]
    vectors = np.array([list(r.feature_vector) for r in records], dtype=np.float64)
    print(f"  Tổng: {len(ids)} vectors, {vectors.shape[1]} chiều")

    # 2. Tính mean và std theo từng chiều
    mean = vectors.mean(axis=0)
    std  = vectors.std(axis=0)

    zero_std = np.sum(std < 1e-8)
    if zero_std > 0:
        print(f"  Cảnh báo: {zero_std} chiều có std ≈ 0 (hằng số), sẽ cho giá trị 0 sau normalize")

    # 3. Lưu params
    np.savez(PARAMS_PATH, mean=mean, std=std)
    print(f"Params đã lưu: {PARAMS_PATH}")

    # 4. Normalize
    vectors_norm = (vectors - mean) / (std + 1e-8)

    # 5. Cập nhật DB theo batch
    print("Cập nhật vectors vào DB...")
    for i in tqdm(range(len(ids))):
        db.query(LeafImage).filter(LeafImage.image_id == ids[i]).update(
            {LeafImage.feature_vector: vectors_norm[i].tolist()},
            synchronize_session=False
        )
        if (i + 1) % BATCH_SIZE == 0:
            db.commit()

    db.commit()
    db.close()

    print(f"\nHoàn tất: {len(ids)} vectors đã được Z-score normalize và lưu vào DB.")
    print(f"Dùng features/zscore.py để normalize vector query khi tìm kiếm.")


if __name__ == "__main__":
    compute_and_apply()
