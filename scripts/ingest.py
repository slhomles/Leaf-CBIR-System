"""Trích xuất 4 nhóm đặc trưng từ data/processed và lưu vào PostgreSQL.

LƯU Ý: chỉ tạo schema mới nếu bảng chưa tồn tại. Nếu schema cũ (1 cột feature_vector)
vẫn còn, hãy DROP TABLE leaf_images trong DB trước khi chạy lại — script này không
tự migrate vì cấu trúc cột đã thay đổi hoàn toàn.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from tqdm import tqdm

from db.database import Base, SessionLocal, engine
from db.models import LeafImage
from features.pipeline import extract_all

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 4 feature vectors from data/processed and save them to DB."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite vectors for images that already exist.",
    )
    return parser.parse_args()


def ingest(force: bool = False) -> None:
    Base.metadata.create_all(bind=engine)

    image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")))
    print(f"Found {len(image_paths)} images in {DATA_DIR}")

    db = SessionLocal()
    success = 0
    errors: list[tuple[str, str]] = []
    start = time.time()

    for path in tqdm(image_paths, desc="Extracting"):
        file_name = os.path.basename(path)
        image_id = int(os.path.splitext(file_name)[0])

        existing = db.query(LeafImage).filter_by(image_id=image_id).first()
        if existing and not force:
            success += 1
            continue

        try:
            vectors = extract_all(path)
            if existing:
                existing.file_name = file_name
                existing.file_path = f"/storage/leaves/{file_name}"
                existing.shape_vector = vectors["shape"]
                existing.color_vector = vectors["color"]
                existing.texture_vector = vectors["texture"]
                existing.venation_vector = vectors["venation"]
            else:
                db.add(
                    LeafImage(
                        image_id=image_id,
                        file_name=file_name,
                        file_path=f"/storage/leaves/{file_name}",
                        shape_vector=vectors["shape"],
                        color_vector=vectors["color"],
                        texture_vector=vectors["texture"],
                        venation_vector=vectors["venation"],
                    )
                )
            db.commit()
            success += 1
        except Exception as exc:
            db.rollback()
            errors.append((file_name, str(exc)))

    db.close()
    elapsed = time.time() - start

    print(f"\nDone: {success}/{len(image_paths)} images in {elapsed:.1f}s")
    if errors:
        print(f"Errors ({len(errors)}):")
        for fname, err in errors[:10]:
            print(f"  - {fname}: {err}")


if __name__ == "__main__":
    ingest(force=parse_args().force)
