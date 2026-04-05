"""Trích xuất đặc trưng toàn bộ ảnh trong data/processed/ và đẩy vào PostgreSQL."""

import os
import sys
import glob
import time

# Thêm project root vào path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from tqdm import tqdm
from db.database import engine, Base, SessionLocal
from db.models import LeafImage
from features.pipeline import extract_all


DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def ingest():
    # Kích hoạt pgvector trước khi khởi tạo bảng chứa VECTOR()
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
        conn.commit()
        
    # Tạo bảng nếu chưa có
    Base.metadata.create_all(bind=engine)

    image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")))
    print(f"Tìm thấy {len(image_paths)} ảnh trong {DATA_DIR}")

    db = SessionLocal()
    success = 0
    errors = []

    start = time.time()

    for path in tqdm(image_paths, desc="Đang trích xuất"):
        file_name = os.path.basename(path)
        # image_id lấy từ tên file (vd: 1001.jpg -> 1001)
        image_id = int(os.path.splitext(file_name)[0])

        # Bỏ qua nếu đã tồn tại
        if db.query(LeafImage).filter_by(image_id=image_id).first():
            success += 1
            continue

        try:
            vector = extract_all(path)
            record = LeafImage(
                image_id=image_id,
                file_name=file_name,
                file_path=f"/storage/leaves/{file_name}",
                feature_vector=vector,
            )
            db.add(record)
            db.commit()
            success += 1
        except Exception as e:
            db.rollback()
            errors.append((file_name, str(e)))

    db.close()
    elapsed = time.time() - start

    print(f"\nHoàn tất: {success}/{len(image_paths)} ảnh trong {elapsed:.1f}s")
    if errors:
        print(f"Lỗi ({len(errors)}):")
        for fname, err in errors[:10]:
            print(f"  - {fname}: {err}")


if __name__ == "__main__":
    ingest()
