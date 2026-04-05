"""Trích xuất, Chuẩn hóa (Z-Score), Đánh Trọng số (Weighting), và Ingest vào PostgreSQL."""

import os
import sys
import glob
import time
import joblib
import numpy as np

# Thêm project root vào path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from tqdm import tqdm
from sqlalchemy import text
from db.database import engine, Base, SessionLocal
from db.models import LeafImage
from features.pipeline import extract_all
from core.config import FEATURE_WEIGHTS

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Phân bổ trọng số từ config cho các khối đặc trưng (Shape:10, Color:402, Texture:54, Sym:5, Vein:5)
# Chúng ta sẽ ép 5 biến từ FEATURE_WEIGHTS vào một mảng 476 chiều duy nhất để tiện nhân Numpy
# Giả định mảng FEATURE_WEIGHTS truyền từ config là [w_shape, w_color, w_texture, w_sym, w_vein]

w_shape = FEATURE_WEIGHTS[0] if len(FEATURE_WEIGHTS) > 0 else 5.0
w_color = FEATURE_WEIGHTS[1] if len(FEATURE_WEIGHTS) > 1 else 0.5
w_tex   = FEATURE_WEIGHTS[2] if len(FEATURE_WEIGHTS) > 2 else 1.0
w_sym   = FEATURE_WEIGHTS[3] if len(FEATURE_WEIGHTS) > 3 else 3.0
w_vein  = FEATURE_WEIGHTS[4] if len(FEATURE_WEIGHTS) > 4 else 3.0

WEIGHT_ARRAY = np.concatenate([
    np.full(10, w_shape),    # Shape
    np.full(402, w_color),   # Color (Bóp sức mạnh)
    np.full(54, w_tex),      # Texture
    np.full(5, w_sym),       # Symmetry
    np.full(5, w_vein)       # Vein
])

def apply_weighting(vectors: np.ndarray) -> np.ndarray:
    return vectors * WEIGHT_ARRAY

def ingest():
    from sklearn.preprocessing import StandardScaler

    image_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")))
    print(f"[*] TÌM THẤY {len(image_paths)} ẢNH TRONG THƯ MỤC. BẮT ĐẦU CHẠY...")

    start = time.time()
    
    # BƯỚC 1: TRÍCH XUẤT RAW VECTOR VÀO BỘ NHỚ RAM TRƯỚC
    X_raw = []
    metadata = []
    
    for path in tqdm(image_paths, desc="1. Trích xuất Đặc trưng OpenCV"):
        try:
            vector = extract_all(path)
            X_raw.append(vector)
            
            file_name = os.path.basename(path)
            image_id = int(os.path.splitext(file_name)[0])
            metadata.append((image_id, file_name, f"/storage/leaves/{file_name}"))
        except Exception as e:
            print(f"Lỗi ảnh {path}: {e}")

    X_raw = np.array(X_raw)
    
    # BƯỚC 2: TRAIN SCALER & APPLY Z-SCORE
    print(f"[*] 2. Đang huấn luyện Z-Score Scaler trên {X_raw.shape[0]} mẫu dữ liệu...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # BƯỚC 3: ÁP DỤNG TRỌNG SỐ WEIGHTING (Shape x5, Color x0.5...)
    print(f"[*] 3. Đang áp dụng Mảng chênh lệch Trọng số (Weighting)...")
    X_final = apply_weighting(X_scaled)
    
    # Save Scaler cho Server Web dùng sau này
    joblib.dump(scaler, SCALER_PATH)
    print(f"[+] Đã lưu mấu Standard Scaler tại: {SCALER_PATH}")

    # BƯỚC 4: LÀM SẠCH VÀ CHUẨN BỊ DATABASE
    print("[*] 4. Bắt đầu đẩy dữ liệu vào PostgreSQL...")
    with engine.connect() as conn:
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector;'))
        # Cường chế xoá trắng bảng cũ để tránh bị mix lộn xộn vector cũ/mới
        conn.execute(text('DROP TABLE IF EXISTS leaf_images CASCADE;'))
        conn.commit()
        
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    success = 0
    # Insert theo kiểu Batch/Iterate
    for i in tqdm(range(len(metadata)), desc="Nạp vào DB"):
        img_id, fname, fpath = metadata[i]
        vector_to_insert = X_final[i].tolist() # Chuyển về list thuần để cho Vector(476)
        
        record = LeafImage(
            image_id=img_id,
            file_name=fname,
            file_path=fpath,
            feature_vector=vector_to_insert
        )
        db.add(record)
        success += 1
        
        if success % 100 == 0:
            db.commit() # Lưu từng 100 dòng cho an toàn
            
    db.commit() # Lưu dòng cuối
    db.close()
    
    elapsed = time.time() - start
    print(f"\n[🚀 HOÀN TẤT!] Nạp trót lọt {success} ảnh cực phẩm vào Database trong {elapsed:.1f}s")
    print(f"Mô hình đã dẹp xong rác dữ liệu! Hãy chạy test_search.py nhé!")

if __name__ == "__main__":
    ingest()
