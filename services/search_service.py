"""Dịch vụ Tìm kiếm ảnh tương đồng (Search Service). 
Đóng gói quy trình: Nhận ảnh -> Gọi Pipeline trích xuất -> Gọi DB so sánh -> Trả Top 5"""

from sqlalchemy.orm import Session
from db.models import LeafImage
from features.pipeline import extract_all
from features.preprocess import preprocess_image
import joblib
import os
import numpy as np
from core.config import FEATURE_WEIGHTS

# Trọng số y hệt như quá trình nhồi Database
w_shape = FEATURE_WEIGHTS[0] if len(FEATURE_WEIGHTS) > 0 else 5.0
w_color = FEATURE_WEIGHTS[1] if len(FEATURE_WEIGHTS) > 1 else 0.5
w_tex   = FEATURE_WEIGHTS[2] if len(FEATURE_WEIGHTS) > 2 else 1.0
w_sym   = FEATURE_WEIGHTS[3] if len(FEATURE_WEIGHTS) > 3 else 3.0
w_vein  = FEATURE_WEIGHTS[4] if len(FEATURE_WEIGHTS) > 4 else 3.0

WEIGHT_ARRAY = np.concatenate([
    np.full(10, w_shape),    # Shape
    np.full(402, w_color),   # Color
    np.full(54, w_tex),      # Texture
    np.full(5, w_sym),       # Symmetry
    np.full(5, w_vein)       # Vein
])

class SearchService:
    def __init__(self, db: Session):
        self.db = db

    def search_by_l2(self, query_vector: list[float], limit: int = 5):
        """
        [Phương pháp 1]: Tìm 5 ảnh giống nhất bằng Khoảng cách Euclidean L2.
        Vector càng gần nhau về độ lớn, giá trị khoảng cách càng tiến về 0.
        """
        return self.db.query(LeafImage)\
                      .order_by(LeafImage.feature_vector.l2_distance(query_vector))\
                      .limit(limit)\
                      .all()

    def search_by_cosine(self, query_vector: list[float], limit: int = 5):
        """
        [Phương pháp 2]: Tìm 5 ảnh giống nhất bằng Độ tương đồng Cosine.
        Đo lường góc độ (hướng) của mảng Vector đại diện.
        """
        return self.db.query(LeafImage)\
                      .order_by(LeafImage.feature_vector.cosine_distance(query_vector))\
                      .limit(limit)\
                      .all()

    def process_and_search(self, image_path: str, method: str = 'l2', limit: int = 5):
        """Luồng hoàn chỉnh: Nhận ảnh -> Trích xuất 476 Vector -> Quét DB."""
        # BƯỚC 0: TIỀN XỬ LÝ (Xóa nền trắng, dập thành nền đen, Crop tỷ lệ chuẩn)
        try:
            clean_image_path = preprocess_image(image_path)
        except Exception as e:
            raise ValueError(f"Khởi chạy vấp lỗi tại bước xóa phông nền: {e}")

        # BƯỚC 1: Gọi Pipeline trích xuất toàn bộ 5 loại đặc trưng thực tế từ tệp sạch
        try:
            raw_vector = extract_all(clean_image_path)
        except Exception as e:
            # Xóa rác tệp tạm nếu lỗi
            if os.path.exists(clean_image_path) and clean_image_path != image_path:
                os.remove(clean_image_path)
            raise ValueError(f"Khởi chạy vấp lỗi trích xuất ảnh: {e}")
            
        # Xóa rác tệp tạm sau khi trích xuất xong (Tiết kiệm bộ nhớ)
        if os.path.exists(clean_image_path) and clean_image_path != image_path:
            os.remove(clean_image_path)

        # BƯỚC 1.5: CHUẨN HOÁ Z-SCORE VÀ ÁP TRỌNG SỐ (CỰC KỲ QUAN TRỌNG)
        scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError("Mô hình Scaler không tồn tại! Vui lòng chạy lại file ingest.py trước!")
        
        scaler = joblib.load(scaler_path)
        # Scaler chỉ nhận mảng 2D cho transform nên ta cấu hình lại Dims
        raw_vector_2d = np.array(raw_vector).reshape(1, -1)
        scaled_vector = scaler.transform(raw_vector_2d)[0]
        
        # Nhân với Ma trận Cân Bằng Trọng Số
        final_query_vector = (scaled_vector * WEIGHT_ARRAY).tolist()

        # BƯỚC 2: Ráp Mảng Vector đã Tinh chỉnh vào SQL Truy Vấn Database
        if method.lower() == 'cosine':
            results = self.search_by_cosine(final_query_vector, limit)
        else:
            results = self.search_by_l2(final_query_vector, limit)
            
        return results
