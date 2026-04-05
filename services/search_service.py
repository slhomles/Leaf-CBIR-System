"""Dịch vụ Tìm kiếm ảnh tương đồng (Search Service). 
Đóng gói quy trình: Nhận ảnh -> Gọi Pipeline trích xuất -> Gọi DB so sánh -> Trả Top 5"""

from sqlalchemy.orm import Session
from db.models import LeafImage
from features.pipeline import extract_all

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
        
        # BƯỚC 1: Gọi Pipeline trích xuất toàn bộ 5 loại đặc trưng thực tế
        try:
            query_vector = extract_all(image_path)
        except Exception as e:
            raise ValueError(f"Khởi chạy vấp lỗi trích xuất ảnh: {e}")

        # BƯỚC 2: Ráp Mảng Vector 476 chiều vừa tìm được vào SQL Truy Vấn Database
        if method.lower() == 'cosine':
            results = self.search_by_cosine(query_vector, limit)
        else:
            results = self.search_by_l2(query_vector, limit)
            
        return results
