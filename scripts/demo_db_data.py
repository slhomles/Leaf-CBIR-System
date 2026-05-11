import os
import sys

# Đảm bảo in ra utf-8 trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

# Thêm thư mục gốc vào sys.path để import các module
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from db.database import SessionLocal
from db.models import LeafImage

def demo_data():
    db = SessionLocal()
    # Lấy thử 3 bản ghi đầu tiên trong cơ sở dữ liệu
    records = db.query(LeafImage).limit(3).all()
    
    if not records:
        print("Cơ sở dữ liệu đang trống. Vui lòng chạy pipeline trích xuất trước.")
        db.close()
        return
        
    print(f"Da ket noi thanh cong toi Database. Lay {len(records)} ban ghi de minh chung:\n")
    
    for r in records:
        print(f"--- Thong tin anh: {r.file_name} (ID: {r.image_id}) ---")
        
        # Chỉ in ra tối đa 4 giá trị đầu tiên của mỗi vector để báo cáo gọn gàng
        shape_preview = list(r.shape_vector)[:4] if r.shape_vector is not None else []
        color_preview = list(r.color_vector)[:4] if r.color_vector is not None else []
        texture_preview = list(r.texture_vector)[:4] if r.texture_vector is not None else []
        venation_preview = list(r.venation_vector)[:4] if r.venation_vector is not None else []
        
        # Làm tròn 4 chữ số thập phân cho dễ nhìn
        print(f"   - Shape Vector (10 chieu)    : {[round(v, 4) for v in shape_preview]} ...")
        print(f"   - Color Vector (72 chieu)    : {[round(v, 4) for v in color_preview]} ...")
        print(f"   - Texture Vector (38 chieu)  : {[round(v, 4) for v in texture_preview]} ...")
        print(f"   - Venation Vector (5 chieu)  : {[round(v, 4) for v in venation_preview]} ...\n")
        
    db.close()

if __name__ == "__main__":
    demo_data()
