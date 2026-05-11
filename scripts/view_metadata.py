import sys
import os

# Đảm bảo Python có thể import được các module từ thư mục gốc
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from db.database import SessionLocal
from db.models import LeafImage

def view_metadata():
    db = SessionLocal()
    try:
        # Lấy tất cả bản ghi trong bảng leaf_images
        # Dùng order_by để sắp xếp theo ID cho đẹp
        records = db.query(LeafImage).order_by(LeafImage.image_id).all()
        
        if not records:
            print("❌ Cơ sở dữ liệu hiện tại trống. Chưa có bản ghi nào!")
            return

        print("\n" + "="*80)
        print(f"{'ID':<5} | {'File Name':<30} | {'File Path'}")
        print("="*80)
        
        for record in records:
            # Lấy tên file và đường dẫn (không in vector ra để tránh làm rối màn hình)
            print(f"{record.image_id:<5} | {record.file_name:<30} | {record.file_path}")
            
        print("="*80)
        print(f"✅ Tổng cộng: {len(records)} bản ghi trong CSDL.")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"❌ Có lỗi xảy ra khi truy vấn CSDL: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    view_metadata()
