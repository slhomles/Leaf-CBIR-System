import os
import cv2
import glob
import numpy as np
from tqdm import tqdm

# Thêm đường dẫn để gọi module nội bộ của dự án
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from features.preprocess import preprocess_image

def rebuild_dataset():
    input_dir = os.path.join(project_root, "data", "raw")
    output_dir = os.path.join(project_root, "data", "processed")
    
    # Tạo thư mục gốc nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Quét tất cả file Raw
    search_path = os.path.join(input_dir, "*.*")
    all_files = glob.glob(search_path)
    
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_paths = [f for f in all_files if f.lower().endswith(valid_extensions)]
    
    print(f"\n============================================================")
    print(f" [*] Tái Tái Thiết Lập Dữ Liệu (REBUILD DATASET)")
    print(f" [*] Đã tìm thấy {len(image_paths)} ảnh Thô (RAW).")
    print(f"============================================================")
    print("Bắt đầu chạy tiến trình lướt qua toàn bộ ảnh, ÁP DỤNG CẮT XÉN CHUẨN MỰC Bounding Box...\n")
    
    for img_path in tqdm(image_paths, desc="Rebuilding Dataset"):
        try:
            # Tận dụng luôn module tiền xử lý ta viết lúc nãy (tự tạo ra 1 file rác temp)
            temp_output = preprocess_image(img_path)
            
            # Tính toán đường dẫn mới
            filename = os.path.basename(img_path)
            save_path = os.path.join(output_dir, filename)
            
            # Dùng numpy đọc lại file rác từ preprocess_image (để né lỗi Tiếng Việt)
            buf = np.fromfile(temp_output, dtype=np.uint8)
            processed_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            
            # Ghi đè vào data/processed
            cv2.imencode('.jpg', processed_img)[1].tofile(save_path)
            
            # Dọn vệ sinh: xóa file rác để khỏi đầy ổ cứng
            if temp_output != img_path and os.path.exists(temp_output):
                os.remove(temp_output)
                
        except Exception as e:
            print(f"\n[Lỗi] Không thể xử lý ảnh {img_path}: {e}")

if __name__ == "__main__":
    rebuild_dataset()
    print(f"\n============================================================")
    print(" HOÀN TẤT ĐỒNG BỘ DỮ LIỆU. ")
    print(" File Dữ liệu Siêu Chuẩn đã sẵn sàng trong thư mục: data/processed.")
    print(" BƯỚC TIẾP THEO: Vui lòng chạy lệnh Nạp Cơ sở dữ liệu Vector:")
    print(" >>> python scripts/ingest.py")
    print(f"============================================================")
