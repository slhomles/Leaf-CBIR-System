import os
import cv2
import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import được features
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.preprocess import preprocess

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def main():
    # Đảm bảo thư mục processed tồn tại
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    
    # Lấy danh sách ảnh
    image_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.jpg')]
    total = len(image_files)
    print(f"Starting to process {total} images from {RAW_DIR} to {PROCESSED_DIR}...")
    
    for i, filename in enumerate(image_files, 1):
        img_path = os.path.join(RAW_DIR, filename)
        out_path = os.path.join(PROCESSED_DIR, filename)
        
        try:
            # Tiền xử lý (Xóa nền -> Resize & Padding)
            processed_img = preprocess(img_path)
            
            # Ghi file (sử dụng imencode để an toàn với đường dẫn có dấu)
            is_success, buffer = cv2.imencode(".jpg", processed_img)
            if is_success:
                buffer.tofile(out_path)
            else:
                print(f"Encode failed for image: {img_path}")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
        if i % 100 == 0:
            print(f"Processed {i}/{total} images...")

    print(f"Done! All images have been saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    main()
