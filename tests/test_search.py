import sys
import os
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np

# Thêm đường dẫn root dự án vào sys.path để có cấu trúc Module chuẩn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import SessionLocal
from services.search_service import SearchService

def get_image_path():
    """Bật hộp thoại Graphical UI (Tkinter) để chọn file trực quan giống test_extractors."""
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    print("\n[!] Đang mở hộp thoại chọn tệp hình ảnh. (Nếu không thấy, hãy kiểm tra thanh Taskbar bên dưới màn hình!)")
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh lá cây để tìm kiếm",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_path

def display_images_cv2(query_path: str, results: list, title: str):
    """Sử dụng mã OpenCV nội tại để bật Popup ghép các ảnh lại."""
    # 1. Đọc ảnh Query
    buf = np.fromfile(query_path, dtype=np.uint8)
    query_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    
    # Ép kích thước về chung 1 khuôn 300x300
    h, w = 300, 300
    query_img = cv2.resize(query_img, (w, h))
    cv2.putText(query_img, "QUERY", (10, 35), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
    
    images_to_show = [query_img]
    
    # Khâu vạch phân cách
    separator = np.ones((h, 20, 3), dtype=np.uint8) * 200
    images_to_show.append(separator)
    
    # 2. Đọc 5 ảnh Result
    for rank, leaf in enumerate(results, 1):
        file_name = leaf.file_name
        # Tìm đường dẫn cứng của ảnh trên ổ đĩa dựa theo tên (vì trên DB đường dẫn có thể ảo)
        real_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", file_name)
        
        if os.path.exists(real_path):
            img_buf = np.fromfile(real_path, dtype=np.uint8)
            img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (w, h))
        else:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(img, "FILE LOST", (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
            
        cv2.putText(img, f"TOP {rank}", (10, 35), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"ID: {leaf.image_id}", (10, 280), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1)
        
        images_to_show.append(img)
    
    # Ghép chuỗi ngang
    final_output = cv2.hconcat(images_to_show)
    
    # Bật Popup (Mặc định sẽ block màn hình Terminal cho đến khi Tắt Popup)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1400, 320)
    cv2.imshow(title, final_output)
    print(f"\n >>> Đã bật Cửa sổ hiển thị ảnh (Popup) - Nhấn Phím Bất Kỳ vào cửa sổ đó để đóng...")
    cv2.waitKey(0) # Đợi người dùng tắt
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = get_image_path()

    if not image_path:
        print("\nBẠN ĐÃ HỦY CHỌN ẢNH. Kết thúc chương trình.")
        return

    print("\n============================================================")
    print(" BẮT ĐẦU QUÁ TRÌNH TÌM KIẾM ẢNH TƯƠNG ĐỒNG (CBIR SEARCH)")
    print("============================================================")
    print(f" Ảnh đầu vào (Query): {image_path}\n")

    print("[*] Đang truy cập Cơ sở dữ liệu Vector Database...")
    try:
        db = SessionLocal()
        search_engine = SearchService(db)
    except Exception as e:
        print(f"[!] Lỗi kết nối CSDL (Đảm bảo Docker Desktop của bạn đang chạy!): {e}")
        return

    try:
        # PHƯƠNG PHÁP 1: KHOẢNG CÁCH EUCLIDEAN (L2)
        print("[*] Đang gửi ảnh vào Pipeline trích xuất & Tìm theo KHOẢNG CÁCH (L2)...")
        results_l2 = search_engine.process_and_search(image_path, method='l2', limit=5)
        
        print("\n   [KẾT QUẢ TOP 5 - L2 DISTANCE]")
        if not results_l2:
            print("     -> RỖNG: Không tìm thấy ảnh. Có thể bạn chưa chạy nạp dữ liệu (ingest) vào Database.")
        else:
            for rank, leaf in enumerate(results_l2, 1):
                # Lưu ý: Các field name (file_name, image_id) lấy từ bảng db/models.py 
                print(f"     #{rank} | ID: {leaf.image_id} | Name: {leaf.file_name} | URL: {leaf.file_path}")
                
            # --- HIỂN THỊ HÌNH ẢNH L2 ---
            display_images_cv2(image_path, results_l2, title="KET QUA 1. KHOANG CACH EUCLIDEAN (L2)")

        # PHƯƠNG PHÁP 2: ĐỘ TƯƠNG ĐỒNG COSINE
        print("\n------------------------------------------------------------")
        print("[*] Đang quét lại CSDL để so sánh bằng ĐỘ TƯƠNG ĐỒNG GÓC (COSINE)...")
        results_cosine = search_engine.process_and_search(image_path, method='cosine', limit=5)
        
        print("\n   [KẾT QUẢ TOP 5 - COSINE SIMILARITY]")
        if not results_cosine:
            print("     -> RỖNG: Không tìm thấy bản ghi nào.")
        else:
            for rank, leaf in enumerate(results_cosine, 1):
                print(f"     #{rank} | ID: {leaf.image_id} | Name: {leaf.file_name} | URL: {leaf.file_path}")
                
            # --- HIỂN THỊ HÌNH ẢNH COSINE ---
            display_images_cv2(image_path, results_cosine, title="KET QUA 2. DO TUONG DONG GOC (COSINE)")

    except Exception as e:
        print(f"\n[!] CÓ LỖI XẢY RA DO DỮ LIỆU ĐẦU VÀO: {e}")
    finally:
        db.close()
        print("\n============================================================")
        print(" KIỂM THỬ HOÀN TẤT")
        print("============================================================")

if __name__ == "__main__":
    main()
