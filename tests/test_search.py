import sys
import os
import tkinter as tk
from tkinter import filedialog

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

    except Exception as e:
        print(f"\n[!] CÓ LỖI XẢY RA DO DỮ LIỆU ĐẦU VÀO: {e}")
    finally:
        db.close()
        print("\n============================================================")
        print(" KIỂM THỬ HOÀN TẤT")
        print("============================================================")

if __name__ == "__main__":
    main()
