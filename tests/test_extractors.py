import os
import sys

# Thêm thư mục gốc của project (Leaf-CBIR-System) vào đường dẫn hệ thống
# Điều này giúp Python hiểu việc import các module bên trong thư mục features/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from features.extractors.vein_extractor import extract_vein_features
from features.extractors.symmetry_extractor import extract_symmetry_features

def run_tests(image_path: str):
    print("=" * 60)
    print(f" KIỂM THỬ TRÍCH XUẤT ĐẶC TRƯNG LÁ")
    print("=" * 60)
    print(f" Đường dẫn ảnh : {image_path}\n")

    if not os.path.exists(image_path):
        print(f" [LỖI] Không tìm thấy tệp ảnh vật lý tại đường dẫn trên.")
        print(" Vui lòng kiểm tra lại đường dẫn.")
        return

    # 1. Trích xuất đặc trưng gân lá (Vein)
    print(" [*] Đang xử lý nhóm đặc trưng Gân lá (Vein Features)...")
    try:
        vein_result = extract_vein_features(image_path)
        print("   [KẾT QUẢ]")
        for key, value in vein_result.items():
            print(f"     - {key:<20} : {value}")
    except Exception as e:
        print(f"   [THẤT BẠI] Gặp lỗi : {e}")

    print()

    # 2. Trích xuất đặc trưng đối xứng (Symmetry)
    print(" [*] Đang xử lý nhóm đặc trưng Đối xứng (Symmetry Features)...")
    try:
        sym_result = extract_symmetry_features(image_path)
        print("   [KẾT QUẢ]")
        for key, value in sym_result.items():
            print(f"     - {key:<20} : {value}")
    except Exception as e:
        print(f"   [THẤT BẠI] Gặp lỗi : {e}")

    print("\n" + "=" * 60)
    print(" KIỂM THỬ HOÀN TẤT")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Nếu có gõ đường dẫn sau câu lệnh
        target_image = sys.argv[1]
    else:
        # Khởi chạy cửa sổ chọn tệp GUI bằng Tkinter
        import tkinter as tk
        from tkinter import filedialog
        
        # Ẩn cửa sổ nền của GUI
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True) # Đảm bảo popup nổi lên trên cùng
        
        print("\n [!] Đang mở hộp thoại chọn tệp hình ảnh. (Nếu không thấy, hãy kiểm tra thanh Taskbar bên dưới màn hình!)")
        target_image = filedialog.askopenfilename(
            title="Chọn một bức ảnh lá cây để trích xuất",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All Files", "*.*")]
        )
        
        # Đóng hẳn giao diện ngầm
        root.destroy()
        
        if not target_image:
            print("\n [!] Bạn đã huỷ bỏ lựa chọn. Chương trình kết thúc.\n")
            sys.exit(0)
            
    run_tests(target_image)
