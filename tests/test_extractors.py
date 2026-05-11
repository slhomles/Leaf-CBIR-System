"""Smoke test 4 extractor (Shape / Color / Texture / Vein) trên 1 ảnh."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from features.extractors.shape import ShapeExtractor
from features.extractors.color import ColorExtractor
from features.extractors.texture import TextureExtractor
from features.extractors.vein_extractor import VeinExtractor

EXTRACTORS = [
    ("Shape    (10-D)", ShapeExtractor()),
    ("Color    (72-D)", ColorExtractor()),
    ("Texture  (38-D)", TextureExtractor()),
    ("Venation (5-D) ", VeinExtractor()),
]


def run_tests(image_path: str) -> None:
    print("=" * 60)
    print(" KIỂM THỬ TRÍCH XUẤT ĐẶC TRƯNG LÁ")
    print("=" * 60)
    print(f" Ảnh : {image_path}\n")

    if not os.path.exists(image_path):
        print(f" [LỖI] Không tìm thấy ảnh: {image_path}")
        return

    for name, extractor in EXTRACTORS:
        print(f" [*] {name}")
        try:
            result = extractor.extract(image_path)
            print(f"     - Số chiều: {len(result)}")
            preview_keys = list(result.keys())[:3]
            for key in preview_keys:
                print(f"     - {key:<28}: {result[key]:.6f}")
            if len(result) > 3:
                print(f"     - ... (+{len(result) - 3} chiều khác)")
        except Exception as exc:
            print(f"     [THẤT BẠI] {exc}")
        print()

    print("=" * 60)
    print(" KIỂM THỬ HOÀN TẤT")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        print("\n [!] Đang mở hộp thoại chọn tệp ảnh...")
        target = filedialog.askopenfilename(
            title="Chọn một bức ảnh lá cây",
            filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All", "*.*")],
        )
        root.destroy()
        if not target:
            print(" [!] Đã huỷ.")
            sys.exit(0)

    run_tests(target)
