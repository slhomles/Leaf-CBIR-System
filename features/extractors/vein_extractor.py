"""Vein extractor — 5 chiều: density, edge density, thickness, contrast, uniformity.

Tiền xử lý chung (theo spec):
  1. Lấy leaf_mask từ ảnh đã preprocess (nền đen).
  2. Blackhat morphology trên grayscale với kernel ELLIPSE (15×15)
     để làm nổi gân tối trên nền lá sáng.
  3. Threshold blackhat → veins_mask (gân nhị phân).
  4. Canny edge detection trên veins_mask → edges_mask.
  5. Áp leaf_mask để chỉ giữ gân/cạnh nằm trong vùng lá.
"""

from __future__ import annotations

import cv2
import numpy as np

from features.extractors.base import BaseExtractor


class VeinExtractor(BaseExtractor):
    BLACKHAT_KERNEL_SIZE = (15, 15)
    BLACKHAT_THRESHOLD = 15      # ngưỡng để chuyển blackhat → veins_mask nhị phân
    CANNY_LOW = 50
    CANNY_HIGH = 150
    EPS = 1e-6

    def extract(self, image_path: str) -> dict:
        image = self._read_image(image_path)
        leaf_mask = self._get_leaf_mask(image)

        leaf_area = int(np.count_nonzero(leaf_mask))
        if leaf_area == 0:
            return {
                "vein_density": 0.0,
                "vein_edge_density": 0.0,
                "vein_thickness": 0.0,
                "vein_contrast": 0.0,
                "vein_uniformity": 0.0,
            }

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.BLACKHAT_KERNEL_SIZE)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        _, veins_mask = cv2.threshold(blackhat, self.BLACKHAT_THRESHOLD, 255, cv2.THRESH_BINARY)
        veins_inside = cv2.bitwise_and(veins_mask, leaf_mask)

        edges_mask = cv2.Canny(veins_inside, self.CANNY_LOW, self.CANNY_HIGH)
        edges_inside = cv2.bitwise_and(edges_mask, leaf_mask)

        vein_area = int(np.count_nonzero(veins_inside))
        edge_area = int(np.count_nonzero(edges_inside))

        # F121 vein_density
        vein_density = vein_area / leaf_area

        # F122 vein_edge_density
        vein_edge_density = edge_area / leaf_area

        # F123 vein_thickness
        vein_thickness = vein_area / (edge_area + self.EPS)

        # F124, F125 — thống kê cường độ blackhat trên các pixel gân
        blackhat_inside = cv2.bitwise_and(blackhat, blackhat, mask=leaf_mask)
        vein_pixels = blackhat_inside[veins_inside > 0]
        if vein_pixels.size > 0:
            vein_contrast = float(np.mean(vein_pixels))
            vein_uniformity = float(np.std(vein_pixels))
        else:
            vein_contrast = 0.0
            vein_uniformity = 0.0

        return {
            "vein_density": float(vein_density),
            "vein_edge_density": float(vein_edge_density),
            "vein_thickness": float(vein_thickness),
            "vein_contrast": vein_contrast,
            "vein_uniformity": vein_uniformity,
        }


if __name__ == "__main__":
    import os
    import sys

    test_image = sys.argv[1] if len(sys.argv) > 1 else \
        "d:/tailieuhoctap/Nam4Ky2/Multimedia_database_system/leafsearch_project/data/processed/1001.jpg"

    if not os.path.exists(test_image):
        print(f"Ảnh test không tồn tại: {test_image}")
        sys.exit(1)

    extractor = VeinExtractor()
    features = extractor.extract(test_image)
    print(f"Số chiều: {len(features)} (kỳ vọng 5)")
    for k, v in features.items():
        print(f"  {k}: {v:.6f}")
