"""Shape extractor — 10 chiều: 7 hình học + 3 Hu moments (log-transformed)."""

import math

import cv2
import numpy as np

from features.extractors.base import BaseExtractor


class ShapeExtractor(BaseExtractor):
    """
    Trích xuất 10 đặc trưng hình dáng từ ảnh lá (giả định nền đen).

    Thứ tự trong vector (bắt buộc — định nghĩa shape_vector VECTOR(10)):
      F01 aspect_ratio          = min(w,h) / max(w,h)
      F02 solidity              = area / convex_area
      F03 circularity           = 4πA / P²
      F04 convexity             = perimeter_hull / perimeter_contour
      F05 extent                = area / (w · h)
      F06 eccentricity          = sqrt(1 - b²/a²) từ fitEllipse
      F07 relative_center_of_mass = min(cy - y_bb, y_bb + h - cy) / h
      F08 hu_1   (log-transformed)
      F09 hu_2   (log-transformed)
      F10 hu_3   (log-transformed)
    """

    def extract(self, image_path: str) -> dict:
        image = self._read_image(image_path)
        mask = self._get_leaf_mask(image)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Không tìm thấy chiếc lá trong ảnh.")

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if area == 0 or perimeter == 0:
            raise ValueError("Contour không hợp lệ (diện tích hoặc chu vi = 0).")

        x, y, w, h = cv2.boundingRect(c)

        # F01 aspect_ratio
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0

        # F05 extent
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0.0

        # F02 solidity, F04 convexity
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, True)
        solidity = area / hull_area if hull_area > 0 else 0.0
        convexity = hull_perimeter / perimeter if perimeter > 0 else 0.0

        # F03 circularity
        circularity = (4.0 * math.pi * area) / (perimeter ** 2)

        # F06 eccentricity
        eccentricity = 0.0
        if len(c) >= 5:
            (_, _), (MA, ma), _ = cv2.fitEllipse(c)
            a = max(MA, ma) / 2.0
            b = min(MA, ma) / 2.0
            if a > 0:
                eccentricity = math.sqrt(max(0.0, 1.0 - (b ** 2) / (a ** 2)))

        # F07 relative_center_of_mass — khoảng cách trọng tâm tới mép gần hơn / chiều cao bbox
        moments = cv2.moments(c)
        relative_center_of_mass = 0.0
        if moments["m00"] != 0 and h > 0:
            cy = moments["m01"] / moments["m00"]
            relative_center_of_mass = min(cy - y, (y + h) - cy) / h

        # F08-F10 Hu moments với log-transform: -sign(h) · log10(|h|)
        hu_raw = cv2.HuMoments(moments).flatten()
        hu = []
        for i in range(3):
            val = float(hu_raw[i])
            if val == 0.0:
                hu.append(0.0)
            else:
                hu.append(-math.copysign(1.0, val) * math.log10(abs(val) + 1e-10))

        return {
            "aspect_ratio": float(aspect_ratio),
            "solidity": float(solidity),
            "circularity": float(circularity),
            "convexity": float(convexity),
            "extent": float(extent),
            "eccentricity": float(eccentricity),
            "relative_center_of_mass": float(relative_center_of_mass),
            "hu_moment_1": hu[0],
            "hu_moment_2": hu[1],
            "hu_moment_3": hu[2],
        }


if __name__ == "__main__":
    import os
    import sys

    test_image = sys.argv[1] if len(sys.argv) > 1 else \
        "d:/tailieuhoctap/Nam4Ky2/Multimedia_database_system/leafsearch_project/data/processed/1001.jpg"

    if not os.path.exists(test_image):
        print(f"Ảnh test không tồn tại: {test_image}")
        sys.exit(1)

    extractor = ShapeExtractor()
    features = extractor.extract(test_image)
    print(f"Số chiều: {len(features)}")
    for k, v in features.items():
        print(f"  {k}: {v:.6f}")
