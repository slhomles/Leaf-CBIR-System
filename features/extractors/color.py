"""Color extractor — Radial Spatial Histogram HSV (72 chiều).

Phân bố màu HSV theo 3 vòng đồng tâm tính từ trọng tâm lá ra mép:
  - Vòng 1 (Core): khoảng cách (0, R/3]
  - Vòng 2 (Middle): (R/3, 2R/3]
  - Vòng 3 (Periphery): (2R/3, R]
Mỗi vòng tính 2D histogram (H × S) với 8 bin H × 3 bin S = 24 bin.
Tổng: 24 × 3 = 72 chiều.

Đặc tính bất biến:
  - Bất biến xoay: chỉ dựa khoảng cách pixel → centroid.
  - Bất biến tỷ lệ: chuẩn hóa từng vòng theo tổng pixel của vòng đó.
"""

from __future__ import annotations

import cv2
import numpy as np

from features.extractors.base import BaseExtractor


class ColorExtractor(BaseExtractor):
    H_BINS = 8        # Hue: 8 bin trải [0, 180) (OpenCV scale)
    S_BINS = 3        # Saturation: 3 bin trải [0, 256)
    NUM_RINGS = 3     # Core / Middle / Periphery

    BINS_PER_RING = H_BINS * S_BINS                # 24
    TOTAL_DIM = BINS_PER_RING * NUM_RINGS          # 72

    def extract(self, image_path: str) -> dict:
        image = self._read_image(image_path)
        mask = self._get_leaf_mask(image)

        if np.count_nonzero(mask) == 0:
            raise ValueError("Không tìm thấy pixel lá trong ảnh.")

        # 1. Trọng tâm lá từ moments của mask
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            raise ValueError("Mask lá rỗng — không tính được trọng tâm.")
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        # 2. Khoảng cách từng pixel tới trọng tâm
        h, w = mask.shape
        ys, xs = np.indices((h, w), dtype=np.float32)
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

        # 3. Bán kính lớn nhất trong vùng lá
        leaf_dist = dist[mask > 0]
        r_max = float(leaf_dist.max())
        if r_max <= 0:
            raise ValueError("Bán kính lá = 0 — ảnh không hợp lệ.")

        # 4. Mặt nạ 3 vòng đồng tâm
        r1 = r_max / 3.0
        r2 = 2.0 * r_max / 3.0
        r3 = r_max

        leaf_bool = mask > 0
        ring_masks = [
            leaf_bool & (dist >= 0) & (dist < r1),
            leaf_bool & (dist >= r1) & (dist < r2),
            leaf_bool & (dist >= r2) & (dist <= r3 + 1e-6),
        ]

        # 5. HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]

        # 6. Histogram H×S cho từng vòng → flatten → ghép
        result: dict[str, float] = {}
        for r_idx, ring_mask in enumerate(ring_masks, start=1):
            h_pixels = h_ch[ring_mask]
            s_pixels = s_ch[ring_mask]

            if h_pixels.size == 0:
                ring_hist = np.zeros(self.BINS_PER_RING, dtype=np.float64)
            else:
                hist2d, _, _ = np.histogram2d(
                    h_pixels.astype(np.float64),
                    s_pixels.astype(np.float64),
                    bins=[self.H_BINS, self.S_BINS],
                    range=[[0.0, 180.0], [0.0, 256.0]],
                )
                total = hist2d.sum()
                if total > 0:
                    hist2d = hist2d / (total + 1e-10)
                ring_hist = hist2d.flatten()  # (H_BINS, S_BINS) → 24

            for i, value in enumerate(ring_hist):
                result[f"ring{r_idx}_h{i // self.S_BINS}_s{i % self.S_BINS}"] = float(value)

        return result


if __name__ == "__main__":
    import os
    import sys

    test_image = sys.argv[1] if len(sys.argv) > 1 else \
        "d:/tailieuhoctap/Nam4Ky2/Multimedia_database_system/leafsearch_project/data/processed/1001.jpg"

    if not os.path.exists(test_image):
        print(f"Ảnh test không tồn tại: {test_image}")
        sys.exit(1)

    extractor = ColorExtractor()
    features = extractor.extract(test_image)
    print(f"Số chiều: {len(features)} (kỳ vọng 72)")
    print("Tổng các bin (kỳ vọng ≈ 3 — mỗi vòng tổng = 1):",
          f"{sum(features.values()):.4f}")
