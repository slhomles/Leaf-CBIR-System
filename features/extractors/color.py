import cv2
import numpy as np

from features.extractors.base import BaseExtractor


class ColorExtractor(BaseExtractor):
    """
    Trích xuất đặc trưng màu sắc từ ảnh lá (nền đen).

    Nhóm 1 - Color Moments (9 đặc trưng):
      - Mean, Std, Skewness cho từng kênh H, S, V (normalize về [0,1])

    Nhóm 2 - CCV Ratio (1 đặc trưng):
      - Tỉ lệ pixel màu thuộc vùng coherent / tổng pixel lá
        (đo mức độ màu sắc phân bố thành vùng đồng nhất)

    Tổng cộng: 10 đặc trưng
    """

    def __init__(self, hist_bins: tuple = (8, 4, 4), ccv_threshold: int = 25):
        """
        Args:
            hist_bins: Số bins theo từng kênh (H, S, V) — dùng để lượng tử hóa CCV.
            ccv_threshold: Số pixel tối thiểu để một vùng được xem là "coherent".
        """
        self.hist_bins = hist_bins
        self.ccv_threshold = ccv_threshold

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _skewness(arr: np.ndarray) -> float:
        """Moment bậc 3 dạng căn bậc ba: cbrt(E[(x - μ)^3])."""
        mean = np.mean(arr)
        third_moment = np.mean((arr - mean) ** 3)
        return float(np.cbrt(third_moment))

    # ------------------------------------------------------------------
    # Nhóm 1: Color Moments
    # ------------------------------------------------------------------

    def _compute_color_moments(self, hsv_pixels: np.ndarray) -> dict:
        """
        Tính Mean, Std, Skewness cho từng kênh H, S, V.
        Giá trị được normalize: H/180, S/255, V/255.
        """
        scales = [180.0, 255.0, 255.0]
        channel_names = ['H', 'S', 'V']
        moments = {}
        for i, (ch, scale) in enumerate(zip(channel_names, scales)):
            vals = hsv_pixels[:, i].astype(np.float64) / scale
            moments[f'color_mean_{ch}'] = float(np.mean(vals))
            moments[f'color_std_{ch}'] = float(np.std(vals))
            moments[f'color_skewness_{ch}'] = self._skewness(vals)
        return moments

    # ------------------------------------------------------------------
    # Nhóm 2: CCV Ratio
    # ------------------------------------------------------------------

    def _compute_ccv_ratio(self, hsv: np.ndarray, mask: np.ndarray) -> float:
        """
        Tính tỉ lệ pixel màu thuộc vùng coherent trên tổng pixel lá.
        Coherent: pixel thuộc vùng liên thông có diện tích >= ccv_threshold.
        """
        h_bins, s_bins, v_bins = self.hist_bins
        total_bins = h_bins * s_bins * v_bins

        h_ch, s_ch, v_ch = cv2.split(hsv)

        h_q = (h_ch.astype(np.float32) / 180.0 * h_bins).astype(np.int32).clip(0, h_bins - 1)
        s_q = (s_ch.astype(np.float32) / 256.0 * s_bins).astype(np.int32).clip(0, s_bins - 1)
        v_q = (v_ch.astype(np.float32) / 256.0 * v_bins).astype(np.int32).clip(0, v_bins - 1)

        bin_img = h_q * (s_bins * v_bins) + s_q * v_bins + v_q
        bin_img[mask == 0] = -1

        total_coherent = 0
        for b in range(total_bins):
            bin_mask = (bin_img == b).astype(np.uint8)
            if not np.any(bin_mask):
                continue
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
            for lbl in range(1, num_labels):
                if stats[lbl, cv2.CC_STAT_AREA] >= self.ccv_threshold:
                    total_coherent += stats[lbl, cv2.CC_STAT_AREA]

        total_leaf_pixels = float(np.sum(mask > 0))
        if total_leaf_pixels == 0:
            return 0.0
        return float(total_coherent / total_leaf_pixels)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image_path: str) -> dict:
        """
        Đọc ảnh và trích xuất đặc trưng màu sắc.
        Giả định ảnh đầu vào đã xử lý nền đen, chiếc lá là đối tượng chính.

        Returns:
            dict với các keys:
              - color_mean_H/S/V, color_std_H/S/V, color_skewness_H/S/V  (9 values)
              - ccv_ratio                                                   (1 value)
            Tổng cộng: 10 đặc trưng
        """
        image = self._read_image(image_path)

        mask = self._get_leaf_mask(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        leaf_pixels = hsv[mask > 0]

        if len(leaf_pixels) == 0:
            raise ValueError("Không tìm thấy pixel lá trong ảnh.")

        result = {}

        # 1. Color Moments (9 values)
        result.update(self._compute_color_moments(leaf_pixels))

        # 2. CCV Ratio (1 value)
        result['ccv_ratio'] = self._compute_ccv_ratio(hsv, mask)

        return result


if __name__ == "__main__":
    import sys
    import os

    test_image = "d:/tailieuhoctap/Nam4Ky2/Multimedia_database_system/leafsearch_project/data/processed/1001.jpg"

    if not os.path.exists(test_image):
        print(f"Ảnh test không tồn tại: {test_image}")
        sys.exit(1)

    extractor = ColorExtractor()
    try:
        features = extractor.extract(test_image)
        print(f"Số lượng đặc trưng màu sắc: {len(features)}")
        print("\n--- Color Features (10 đặc trưng) ---")
        for k, v in features.items():
            print(f"  {k}: {v:.6f}")
    except Exception as e:
        print(f"Lỗi: {e}")
