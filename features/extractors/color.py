import cv2
import numpy as np
from sklearn.cluster import KMeans
import warnings

from features.extractors.base import BaseExtractor


class ColorExtractor(BaseExtractor):
    """
    Trích xuất đặc trưng màu sắc từ ảnh lá (nền đen).

    Nhóm 1 - Color Moments (9 đặc trưng):
      - Mean, Std, Skewness cho từng kênh H, S, V (normalize về [0,1])

    Nhóm 2 - Color Distribution (137 đặc trưng):
      - Color Histogram HSV: H=8 bins, S=4 bins, V=4 bins → 128 giá trị (normalize L2)
      - Dominant Colors via K-Means (k=3): 3 centroid × 3 kênh → 9 giá trị (normalize về [0,1])

    Nhóm 3 - Color Spatial (256 đặc trưng):
      - Color Coherence Vector (CCV): mỗi bin → (coherent, incoherent) → 128×2 giá trị
    """

    def __init__(self, n_dominant: int = 3, hist_bins: tuple = (8, 4, 4), ccv_threshold: int = 25):
        """
        Args:
            n_dominant: Số màu chủ đạo (K trong K-Means).
            hist_bins: Số bins theo từng kênh (H, S, V).
            ccv_threshold: Số pixel tối thiểu để một vùng được xem là "coherent".
        """
        self.n_dominant = n_dominant
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
    # Nhóm 2: Color Distribution
    # ------------------------------------------------------------------

    def _compute_color_histogram(self, hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Tính Color Histogram trên không gian HSV với mask lá.
        Trả về vector 1D đã normalize L2 (kích thước = H*S*V bins).
        """
        h_bins, s_bins, v_bins = self.hist_bins
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], mask,
            [h_bins, s_bins, v_bins],
            [0, 180, 0, 256, 0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def _compute_dominant_colors(self, hsv_pixels: np.ndarray) -> np.ndarray:
        """
        Tìm dominant colors bằng K-Means, sắp xếp theo số pixel (nhiều → ít).
        Trả về vector phẳng kích thước n_dominant*3 (đã normalize về [0,1]).
        """
        k = self.n_dominant
        pixels_f = hsv_pixels.astype(np.float32)

        if len(pixels_f) < k:
            result = np.zeros(k * 3, dtype=np.float32)
            result[:len(pixels_f) * 3] = pixels_f[:k].flatten()
            return result

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans.fit(pixels_f)

        counts = np.bincount(kmeans.labels_, minlength=k)
        order = np.argsort(-counts)
        centroids = kmeans.cluster_centers_[order]

        # Normalize từng kênh về [0, 1]
        centroids[:, 0] /= 180.0  # H: 0-180
        centroids[:, 1] /= 255.0  # S: 0-255
        centroids[:, 2] /= 255.0  # V: 0-255

        return centroids.flatten()

    # ------------------------------------------------------------------
    # Nhóm 3: Color Spatial — Color Coherence Vector (CCV)
    # ------------------------------------------------------------------

    def _compute_ccv(self, hsv: np.ndarray, mask: np.ndarray):
        """
        Tính Color Coherence Vector.
        Mỗi bin phân loại pixel thành coherent (thuộc vùng lớn) hoặc incoherent.
        Trả về (coherent_array, incoherent_array), mỗi mảng có kích thước total_bins,
        đã normalize theo tổng số pixel lá.
        """
        h_bins, s_bins, v_bins = self.hist_bins
        total_bins = h_bins * s_bins * v_bins

        h_ch, s_ch, v_ch = cv2.split(hsv)

        # Lượng tử hóa từng kênh theo số bins
        h_q = (h_ch.astype(np.float32) / 180.0 * h_bins).astype(np.int32).clip(0, h_bins - 1)
        s_q = (s_ch.astype(np.float32) / 256.0 * s_bins).astype(np.int32).clip(0, s_bins - 1)
        v_q = (v_ch.astype(np.float32) / 256.0 * v_bins).astype(np.int32).clip(0, v_bins - 1)

        bin_img = h_q * (s_bins * v_bins) + s_q * v_bins + v_q
        bin_img[mask == 0] = -1  # đánh dấu vùng nền

        coherent = np.zeros(total_bins, dtype=np.float64)
        incoherent = np.zeros(total_bins, dtype=np.float64)

        for b in range(total_bins):
            bin_mask = (bin_img == b).astype(np.uint8)
            if not np.any(bin_mask):
                continue
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
            for lbl in range(1, num_labels):
                area = stats[lbl, cv2.CC_STAT_AREA]
                if area >= self.ccv_threshold:
                    coherent[b] += area
                else:
                    incoherent[b] += area

        total_leaf_pixels = float(np.sum(mask > 0))
        if total_leaf_pixels > 0:
            coherent /= total_leaf_pixels
            incoherent /= total_leaf_pixels

        return coherent, incoherent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image_path: str) -> dict:
        """
        Đọc ảnh và trích xuất toàn bộ đặc trưng màu sắc.
        Giả định ảnh đầu vào đã xử lý nền đen, chiếc lá là đối tượng chính.

        Returns:
            dict với các keys:
              - color_mean_H/S/V, color_std_H/S/V, color_skewness_H/S/V  (9 values)
              - hist_0 … hist_127                                          (128 values)
              - dominant_0 … dominant_8                                   (9 values)
              - ccv_coherent_0 … ccv_coherent_127                         (128 values)
              - ccv_incoherent_0 … ccv_incoherent_127                     (128 values)
            Tổng cộng: 402 đặc trưng
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

        # 2. Color Histogram (128 values)
        hist = self._compute_color_histogram(hsv, mask)
        for i, v in enumerate(hist):
            result[f'hist_{i}'] = float(v)

        # 3. Dominant Colors (9 values)
        dominant = self._compute_dominant_colors(leaf_pixels)
        for i, v in enumerate(dominant):
            result[f'dominant_{i}'] = float(v)

        # 4. Color Coherence Vector (256 values)
        coherent, incoherent = self._compute_ccv(hsv, mask)
        for i in range(len(coherent)):
            result[f'ccv_coherent_{i}'] = float(coherent[i])
            result[f'ccv_incoherent_{i}'] = float(incoherent[i])

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
        # In nhóm Color Moments
        print("\n--- Color Moments (9 đặc trưng) ---")
        for k in ['color_mean_H', 'color_mean_S', 'color_mean_V',
                  'color_std_H', 'color_std_S', 'color_std_V',
                  'color_skewness_H', 'color_skewness_S', 'color_skewness_V']:
            print(f"  {k}: {features[k]:.6f}")
        # In Dominant Colors
        print("\n--- Dominant Colors (9 đặc trưng) ---")
        for i in range(9):
            print(f"  dominant_{i}: {features[f'dominant_{i}']:.6f}")
        print("\n(Histogram và CCV không in để tránh quá dài)")
    except Exception as e:
        print(f"Lỗi: {e}")
