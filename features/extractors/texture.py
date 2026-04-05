import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from features.extractors.base import BaseExtractor


class TextureExtractor(BaseExtractor):
    """
    Trích xuất đặc trưng kết cấu từ ảnh lá (nền đen).

    1. GLCM / Haralick (4 đặc trưng):
       Contrast, Energy, Homogeneity, Correlation — trung bình trên 4 hướng.

    2. Gabor Filters (40 đặc trưng):
       4 góc × 5 tần số = 20 bộ lọc → Mean + Variance mỗi bộ lọc.

    3. LBP — Local Binary Pattern (10 đặc trưng):
       Histogram của LBP (P=8, R=1, uniform) — normalize về tổng bằng 1.
    """

    # Gabor: 4 hướng, 5 tần số → 20 bộ lọc
    GABOR_ORIENTATIONS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    GABOR_FREQUENCIES  = [0.1, 0.2, 0.3, 0.4, 0.5]   # cycles/pixel

    # LBP
    LBP_P = 8
    LBP_R = 1

    # GLCM
    GLCM_LEVELS  = 64    # quantize grayscale từ 256 → 64 mức
    GLCM_ANGLES  = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _crop_roi(self, gray: np.ndarray, mask: np.ndarray):
        """Crop vùng bounding-box của lá để giảm tính toán không cần thiết."""
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return gray, mask
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return gray[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    # ------------------------------------------------------------------
    # 1. GLCM / Haralick
    # ------------------------------------------------------------------

    def _compute_glcm_features(self, gray: np.ndarray, mask: np.ndarray) -> dict:
        """
        Tính 4 đặc trưng Haralick từ GLCM:
        Contrast, Energy, Homogeneity, Correlation.
        Mỗi đặc trưng là trung bình trên 4 hướng (0°, 45°, 90°, 135°).
        """
        roi, roi_mask = self._crop_roi(gray, mask)

        if roi.size == 0:
            return {
                'glcm_contrast': 0.0, 'glcm_energy': 0.0,
                'glcm_homogeneity': 0.0, 'glcm_correlation': 0.0
            }

        # Đặt pixel nền về 0 và quantize 256 → GLCM_LEVELS mức
        roi = roi.copy()
        roi[roi_mask == 0] = 0
        roi_q = (roi // (256 // self.GLCM_LEVELS)).astype(np.uint8).clip(0, self.GLCM_LEVELS - 1)

        glcm = graycomatrix(
            roi_q,
            distances=[1],
            angles=self.GLCM_ANGLES,
            levels=self.GLCM_LEVELS,
            symmetric=True,
            normed=True
        )

        return {
            'glcm_contrast':     float(np.mean(graycoprops(glcm, 'contrast'))),
            'glcm_energy':       float(np.mean(graycoprops(glcm, 'energy'))),
            'glcm_homogeneity':  float(np.mean(graycoprops(glcm, 'homogeneity'))),
            'glcm_correlation':  float(np.mean(graycoprops(glcm, 'correlation'))),
        }

    # ------------------------------------------------------------------
    # 2. Gabor Filters
    # ------------------------------------------------------------------

    def _compute_gabor_features(self, gray: np.ndarray, mask: np.ndarray) -> dict:
        """
        Áp dụng bank 20 bộ lọc Gabor (4 góc × 5 tần số).
        Với mỗi bộ lọc: lấy Mean và Variance của phản hồi trên vùng lá.
        Trả về 40 đặc trưng.
        """
        gray_f = gray.astype(np.float32)
        features = {}
        idx = 0

        for freq in self.GABOR_FREQUENCIES:
            for theta in self.GABOR_ORIENTATIONS:
                kernel = cv2.getGaborKernel(
                    ksize=(21, 21),
                    sigma=4.0,
                    theta=theta,
                    lambd=1.0 / freq,
                    gamma=0.5,
                    psi=0.0,
                    ktype=cv2.CV_32F
                )
                filtered = cv2.filter2D(gray_f, cv2.CV_32F, kernel)
                leaf_vals = filtered[mask > 0]

                features[f'gabor_mean_{idx}'] = float(np.mean(leaf_vals))
                features[f'gabor_var_{idx}']  = float(np.var(leaf_vals))
                idx += 1

        return features  # 20 bộ lọc × 2 = 40 đặc trưng

    # ------------------------------------------------------------------
    # 3. LBP
    # ------------------------------------------------------------------

    def _compute_lbp_features(self, gray: np.ndarray, mask: np.ndarray) -> dict:
        """
        Tính histogram LBP (P=8, R=1, uniform) trên vùng lá.
        Uniform LBP có P+2 = 10 giá trị duy nhất → histogram 10 bins.
        """
        P, R = self.LBP_P, self.LBP_R
        lbp = local_binary_pattern(gray, P=P, R=R, method='uniform')
        leaf_lbp = lbp[mask > 0]

        n_bins = P + 2  # 10 bins cho P=8
        hist, _ = np.histogram(leaf_lbp, bins=n_bins, range=(0, n_bins), density=True)

        return {f'lbp_{i}': float(v) for i, v in enumerate(hist)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image_path: str) -> dict:
        """
        Đọc ảnh và trích xuất toàn bộ đặc trưng kết cấu.
        Giả định ảnh đầu vào đã xử lý nền đen, chiếc lá là đối tượng chính.

        Returns:
            dict với các keys:
              - glcm_contrast, glcm_energy, glcm_homogeneity, glcm_correlation   (4 values)
              - gabor_mean_0 … gabor_mean_19, gabor_var_0 … gabor_var_19         (40 values)
              - lbp_0 … lbp_9                                                    (10 values)
            Tổng cộng: 54 đặc trưng
        """
        image = self._read_image(image_path)

        mask = self._get_leaf_mask(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if np.sum(mask > 0) == 0:
            raise ValueError("Không tìm thấy pixel lá trong ảnh.")

        result = {}

        # 1. GLCM / Haralick (4 values)
        result.update(self._compute_glcm_features(gray, mask))

        # 2. Gabor Filters (40 values)
        result.update(self._compute_gabor_features(gray, mask))

        # 3. LBP (10 values)
        result.update(self._compute_lbp_features(gray, mask))

        return result


if __name__ == "__main__":
    import sys
    import os

    test_image = "d:/tailieuhoctap/Nam4Ky2/Multimedia_database_system/leafsearch_project/data/processed/1001.jpg"

    if not os.path.exists(test_image):
        print(f"Ảnh test không tồn tại: {test_image}")
        sys.exit(1)

    extractor = TextureExtractor()
    try:
        features = extractor.extract(test_image)
        print(f"Số lượng đặc trưng kết cấu: {len(features)}")

        print("\n--- GLCM / Haralick (4 đặc trưng) ---")
        for k in ['glcm_contrast', 'glcm_energy', 'glcm_homogeneity', 'glcm_correlation']:
            print(f"  {k}: {features[k]:.6f}")

        print("\n--- Gabor (40 đặc trưng, 5 mẫu đầu) ---")
        for i in range(5):
            print(f"  gabor_mean_{i}: {features[f'gabor_mean_{i}']:.6f}  "
                  f"gabor_var_{i}: {features[f'gabor_var_{i}']:.6f}")

        print("\n--- LBP Histogram (10 đặc trưng) ---")
        for i in range(10):
            print(f"  lbp_{i}: {features[f'lbp_{i}']:.6f}")
    except Exception as e:
        print(f"Lỗi: {e}")
