import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from features.extractors.base import BaseExtractor


class TextureExtractor(BaseExtractor):
    """
    Trích xuất đặc trưng kết cấu từ ảnh lá (nền đen).

    1. GLCM / Haralick (4 đặc trưng):
       Contrast, Energy, Homogeneity, Correlation — trung bình trên 4 hướng.

    2. Gabor Filters (5 đặc trưng):
       5 tần số × trung bình mean-response của 4 hướng → 5 giá trị.

    3. LBP Entropy (1 đặc trưng):
       Shannon entropy của histogram LBP (P=8, R=1, uniform).

    Tổng cộng: 10 đặc trưng
    """

    # Gabor: 4 hướng, 5 tần số
    GABOR_ORIENTATIONS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    GABOR_FREQUENCIES  = [0.1, 0.2, 0.3, 0.4, 0.5]   # cycles/pixel

    # LBP
    LBP_P = 8
    LBP_R = 1

    # GLCM
    GLCM_LEVELS = 64    # quantize grayscale từ 256 → 64 mức
    GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

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
    # 2. Gabor Filters — tổng hợp theo tần số
    # ------------------------------------------------------------------

    def _compute_gabor_features(self, gray: np.ndarray, mask: np.ndarray) -> dict:
        """
        Áp dụng bank Gabor (4 góc × 5 tần số).
        Với mỗi tần số: lấy trung bình mean-response của 4 hướng → 1 giá trị.
        Trả về 5 đặc trưng (1 per frequency).
        """
        gray_f = gray.astype(np.float32)
        features = {}

        for freq_idx, freq in enumerate(self.GABOR_FREQUENCIES):
            mean_responses = []
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
                mean_responses.append(float(np.mean(leaf_vals)))

            features[f'gabor_freq_{freq_idx}'] = float(np.mean(mean_responses))

        return features  # 5 đặc trưng

    # ------------------------------------------------------------------
    # 3. LBP Entropy
    # ------------------------------------------------------------------

    def _compute_lbp_entropy(self, gray: np.ndarray, mask: np.ndarray) -> dict:
        """
        Tính Shannon entropy của histogram LBP (P=8, R=1, uniform).
        Entropy đo mức độ đa dạng micro-texture của bề mặt lá.
        """
        P, R = self.LBP_P, self.LBP_R
        lbp = local_binary_pattern(gray, P=P, R=R, method='uniform')
        leaf_lbp = lbp[mask > 0]

        n_bins = P + 2  # 10 bins cho P=8
        hist, _ = np.histogram(leaf_lbp, bins=n_bins, range=(0, n_bins), density=True)

        # Shannon entropy: H = -sum(p * log2(p)), bỏ qua bins p=0
        p = hist[hist > 0]
        entropy = float(-np.sum(p * np.log2(p)))

        return {'lbp_entropy': entropy}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, image_path: str) -> dict:
        """
        Đọc ảnh và trích xuất đặc trưng kết cấu.
        Giả định ảnh đầu vào đã xử lý nền đen, chiếc lá là đối tượng chính.

        Returns:
            dict với các keys:
              - glcm_contrast, glcm_energy, glcm_homogeneity, glcm_correlation  (4 values)
              - gabor_freq_0 … gabor_freq_4                                      (5 values)
              - lbp_entropy                                                       (1 value)
            Tổng cộng: 10 đặc trưng
        """
        image = self._read_image(image_path)

        mask = self._get_leaf_mask(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if np.sum(mask > 0) == 0:
            raise ValueError("Không tìm thấy pixel lá trong ảnh.")

        result = {}

        # 1. GLCM / Haralick (4 values)
        result.update(self._compute_glcm_features(gray, mask))

        # 2. Gabor per frequency (5 values)
        result.update(self._compute_gabor_features(gray, mask))

        # 3. LBP Entropy (1 value)
        result.update(self._compute_lbp_entropy(gray, mask))

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
        print("\n--- Texture Features (10 đặc trưng) ---")
        for k, v in features.items():
            print(f"  {k}: {v:.6f}")
    except Exception as e:
        print(f"Lỗi: {e}")
