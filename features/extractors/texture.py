"""Texture extractor — 38 chiều: 4 GLCM Haralick + 24 Gabor (12 filter × mean+var) + 10 LBP riu2."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from features.extractors.base import BaseExtractor


class TextureExtractor(BaseExtractor):
    """
    Bố cục vector (38 chiều):
      [0:4]   GLCM Haralick: contrast, energy, homogeneity, correlation
              (trung bình trên 4 hướng [0°, 45°, 90°, 135°])
      [4:28]  Gabor bank: 4 hướng × 3 scale = 12 filter, mỗi filter (mean, var)
              → 24 chiều, sắp xếp [θ0_s0_mean, θ0_s0_var, θ0_s1_mean, θ0_s1_var, ...]
      [28:38] LBP riu2 (P=8, R=1) histogram chuẩn hóa, 10 bin
    """

    GLCM_LEVELS = 256
    GLCM_DISTANCES = [1]
    GLCM_ANGLES = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    GLCM_PROPS = ("contrast", "energy", "homogeneity", "correlation")

    GABOR_THETAS = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    # 3 scale qua thay đổi sigma + bước sóng (lambd) — tham chiếu spec sigma ∈ {1, 2, 3}
    GABOR_SCALES = [
        {"sigma": 2.0, "lambd": 4.0},
        {"sigma": 4.0, "lambd": 8.0},
        {"sigma": 6.0, "lambd": 12.0},
    ]
    GABOR_KSIZE = (21, 21)
    GABOR_GAMMA = 0.5
    GABOR_PSI = 0.0

    LBP_P = 8
    LBP_R = 1
    LBP_BINS = 10  # P + 2 (uniform riu2)

    def extract(self, image_path: str) -> dict:
        image = self._read_image(image_path)
        mask = self._get_leaf_mask(image)
        if np.count_nonzero(mask) == 0:
            raise ValueError("Không tìm thấy pixel lá trong ảnh.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        result: dict[str, float] = {}
        result.update(self._compute_glcm(gray, mask))
        result.update(self._compute_gabor(gray, mask))
        result.update(self._compute_lbp(gray, mask))
        return result

    # ---------------------------------------------------------------- GLCM
    def _compute_glcm(self, gray: np.ndarray, mask: np.ndarray) -> dict:
        roi, roi_mask = self._crop_roi(gray, mask)
        if roi.size == 0:
            return {f"glcm_{p}": 0.0 for p in self.GLCM_PROPS}

        roi = roi.copy()
        roi[roi_mask == 0] = 0  # vùng nền = 0 trong ma trận GLCM

        glcm = graycomatrix(
            roi.astype(np.uint8),
            distances=self.GLCM_DISTANCES,
            angles=self.GLCM_ANGLES,
            levels=self.GLCM_LEVELS,
            symmetric=True,
            normed=True,
        )

        return {
            f"glcm_{prop}": float(np.mean(graycoprops(glcm, prop)))
            for prop in self.GLCM_PROPS
        }

    @staticmethod
    def _crop_roi(gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            return gray, mask
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return gray[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    # ---------------------------------------------------------------- Gabor
    def _compute_gabor(self, gray: np.ndarray, mask: np.ndarray) -> dict:
        gray_f = gray.astype(np.float32)
        leaf_pixels = mask > 0
        if not np.any(leaf_pixels):
            return self._zero_gabor()

        result: dict[str, float] = {}
        for t_idx, theta in enumerate(self.GABOR_THETAS):
            for s_idx, scale in enumerate(self.GABOR_SCALES):
                kernel = cv2.getGaborKernel(
                    self.GABOR_KSIZE,
                    sigma=scale["sigma"],
                    theta=theta,
                    lambd=scale["lambd"],
                    gamma=self.GABOR_GAMMA,
                    psi=self.GABOR_PSI,
                    ktype=cv2.CV_32F,
                )
                filtered = cv2.filter2D(gray_f, cv2.CV_32F, kernel)
                vals = filtered[leaf_pixels]
                result[f"gabor_t{t_idx}_s{s_idx}_mean"] = float(np.mean(vals))
                result[f"gabor_t{t_idx}_s{s_idx}_var"] = float(np.var(vals))
        return result

    def _zero_gabor(self) -> dict:
        out: dict[str, float] = {}
        for t_idx in range(len(self.GABOR_THETAS)):
            for s_idx in range(len(self.GABOR_SCALES)):
                out[f"gabor_t{t_idx}_s{s_idx}_mean"] = 0.0
                out[f"gabor_t{t_idx}_s{s_idx}_var"] = 0.0
        return out

    # ---------------------------------------------------------------- LBP
    def _compute_lbp(self, gray: np.ndarray, mask: np.ndarray) -> dict:
        lbp = local_binary_pattern(gray, P=self.LBP_P, R=self.LBP_R, method="uniform")
        leaf_lbp = lbp[mask > 0]

        if leaf_lbp.size == 0:
            return {f"lbp_bin_{i}": 0.0 for i in range(self.LBP_BINS)}

        bin_edges = np.arange(self.LBP_BINS + 1)
        hist, _ = np.histogram(leaf_lbp, bins=bin_edges)
        total = hist.sum()
        if total > 0:
            hist = hist / (total + 1e-10)

        return {f"lbp_bin_{i}": float(hist[i]) for i in range(self.LBP_BINS)}


if __name__ == "__main__":
    import os
    import sys

    test_image = sys.argv[1] if len(sys.argv) > 1 else \
        "d:/tailieuhoctap/Nam4Ky2/Multimedia_database_system/leafsearch_project/data/processed/1001.jpg"

    if not os.path.exists(test_image):
        print(f"Ảnh test không tồn tại: {test_image}")
        sys.exit(1)

    extractor = TextureExtractor()
    features = extractor.extract(test_image)
    print(f"Số chiều: {len(features)} (kỳ vọng 38)")
    for k, v in features.items():
        print(f"  {k}: {v:.6f}")
