import cv2
import numpy as np

from features.extractors.base import BaseExtractor


class VeinExtractor(BaseExtractor):
    """
    Trích xuất đặc trưng gân lá (Vein Features) gồm 5 tham số:
        - vein_density: Tỷ lệ diện tích bề mặt gân so với phiến lá.
        - vein_edge_density: Mật độ đường cạnh/viền của cấu trúc gân.
        - vein_thickness: Ước lượng độ dày của sợi gân.
        - vein_contrast: Độ nổi tương phản màu sắc gân.
        - vein_uniformity: Độ phân bố đồng đều của mạng gân.
    """

    def extract(self, image_path: str) -> dict:
        img = self._read_image(image_path)

        # BƯỚC 1: Phân tách vùng lá (Leaf Mask) khỏi nền
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        _, leaf_mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Làm liền mặt lá bằng Morphology
        kernel_leaf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_leaf, iterations=3)

        # BƯỚC 2: Trích xuất mảng gân (Veins)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blackhat lọc các chi tiết có cường độ màu tối hơn nền lá (đặc thù gân lá)
        kernel_vein = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_vein)

        # Phân ngưỡng để giới hạn các vết hằn rõ rệt
        _, vein_mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)

        # Dùng Canny để bắt các đường chằng chịt của gân
        edges = cv2.Canny(gray, 50, 150)

        # Bước 3: Chỉ giữ lại gân & viền nằm THUỘC vùng phiến lá
        veins_inside = cv2.bitwise_and(vein_mask, leaf_mask)
        edges_inside = cv2.bitwise_and(edges, leaf_mask)

        leaf_area = np.count_nonzero(leaf_mask)
        vein_area = np.count_nonzero(veins_inside)
        edge_area = np.count_nonzero(edges_inside)

        if leaf_area == 0:
            return {
                "vein_density": 0.0,
                "vein_edge_density": 0.0,
                "vein_thickness": 0.0,
                "vein_contrast": 0.0,
                "vein_uniformity": 0.0
            }

        # [1] Mật độ gân
        vein_density = vein_area / leaf_area

        # [2] Mật độ cạnh gân
        vein_edge_density = edge_area / leaf_area

        # [3] Độ dày sợi gân (Diện tích / Chiều dài cạnh bao quanh)
        vein_thickness = vein_area / (edge_area + 1e-6)  # Tránh chia cho 0

        # Khu vực lấy cường độ gân
        blackhat_inside = cv2.bitwise_and(blackhat, leaf_mask)
        active_veins = blackhat_inside[veins_inside > 0]

        vein_contrast = 0.0
        vein_uniformity = 0.0
        if len(active_veins) > 0:
            # [4] Độ khắc nổi / màu sắc tương phản (Mean pixel intensity of veins)
            vein_contrast = np.mean(active_veins)

            # [5] Mức độ phân bổ đồng đều (Standard Deviation of veins)
            vein_uniformity = np.std(active_veins)

        return {
            "vein_density": round(float(vein_density), 4),
            "vein_edge_density": round(float(vein_edge_density), 4),
            "vein_thickness": round(float(vein_thickness), 4),
            "vein_contrast": round(float(vein_contrast), 4),
            "vein_uniformity": round(float(vein_uniformity), 4)
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        extractor = VeinExtractor()
        print(extractor.extract(sys.argv[1]))
