import cv2
import numpy as np

def extract_vein_features(image_path: str) -> dict:
    """
    Trích xuất đặc trưng gân lá (Vein Features).
    Đầu vào: Nhận vào đường dẫn nguyên bản của ảnh.
    Đầu ra: Dictionary chứa 2 tham số:
        - vein_density: Tỷ lệ diện tích bề mặt gân so với phiến lá.
        - vein_edge_density: Mật độ đường cạnh/viền của cấu trúc gân.
    """
    # Đọc ảnh hỗ trợ tốt đường dẫn Unicode trên Windows
    buf = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

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
    
    vein_area = np.count_nonzero(veins_inside)
    edge_area = np.count_nonzero(edges_inside)
    leaf_area = np.count_nonzero(leaf_mask)
    
    vein_density = vein_area / leaf_area if leaf_area > 0 else 0.0
    edge_density = edge_area / leaf_area if leaf_area > 0 else 0.0

    return {
        "vein_density": round(float(vein_density), 4),
        "vein_edge_density": round(float(edge_density), 4)
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(extract_vein_features(sys.argv[1]))
