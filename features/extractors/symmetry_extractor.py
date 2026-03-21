import cv2
import numpy as np
import math

def extract_symmetry_features(image_path: str) -> dict:
    """
    Trích xuất đặc trưng tính đối xứng của lá (Symmetry Features).
    Đầu vào: Đường dẫn của ảnh lá.
    Đầu ra: Dictionary chứa 2 tham số:
        - asymmetry_index: Mức độ lệch tự nhiên của 2 nửa lá (tính theo trục PCA).
        - center_of_mass_shift: Khoảng cách lệch (pixel) giữa điểm cân bằng hình học và trọng lượng điểm màu.
    """
    # Đọc ảnh linh hoạt hỗ trợ Unicode Windows
    buf = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    # Tạo mask cô lập phiến lá
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Nếu hình không thể chiết xuất lá, trả về giá trị thấp nhất
    points = np.column_stack(np.where(mask > 0))  # Lấy array (row, col)
    if points.shape[0] < 10:
        return {"asymmetry_index": 0.0, "center_of_mass_shift": 0.0}

    # Tính tâm hình học Geometric Centroid
    pts_float = points.astype(np.float64)
    centroid_geom = pts_float.mean(axis=0)  # ra 2 tọa độ (cy, cx)
    pts_centered = pts_float - centroid_geom

    # Áp dụng PCA (Phân tích Thành phần Chính) tìm chiều dài sinh học của chiếc lá dẫu nó xoay
    cov = np.cov(pts_centered.T)
    evals, evecs = np.linalg.eigh(cov)
    principal_axis = evecs[:, np.argmax(evals)]  
    
    # Tìm trục vuông góc với chiều thân lá để phân tách bên phải & bên trái
    perp_axis = np.array([-principal_axis[1], principal_axis[0]])
    proj_perp = pts_centered @ perp_axis

    # Tính diện tích 2 bên
    area_left = np.sum(proj_perp > 0)
    area_right = np.sum(proj_perp < 0)
    total = area_left + area_right

    # Tính Index (0 = hoàn toàn đối xứng, chạy dần đến 1 sẽ bị lệch bẹp 1 bên)
    ai = abs(area_left - area_right) / total if total != 0 else 0.0

    # Phân tích Center of Mass Shift
    cy_geom, cx_geom = centroid_geom[0], centroid_geom[1]
    
    # Mật độ phân bố Moment khối lượng trên cấu trúc màu của Mask nguyên bản
    M = cv2.moments(mask)
    cx_mass = M["m10"] / M["m00"] if M["m00"] != 0 else cx_geom
    cy_mass = M["m01"] / M["m00"] if M["m00"] != 0 else cy_geom

    # Khoảng cách Euclide (Pixel distance)
    cms = math.sqrt((cx_geom - cx_mass) ** 2 + (cy_geom - cy_mass) ** 2)

    return {
        "asymmetry_index": round(float(ai), 4),
        "center_of_mass_shift": round(float(cms), 4)
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(extract_symmetry_features(sys.argv[1]))
