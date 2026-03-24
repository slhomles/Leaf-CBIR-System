import cv2
import numpy as np
import math

def extract_symmetry_features(image_path: str) -> dict:
    """
    Trích xuất đặc trưng tính đối xứng của lá (Symmetry Features) gồm 5 tham số.
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
        return {
            "longitudinal_asymmetry": 0.0,
            "transverse_asymmetry": 0.0,
            "center_of_mass_shift": 0.0,
            "length_asymmetry": 0.0,
            "width_asymmetry": 0.0
        }

    # Tính tâm hình học Geometric Centroid
    pts_float = points.astype(np.float64)
    centroid_geom = pts_float.mean(axis=0)  # ra 2 tọa độ (cy, cx)
    pts_centered = pts_float - centroid_geom

    # Lập ma trận hiệp phương sai tìm cấu trúc tự nhiên
    cov = np.cov(pts_centered.T)
    evals, evecs = np.linalg.eigh(cov)
    
    # Eigenvector eigenvalue cao nhất = Trục dọc tự nhiên (Dọc chiều dài lá)
    principal_axis = evecs[:, np.argmax(evals)]  
    
    # Trục phụ vuông góc với trục dọc (Chiều Ngang lá)
    perp_axis = np.array([-principal_axis[1], principal_axis[0]])
    
    # Biến đổi toạ độ: Trục dọc ngăn lá thành Ngọn và Đáy.
    proj_longitudinal = pts_centered @ principal_axis
    # Trục ngang ngăn lá thành Cánh Trái và Cánh Phải.
    proj_transverse = pts_centered @ perp_axis

    total_area = float(points.shape[0])

    # [1] Longitudinal Asymmetry: Khác biệt Nửa Trái - Nửa Phải
    area_left = np.sum(proj_transverse > 0)
    area_right = np.sum(proj_transverse < 0)
    long_ai = abs(area_left - area_right) / total_area if total_area != 0 else 0.0

    # [2] Transverse Asymmetry: Khác biệt Nửa Ngọn - Nửa Đáy
    area_top = np.sum(proj_longitudinal > 0)
    area_bottom = np.sum(proj_longitudinal < 0)
    trans_ai = abs(area_top - area_bottom) / total_area if total_area != 0 else 0.0

    # [3] Center of Mass Shift
    cy_geom, cx_geom = centroid_geom[0], centroid_geom[1]
    M = cv2.moments(mask)
    cx_mass = M["m10"] / M["m00"] if M["m00"] != 0 else cx_geom
    cy_mass = M["m01"] / M["m00"] if M["m00"] != 0 else cy_geom
    # Khoảng cách Euclide (Pixel distance)
    cms = math.sqrt((cx_geom - cx_mass) ** 2 + (cy_geom - cy_mass) ** 2)

    # [4] Độ lệch sinh học Chiều Dài (Từ trọng tâm bắn ra đỉnh và đáy lá có bằng nhau?)
    max_height = np.max(proj_longitudinal)
    min_height = abs(np.min(proj_longitudinal))
    total_height = max_height + min_height
    length_ai = abs(max_height - min_height) / total_height if total_height != 0 else 0.0

    # [5] Độ sải ngực của lá Chiều Rộng (Trái thò ra dài hơn hay Phải thò ra dài hơn?)
    max_width_left = np.max(proj_transverse)
    max_width_right = abs(np.min(proj_transverse))
    total_width = max_width_left + max_width_right
    width_ai = abs(max_width_left - max_width_right) / total_width if total_width != 0 else 0.0

    return {
        "longitudinal_asymmetry": round(float(long_ai), 4),
        "transverse_asymmetry": round(float(trans_ai), 4),
        "center_of_mass_shift": round(float(cms), 4),
        "length_asymmetry": round(float(length_ai), 4),
        "width_asymmetry": round(float(width_ai), 4)
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(extract_symmetry_features(sys.argv[1]))
