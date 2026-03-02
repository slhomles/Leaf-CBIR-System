import cv2
import numpy as np
import math

class ShapeExtractor:
    """
    Trích xuất 10 đặc trưng hình dáng từ ảnh lá:
    1. Aspect Ratio
    2. Solidity
    3. Circularity (Compactness)
    4. Convexity
    5. Extent
    6. Eccentricity
    7. Relative Center of Mass
    8. Hu Moment 1
    9. Hu Moment 2
    10. Hu Moment 3
    """
    def __init__(self):
        pass

    def extract(self, image_path: str) -> dict:
        """
        Đọc ảnh và trích xuất các đặc trưng hình dáng.
        Giả định ảnh đầu vào đã cắt nền (nền đen) và object chính là chiếc lá.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")

        # Chuyển ảnh sang grayscale và tạo nhị phân để tìm contour
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Tìm contour của chiếc lá
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("Không tìm thấy chiếc lá trong ảnh.")
            
        # Lấy contour lớn nhất (đáng lẽ là chiếc lá)
        c = max(contours, key=cv2.contourArea)
        
        # 1. Tính toán diện tích và chu vi thực
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if area == 0 or perimeter == 0:
            raise ValueError("Contour không hợp lệ (diện tích hoặc chu vi = 0).")

        # 2. Bounding Rectangle
        x, y, w, h = cv2.boundingRect(c)
        
        # Đặc trưng 1: Aspect Ratio = w / h (hoặc h / w, tùy định nghĩa, thường lấy dài/rộng > 1 hoặc w/h)
        # Để thống nhất, tính min/max để có tỷ lệ < 1
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0

        # Đặc trưng 5: Extent = Phân bố diện tích trong Box
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0

        # 3. Convex Hull
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, True)
        
        # Đặc trưng 2: Solidity
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Đặc trưng 4: Convexity
        convexity = hull_perimeter / perimeter if perimeter > 0 else 0

        # Đặc trưng 3: Circularity
        # Công thức chuẩn: 4 * pi * Area / Perimeter^2
        circularity = (4 * math.pi * area) / (perimeter ** 2)

        # 4. Fit Ellipse cho Eccentricity
        # FitEllipse yêu cầu contour có ít nhất 5 điểm
        eccentricity = 0
        if len(c) >= 5:
            (cx, cy), (MA, ma), angle = cv2.fitEllipse(c)
            # Eccentricity = sqrt(1 - (minor_axis/major_axis)^2)
            if ma > 0:
                a = ma / 2
                b = MA / 2
                eccentricity = math.sqrt(1 - (b**2 / a**2)) if a >= b else math.sqrt(1 - (a**2 / b**2))

        # 5. Moments cho Trọng tâm và Hu Moments
        M = cv2.moments(c)
        
        # Tính Relative Center of Mass (dọc theo trục chính)
        # Tạm tính khoảng cách từ trọng tâm (M['m10']/M['m00'], M['m01']/M['m00']) 
        # So với chiều dài lá lớn nhất (max(w,h))
        relative_center_of_mass = 0
        if M['m00'] != 0:
            cx_m = M['m10'] / M['m00']
            cy_m = M['m01'] / M['m00']
            
            # Gốc lá (cuống) thường nằm ở viền bounding box
            # Tính khoảng cách từ trọng tâm đến một góc hoặc mép (ước tính)
            # Công thức từ doc: Khoảng cách từ cuống lá đến trọng tâm chia cho tổng chiều cao/chiều dài
            # Ta dùng khoảng cách từ cy_m đến đáy (hoặc đỉnh) chia cho h
            # Do ảnh không biết chiều cuống, ta lấy tỷ lệ chiều y của trọng tâm trong bounding box
            relative_center_of_mass = abs(cy_m - y) / h if h > 0 else 0
            
            # Khắc phục trường hợp cuống ở dưới (y + h)
            # relative_center_of_mass có thể được định nghĩa luôn là min(cy_m-y, y+h-cy_m)/h 
            # để biểu diễn độ lệch so với tâm
            relative_center_of_mass = min(abs(cy_m - y), abs(y + h - cy_m)) / h if h > 0 else 0

        # Đặc trưng 8, 9, 10: Hu Moments (1, 2, 3)
        hu_moments = cv2.HuMoments(M)
        # Log transform cho Hu Moments để scale giá trị
        hu = []
        for i in range(3):  # Chỉ lấy 3 giá trị đầu (h1, h2, h3)
            val = hu_moments[i][0]
            if val != 0:
                # Dùng log transform để giá trị dễ so sánh hơn: -sign(h) * log10(abs(h))
                hu.append(-1 * math.copysign(1.0, val) * math.log10(abs(val)))
            else:
                hu.append(0.0)

        # Đóng gói kết quả
        return {
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
            "circularity": circularity,
            "convexity": convexity,
            "extent": extent,
            "eccentricity": eccentricity,
            "relative_center_of_mass": relative_center_of_mass,
            "hu_moment_1": hu[0],
            "hu_moment_2": hu[1],
            "hu_moment_3": hu[2]
        }

if __name__ == "__main__":
    # Test script ad-hoc
    import sys
    
    # Do chạy từ root directory nên để đường dẫn tương đối từ root
    test_image = "d:/tailieuhoctap/Nam4Ky2/Multimedia_database_system/leafsearch_project/data/processed/1001.jpg"
    
    import os
    if not os.path.exists(test_image):
        print(f"Ảnh test không tồn tại: {test_image}")
        sys.exit(1)
        
    extractor = ShapeExtractor()
    try:
        features = extractor.extract(test_image)
        print("Kết quả trích xuất đặc trưng hình dáng:")
        for k, v in features.items():
            print(f"- {k}: {v:.6f}")
    except Exception as e:
        print(f"Lỗi: {e}")
