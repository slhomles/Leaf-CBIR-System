import cv2
import numpy as np
import tempfile
import os

def preprocess_image(image_path: str) -> str:
    """
    Tiền xử lý ảnh động (Dành cho ảnh Query).
    1. Đọc ảnh.
    2. Nếu ảnh nền trắng/sáng -> dùng Otsu Inverse để cắt lấy lá.
    3. Trả về đường dẫn đến 1 bức ảnh mới nền đen đặc (black background).
    """
    # Sửa lỗi OpenCV không đọc được đường dẫn tiếng Việt (Unicode Path Windows)
    buf = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError(f"Không thể giải mã được ảnh từ nguồn: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Otsu Thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Phát hiện màu nền (dựa vào ánh sáng 4 góc)
    h, w = thresh.shape
    corners = [
        thresh[0, 0], thresh[0, w - 1],
        thresh[h - 1, 0], thresh[h - 1, w - 1]
    ]
    # Trung bình góc >= 255/2 thì tức là Nền Trắng/Sáng
    # (Phải Ép kiểu int() để chống lỗi tràn số uint8 của Numpy: 255+255=254)
    if sum(int(c) for c in corners) > (255 * 2):
        thresh = cv2.bitwise_not(thresh)

    # Khử nhiễu hổng (Morphological Closing)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Tìm viền chiếc lá (Contour to nhất)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Nếu xui xẻo không tìm thấy được vật thể, hủy bỏ và trả về gốc (fail-safe)
    if not contours:
        return image_path
        
    c = max(contours, key=cv2.contourArea)

    # Vẽ màng lọc nền đen đặc
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, -1)
    
    # Chặt hình (Apply Mask)
    result = cv2.bitwise_and(img, img, mask=mask)

    # Tùy chọn cực mạnh: Crop khít lại Bounding Box tạo tỉ lệ tỷ chuẩn (giống DB gốc)
    x, y, w_b, h_b = cv2.boundingRect(c)
    # Thêm tí Padding cho lá không chạm viền màn hình (pad=20 pixels)
    pad = 20
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + w_b + pad)
    y2 = min(h, y + h_b + pad)
    
    result = result[y1:y2, x1:x2]

    # Lưu xuống thư mục nháp
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "leafsearch_preprocessed_query.jpg")
    cv2.imwrite(temp_path, result)
    
    return temp_path
