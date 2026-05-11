# BÁO CÁO BÀI TẬP LỚN: HỆ CSDL LƯU TRỮ VÀ TÌM KIẾM ẢNH LÁ CÂY

---

## MỤC LỤC
**CHƯƠNG I: DỮ LIỆU VÀ THUỘC TÍNH**
1. Yêu cầu dữ liệu
2. Thuộc tính của ảnh lá
   - Hình dạng lá
   - Màu sắc lá
   - Tính đối xứng
   - Kết cấu và Gân lá
3. Tổng quan quy trình trích rút
   - Sơ đồ khối
   - Quy trình

**CHƯƠNG II: TIỀN XỬ LÝ ẢNH**
1. Quá trình xử lý ảnh
2. Các bước thực hiện
3. Pseudocode

**CHƯƠNG III: TRÍCH RÚT ĐẶC TRƯNG (40 ĐẶC TRƯNG TỔNG HỢP)**
1. Đặc trưng về màu sắc (10 đặc trưng)
2. Đặc trưng về hình dạng (10 đặc trưng)
3. Đặc trưng về tính đối xứng (5 đặc trưng)
4. Đặc trưng về gân lá (5 đặc trưng)
5. Đặc trưng về kết cấu (Texture - 10 đặc trưng)

**CHƯƠNG IV: SO SÁNH ĐẶC TRƯNG VÀ TÌM KIẾM ẢNH**
1. Chuẩn hóa đặc trưng số (Z-Score Normalization)
2. Tính khoảng cách giữa ảnh truy vấn và ảnh trong Database
3. Truy vấn ảnh lá: Giao diện Streamlit

---

## CHƯƠNG I: DỮ LIỆU VÀ THUỘC TÍNH

### 1. Yêu cầu dữ liệu
Hệ thống sử dụng bộ dữ liệu nhằm tối ưu hóa quá trình nhận diện và tìm kiếm:
- **Số lượng:** Bộ dữ liệu bao gồm nhiều loại lá cây khác nhau, chia vào các folder riêng biệt để phân loại. Số lượng ảnh đảm bảo tính đa dạng về hình dạng, màu sắc, kết cấu.
- **Định dạng ảnh:** `.jpg` / `.png`
- **Kích thước ảnh sau xử lý:** Cắt và đệm viền (Padding) để đạt tỷ lệ khung hình chuẩn.
- **Tỷ lệ khung hình và bố cục vật thể:**
  + Lá cây nằm chính giữa khung hình.
  + Mọi ảnh đều được xử lý để cắt sát viền lá (Bounding Box Crop) và đưa về nền đen tĩnh nhằm dễ dàng trích xuất pixel.

### 2. Thuộc tính của ảnh lá
Dựa vào mã nguồn thực tế, 5 khía cạnh thuộc tính quan trọng nhất được trích xuất (tổng cộng 40 chiều đặc trưng):

**Hình dạng lá (10 thông số)**
- **Lý do lựa chọn:** Hình dáng lá mang tính ổn định sinh học cao nhất để phân biệt giống loài (ví dụ: hình tim, hình bầu dục, lá xẻ thùy).
- **Giá trị thông tin:** Aspect Ratio (tỷ lệ dài/rộng), Solidity (độ đặc), Circularity (độ tròn), Convexity, Eccentricity, và 3 tham số bất biến Hu Moments.

**Màu sắc lá (10 thông số)**
- **Lý do lựa chọn:** Phản ánh rõ tình trạng lá (non, già, sâu bệnh) hoặc loài cây có sắc tố đặc trưng.
- **Giá trị thông tin:** Trích xuất trong không gian HSV để tách biệt ánh sáng (Value). Tính các mô-men (Mean, Std, Skewness) và tỉ lệ màu phân bố tập trung (CCV Ratio).

**Tính đối xứng (5 thông số)**
- **Lý do lựa chọn:** Lá cây thường có tính đối xứng tự nhiên qua trục gân chính.
- **Giá trị thông tin:** Độ lệch tâm hình học so với trọng tâm, sự khác biệt nửa trái/phải, ngọn/đáy.

**Gân lá (5 thông số)**
- **Lý do lựa chọn:** Cấu trúc gân là "dấu vân tay" của thực vật.
- **Giá trị thông tin:** Mật độ gân, độ dày của sợi gân, độ tương phản của vân lá, phát hiện thông qua thuật toán Morphology Blackhat và Canny.

**Kết cấu - Texture (10 thông số)**
- **Lý do lựa chọn:** Bề mặt phiến lá (nhám, trơn bóng, phủ lông) tạo ra đặc trưng kết cấu cục bộ.
- **Giá trị thông tin:** Contrast, Energy, Correlation (từ GLCM), Tần số Gabor và Entropy của LBP.

### 3. Tổng quan quy trình trích rút
- **Sơ đồ khối:** `[Ảnh gốc] -> [Tiền xử lý (Otsu & Crop)] -> [5 Extractors (40 vectors)] -> [Z-Score Normalization] -> [Tìm kiếm (Cosine/L2/L1)]`
- **Quy trình:** Khi người dùng upload ảnh, ảnh sẽ được xóa nền bằng Otsu Threshold, cắt sát viền. Sau đó 5 module trích xuất chạy tuần tự tạo ra mảng 40 phần tử, được chuẩn hóa và tính khoảng cách để tìm top 5 ảnh giống nhất.

---

## CHƯƠNG II: TIỀN XỬ LÝ ẢNH

### 1. Quá trình xử lý ảnh
Tiền xử lý có vai trò tách chiếc lá ra khỏi bối cảnh nhiễu. Dự án dùng thuật toán **Otsu Thresholding** kết hợp xử lý Morphology thay vì FloodFill.

### 2. Các bước thực hiện
1. Đọc ảnh và chuyển sang ảnh xám (Grayscale), áp dụng Gaussian Blur để khử nhiễu.
2. Dùng `cv2.threshold` kết hợp `cv2.THRESH_OTSU` để tìm ngưỡng phân tách lá/nền tự động.
3. Nhận diện nền trắng/sáng bằng cách kiểm tra pixel 4 góc; nếu nền sáng thì đảo ngược mask.
4. Tìm Contour lớn nhất bao quanh chiếc lá.
5. Cắt ảnh bằng Bounding Box và đệm thêm 20 pixel (Padding) để ảnh không bị mất mép.

### 3. Pseudocode (Otsu Bounding Crop)
```python
FUNCTION Tien_Xu_Ly_Anh(image_path):
    img = READ_IMAGE(image_path)
    gray = CONVERT_GRAYSCALE(img)
    blurred = GAUSSIAN_BLUR(gray, 5x5)
    
    # 1. Otsu Threshold tự động tìm ngưỡng
    thresh_mask = OTSU_THRESHOLD(blurred)
    
    # 2. Phát hiện và đảo nền nếu nền sáng
    IF is_white_background(thresh_mask):
        thresh_mask = BITWISE_NOT(thresh_mask)
        
    # 3. Lọc nhiễu Morphological Close
    thresh_mask = MORPHOLOGY_CLOSE(thresh_mask)
    
    # 4. Cắt Bounding Box
    main_contour = FIND_LARGEST_CONTOUR(thresh_mask)
    x, y, w, h = GET_BOUNDING_BOX(main_contour)
    
    # 5. Cắt ảnh với padding 20 pixel
    cropped_leaf = img[y-20 : y+h+20, x-20 : x+w+20]
    
    RETURN cropped_leaf
```

---

## CHƯƠNG III: TRÍCH RÚT ĐẶC TRƯNG

Hệ thống có 5 file Extractor, trả về tổng cộng chính xác 40 giá trị số thực (Float).

### 1. Đặc trưng về màu sắc (`color.py`)
- Phân tích màu trong không gian HSV, loại bỏ nền đen. Tính 9 giá trị Color Moments (Mean, Standard Deviation, Skewness) cho H, S, V. Thêm 1 giá trị CCV (Color Coherence Vector Ratio).
```python
FUNCTION Trich_Rut_Mau(hsv_image, mask):
    leaf_pixels = hsv_image[mask > 0]
    H, S, V = SPLIT(leaf_pixels)
    
    # Moment bậc 1, 2, 3
    color_features = []
    FOR channel IN [H, S, V]:
        color_features.APPEND( MEAN(channel), STD(channel), SKEWNESS(channel) )
        
    # Tỉ lệ pixel đồng nhất (CCV Ratio)
    ccv_ratio = COMPUTE_COHERENT_PIXELS(hsv_image) / TOTAL_PIXELS(leaf_pixels)
    color_features.APPEND(ccv_ratio)
    
    RETURN color_features # (10 features)
```

### 2. Đặc trưng về hình dạng (`shape.py`)
- Dựa trên Contour và Bounding Box.
```python
FUNCTION Trich_Rut_Hinh_Dang(contour, bbox):
    area, perimeter = CALCULATE_AREA_AND_PERIMETER(contour)
    
    aspect_ratio = MIN(w, h) / MAX(w, h)
    solidity = area / HULL_AREA(contour)
    circularity = (4 * PI * area) / (perimeter^2)
    convexity = HULL_PERIMETER(contour) / perimeter
    extent = area / (bbox_w * bbox_h)
    
    ellipse = FIT_ELLIPSE(contour)
    eccentricity = CALCULATE_ECCENTRICITY(ellipse)
    
    hu_moments = CALCULATE_HU_MOMENTS(contour) # Lấy 3 giá trị đầu (Log transform)
    
    RETURN [aspect_ratio, solidity, circularity, convexity, extent, eccentricity, relative_center, hu1, hu2, hu3]
```

### 3. Đặc trưng về tính đối xứng (`symmetry_extractor.py`)
- Phân tích bằng ma trận hiệp phương sai (PCA) tìm trục chính của lá.
```python
FUNCTION Trich_Rut_Doi_Xung(leaf_mask):
    # Tìm trục chính và trục phụ bằng Eigenvectors
    principal_axis, perp_axis = GET_EIGENVECTORS(leaf_mask)
    
    # Chiếu tọa độ pixel lên trục để chia nửa lá
    proj_longitudinal, proj_transverse = PROJECT_POINTS()
    
    long_asymmetry = ABS(AREA_LEFT - AREA_RIGHT) / TOTAL_AREA
    trans_asymmetry = ABS(AREA_TOP - AREA_BOTTOM) / TOTAL_AREA
    
    RETURN [long_asymmetry, trans_asymmetry, center_of_mass_shift, length_asym, width_asym]
```

### 4. Đặc trưng về gân lá (`vein_extractor.py`)
- Sử dụng Morphological Blackhat để lọc vân tối, và Canny để tìm viền gân.
```python
FUNCTION Trich_Rut_Gan(gray_image, mask):
    # Blackhat lấy nếp gấp tối màu
    blackhat = MORPH_BLACKHAT(gray_image)
    vein_mask = THRESHOLD(blackhat, 15)
    
    # Canny lấy viền gân
    edges = CANNY(gray_image, 50, 150)
    
    vein_density = AREA(vein_mask) / AREA(mask)
    vein_edge_density = AREA(edges) / AREA(mask)
    vein_thickness = AREA(vein_mask) / AREA(edges)
    
    RETURN [vein_density, vein_edge_density, vein_thickness, vein_contrast, vein_uniformity]
```

### 5. Đặc trưng về kết cấu (`texture.py`)
- Sử dụng 3 công cụ: Ma trận đồng hiện mức xám (GLCM), Bộ lọc Gabor (Gabor Filters), và LBP (Local Binary Pattern).
```python
FUNCTION Trich_Rut_Ket_Cau(gray_image):
    # GLCM 4 góc
    glcm_features = [Contrast, Energy, Homogeneity, Correlation]
    
    # Gabor 5 tần số (trung bình 4 góc)
    gabor_features = [Freq_0.1, Freq_0.2, Freq_0.3, Freq_0.4, Freq_0.5]
    
    # LBP Entropy
    lbp_hist = LBP_HISTOGRAM(gray_image, P=8, R=1)
    lbp_entropy = SHANNON_ENTROPY(lbp_hist)
    
    RETURN glcm_features + gabor_features + [lbp_entropy] # (10 features)
```

---

## CHƯƠNG IV: SO SÁNH ĐẶC TRƯNG VÀ TÌM KIẾM ẢNH

### 1. Chuẩn hóa đặc trưng số (Z-Score Normalization)
Hệ thống sử dụng **Z-Score Normalization** thay vì Min-Max. Giá trị Mean ($\mu$) và Std ($\sigma$) được tính toán và lưu trước (`data/zscore_params.npz`).
- **Pseudocode:**
```python
FUNCTION Z_Score_Normalize(vector_40d, db_mean_40d, db_std_40d):
    normalized_vector = []
    FOR i = 0 TO 39:
        x = vector_40d[i]
        mu = db_mean_40d[i]
        sigma = db_std_40d[i]
        
        # Áp dụng Z-Score: z = (x - mean) / std
        z_val = (x - mu) / (sigma + 1e-8) # Tránh chia cho 0
        normalized_vector.APPEND(z_val)
        
    RETURN normalized_vector
```

### 2. Tính khoảng cách giữa ảnh truy vấn và ảnh trong Database
Ứng dụng Streamlit (`app.py`) hỗ trợ 3 phép đo khoảng cách:
- **Cosine Distance:** Đo góc giữa 2 vector, dùng để bù trừ khi độ chênh lệch ánh sáng lớn.
- **Euclidean (L2 Distance):** Khoảng cách đường thẳng, hoạt động rất tốt sau khi dữ liệu đã qua Z-Score.
- **Manhattan (L1 Distance):** Tổng độ lệch tuyệt đối, ít bị ảnh hưởng bởi giá trị ngoại lai (outliers).

### 3. Truy vấn ảnh lá: Giao diện Streamlit
Người dùng tải ảnh lên, hệ thống sẽ thực thi luồng sau:
1. Gọi `preprocess_image` để cắt sát lá và nền đen.
2. Gọi `extract_all` trong `pipeline.py` để ra vector 40 chiều.
3. Gọi `normalize` (Z-Score).
4. Sử dụng SQLAlchemy kết nối PostgreSQL (`pgvector`), thực hiện câu truy vấn `ORDER BY vector_column <-> query_vector LIMIT 5`.
5. Hiển thị 5 bức ảnh có độ chênh lệch nhỏ nhất ra màn hình (`show_results`).
