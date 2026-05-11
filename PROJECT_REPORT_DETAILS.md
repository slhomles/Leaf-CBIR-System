# BÁO CÁO CHI TIẾT DỰ ÁN HỆ THỐNG TRUY XUẤT ẢNH LÁ CÂY DỰA TRÊN NỘI DUNG (LEAF-CBIR-SYSTEM)

## 1. Giới thiệu chung về dự án
Dự án **Leaf-CBIR-System** là một hệ thống Truy xuất hình ảnh dựa trên nội dung (Content-Based Image Retrieval - CBIR) chuyên dụng cho việc nhận dạng và tìm kiếm lá cây. Hệ thống cho phép người dùng tải lên một bức ảnh chụp lá cây (ảnh truy vấn - query image) và nhanh chóng tìm ra các bức ảnh lá cây giống nhất trong hệ thống cơ sở dữ liệu dựa trên việc phân tích tính chất vật lý của chiếc lá chứ không phải dựa vào text hoặc nhãn.

**Mục tiêu cốt lõi:**
- Giải quyết bài toán phân loại và tìm kiếm thực vật thông qua hình ảnh bề ngoài.
- Ứng dụng các thuật toán Computer Vision kinh điển kết hợp với cơ sở dữ liệu vector hiện đại để tăng tốc độ truy vấn.

## 2. Kiến trúc và Công nghệ (Tech Stack)
Hệ thống được thiết kế theo kiến trúc Microservices cơ bản:
- **Ngôn ngữ lập trình:** Python 3.11.
- **Backend & API:** Framework **FastAPI** giúp xử lý các endpoint siêu tốc, hỗ trợ xử lý bất đồng bộ, dùng để nhận file ảnh từ người dùng và trả về kết quả tìm kiếm.
- **Xử lý ảnh & Trích xuất đặc trưng:** **OpenCV** (cv2), **NumPy**, **Scikit-image** (skimage), **Scikit-learn** (sklearn).
- **Cơ sở dữ liệu:** **PostgreSQL** kết hợp với extension **pgvector**. Postgres dùng để lưu siêu dữ liệu (đường dẫn, tên lá), trong khi `pgvector` dùng để lưu và tìm kiếm KNN (K-Nearest Neighbors) trên các mảng Vector đặc trưng nhiều chiều.
- **Môi trường chạy DB:** **Docker** và **Docker Compose** dùng để khởi tạo nhanh chóng PostgreSQL container mà không cần cài đặt phức tạp.

## 3. Cấu trúc dữ liệu và Tiền xử lý (Data Preprocessing)
- **Tập dữ liệu (Dataset):** Gồm tối thiểu 500 files ảnh lá cây của nhiều loài sinh học khác nhau, được thu thập từ Kaggle hoặc ảnh chụp thực tế.
- **Tiền xử lý (Preprocessing):** Để đảm bảo tính chính xác, toàn bộ ảnh lá đầu vào (cả ảnh mẫu trong DB và ảnh người dùng upload) đều phải trải qua các bước:
  1. Cắt viền (Bounding box cropping) ôm sát mép chiếc lá.
  2. Xóa phông nền (Background removal) và chuyển phông nền thành màu đen thuần túy để không ảnh hưởng đến các thuật toán màu sắc, hình dáng.
  3. Chuẩn hóa về cùng tỉ lệ khung hình 1:1 và cùng kích thước (VD: 256x256 pixels).

## 4. Logic Trích xuất đặc trưng (Feature Extraction Pipeline)
Hệ thống sử dụng phương pháp "White-box" (có thể giải thích được toán học) thay vì dùng mạng Neural Network "Black-box". Mỗi chiếc lá được bóc tách và tạo thành một **Vector gồm 466 đặc trưng** kết hợp từ 3 nhóm:

### 4.1. Đặc trưng Hình dáng (Shape Features) - 10 đặc trưng
Được xử lý trong `features/extractors/shape.py`, tập trung vào các thông số hình học sau khi tìm được contour của chiếc lá:
1. **Aspect Ratio:** Tỷ lệ khung hình chữ nhật bao quanh lá (giúp phân biệt lá dài như lá sả với lá tròn như lá sen).
2. **Solidity:** Mật độ diện tích thực chia cho diện tích đa giác lồi bao quanh (nhận diện mép xẻ thùy hay mép nguyên).
3. **Circularity (Độ tròn):** Đánh giá mức độ nhẵn của đường viền lá, lá nhiều răng cưa sẽ có độ tròn thấp.
4. **Convexity (Độ lồi):** Phục vụ phát hiện các răng cưa trên mép lá.
5. **Extent (Độ bao phủ):** Nhận diện lá hình tim hoặc hình mũi mác.
6. **Eccentricity (Độ lệch tâm):** Tính toán dựa trên elip tương đương, giúp xác định độ "dẹt" của lá.
7. **Relative Center of Mass (Độ lệch trọng tâm):** Dọc theo trục chính, nhận diện lá phình ở cuống hay phình ở đầu.
8, 9, 10. **Hu Moments (h1, h2, h3):** Các mô-men bất biến với tỷ lệ, phép quay và phép tịnh tiến, mô tả sự phân bố khối lượng/diện tích của lá.

### 4.2. Đặc trưng Màu sắc (Color Features) - 402 đặc trưng
Được xử lý trong `features/extractors/color.py`, sử dụng không gian màu HSV thay vì RGB để mô phỏng tốt hơn cách mắt người cảm nhận màu:
1. **Color Moments (9 đặc trưng):** Mean (Trung bình), Standard Deviation (Độ lệch chuẩn) và Skewness (Độ nghiêng) trên 3 kênh H, S, V.
2. **Color Histogram (128 đặc trưng):** Phân bố tần suất màu sắc với 8 bins cho Hue, 4 bins cho Saturation và 4 bins cho Value.
3. **Dominant Colors (9 đặc trưng):** Thuật toán phân cụm K-Means (K=3) tìm ra 3 màu sắc chủ đạo nhất của chiếc lá.
4. **Color Coherence Vector - CCV (256 đặc trưng):** Bổ sung yếu tố không gian cho màu sắc, phân biệt các màu tụ thành mảng lớn (Coherent) với các màu rải rác (Incoherent).

### 4.3. Đặc trưng Kết cấu (Texture Features) - 54 đặc trưng
Được xử lý trong `features/extractors/texture.py`, tập trung vào các gân lá, đường nét trên bề mặt lá:
1. **GLCM / Haralick (4 đặc trưng):** Ma trận đồng hiện mức xám (Gray Level Co-occurrence Matrix) đo lường Contrast, Energy, Homogeneity, và Correlation.
2. **Gabor Filters (40 đặc trưng):** Bộ lọc phản ứng với các viền nét ở 4 góc độ (0, 45, 90, 135) và 5 tần số khác nhau, mô phỏng cách não bộ nhận diện đường vân lá.
3. **Local Binary Pattern - LBP (10 đặc trưng):** Đánh giá kết cấu vi mô của các điểm ảnh lân cận (bề mặt nhám hay nhẵn).

## 5. Cơ chế Tìm kiếm (Search & Retrieval Mechanism)
Quy trình tìm kiếm một bức ảnh đi theo đường dẫn sau:
1. **Upload:** Client gửi ảnh dạng Binary thông qua API endpoint `/upload` hoặc `/search`.
2. **Extraction Pipeline:** Ảnh đi qua `features/pipeline.py` và lần lượt gọi 3 hàm `extract()` của Shape, Color, Texture. Tất cả kết quả được nối (concatenate) thành 1 vector số thực duy nhất.
3. **Vector Distance Search:** Tại `services/search_service.py`, vector vừa tạo sẽ được gửi câu lệnh SQL xuống CSDL PostgreSQL. `pgvector` sẽ áp dụng thuật toán L2 Distance (Khoảng cách Euclidean) hoặc Cosine Similarity để so sánh vector truy vấn với toàn bộ 500+ vector trong kho dữ liệu.
4. **Response:** Trả về Top $K$ (ví dụ: Top 5) hình ảnh có khoảng cách toán học nhỏ nhất, tức là những chiếc lá có hình dáng, kết cấu và màu sắc tương đồng nhất.

## 6. Tổng kết
Dự án Leaf-CBIR là sự kết hợp hoàn hảo giữa các kiến thức cốt lõi của Computer Vision truyền thống và công nghệ lưu trữ CSDL Vector hiện đại. Hệ thống không sử dụng Deep Learning nên đảm bảo tính dễ giải thích (Explainable AI), tốc độ trích xuất cực nhanh và không yêu cầu Card đồ họa (GPU) để huấn luyện hay chạy model.

---

## 7. SỐ LIỆU THỰC TẾ, CÔNG THỨC VÀ THUẬT TOÁN (ACTUAL METRICS & FORMULAS)

### 7.1. Số liệu Bản ghi Dữ liệu (Data Records)
- **Tổng số lượng ảnh gốc:** 500 files ảnh lá cây.
- **Kích thước CSDL:** 
  - Bảng `Images` (Siêu dữ liệu): 500 bản ghi.
  - Bảng `Features` (Vector đặc trưng): 500 bản ghi.
- **Số chiều Vector (Dimensionality):** Mỗi chiếc lá được chuyển hóa thành một mảng vector một chiều (1D array) chứa chính xác **466 phần tử** (kiểu Float).
- **Tốc độ xử lý (ước tính):** 
  - Trích xuất đặc trưng (1 ảnh): ~0.05 - 0.1 giây/ảnh.
  - Tốc độ truy vấn K-NN trên tập 500 bản ghi: < 0.01 giây.

### 7.2. Các Công thức Toán học Tiêu biểu (Feature Formulas)

**A. Đặc trưng Hình dáng (Shape)**
1. **Aspect Ratio (Tỷ lệ khung hình):** $Aspect Ratio = \frac{min(Width, Height)}{max(Width, Height)}$
2. **Solidity (Độ đặc):** $Solidity = \frac{Area_{leaf}}{Area_{convex\_hull}}$ (Tỷ lệ diện tích lá thật trên diện tích đa giác lồi bao quanh nó).
3. **Circularity (Độ tròn):** $Circularity = \frac{4\pi \times Area_{leaf}}{(Perimeter_{leaf})^2}$
4. **Convexity (Độ lồi):** $Convexity = \frac{Perimeter_{convex\_hull}}{Perimeter_{leaf}}$
5. **Extent (Độ bao phủ):** $Extent = \frac{Area_{leaf}}{Width \times Height}$ (trong Bounding Box).

**B. Đặc trưng Màu sắc (Color Moments)**
Ký hiệu $x_i$ là giá trị màu của pixel thứ $i$, $N$ là tổng số pixel của chiếc lá.
1. **Mean (Trung bình):** $\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$
2. **Standard Deviation (Độ lệch chuẩn):** $\sigma = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2 }$
3. **Skewness (Độ nghiêng):** S = $\sqrt[3]{ \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^3 }$

**C. Đặc trưng Kết cấu (GLCM Texture)**
Ký hiệu $P(i,j)$ là ma trận đồng hiện mức xám.
1. **Contrast (Độ tương phản):** $\sum_{i,j} |i - j|^2 P(i,j)$
2. **Energy (Năng lượng):** $\sum_{i,j} P(i,j)^2$
3. **Homogeneity (Độ đồng nhất):** $\sum_{i,j} \frac{P(i,j)}{1 + |i - j|}$

### 7.3. Thuật toán Tìm kiếm (Search Algorithms)
Hệ thống không duyệt bằng vòng lặp FOR thông thường mà sử dụng thuật toán tìm kiếm lân cận gần nhất tích hợp thẳng trong hệ quản trị CSDL PostgreSQL (`pgvector`):

**1. Thuật toán K-Nearest Neighbors (K-NN) Exact Search:**
Thuật toán này sẽ quét tuần tự (Sequential Scan) qua toàn bộ 500 vector trong CSDL để tính khoảng cách với vector truy vấn (Q) và xếp hạng. Vì dữ liệu mới chỉ có 500 dòng, Exact Search mang lại kết quả đúng 100% mà vẫn đảm bảo thời gian truy xuất theo mili-giây.

**2. Công thức tính Khoảng cách (Distance Metric):**
Hệ thống sử dụng **Khoảng cách Euclidean (L2 Distance)** để đo lường "sự khác biệt" giữa ảnh cần tìm (Query Vector $Q$) và ảnh lưu trữ (Database Vector $V$). Công thức toán học trên không gian 466 chiều:
$$ d(Q, V) = \sqrt{ \sum_{i=1}^{466} (Q_i - V_i)^2 } $$
- Giá trị khoảng cách $d$ càng nhỏ, chứng tỏ hai chiếc lá càng giống nhau (nếu $d = 0$ nghĩa là 2 ảnh giống nhau y hệt).
- Truy vấn SQL `ORDER BY vector_column <-> query_vector LIMIT 5;` sẽ tự động trả về 5 bản ghi có khoảng cách $d$ nhỏ nhất.
