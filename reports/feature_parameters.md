# 📊 Tổng hợp các tham số đặc trưng — Dự án LeafSearch CBIR

> **Tổng số đặc trưng:** 35 tham số
> **Phân nhóm:** Hình dạng (10) · Màu sắc (10) · Kết cấu (10) · Đối xứng (5) · Gân lá (5)
> **File nguồn:** `features/extractors/`

---

## Mục lục

1. [Tiền xử lý chung](#1-tiền-xử-lý-chung)
2. [Nhóm 1 — Đặc trưng Hình dạng (Shape)](#2-nhóm-1--đặc-trưng-hình-dạng-shape--10-tham-số)
3. [Nhóm 2 — Đặc trưng Màu sắc (Color)](#3-nhóm-2--đặc-trưng-màu-sắc-color--10-tham-số)
4. [Nhóm 3 — Đặc trưng Kết cấu (Texture)](#4-nhóm-3--đặc-trưng-kết-cấu-texture--10-tham-số)
5. [Nhóm 4 — Đặc trưng Đối xứng (Symmetry)](#5-nhóm-4--đặc-trưng-đối-xứng-symmetry--5-tham-số)
6. [Nhóm 5 — Đặc trưng Gân lá (Vein)](#6-nhóm-5--đặc-trưng-gân-lá-vein--5-tham-số)
7. [Bảng tóm tắt tổng hợp](#7-bảng-tóm-tắt-tổng-hợp)

---

## 1. Tiền xử lý chung

Trước khi trích xuất đặc trưng, mỗi ảnh lá đi qua pipeline tiền xử lý (xem `features/preprocess.py`):

| Bước | Mô tả   | Kỹ thuật                                                                                                                                           |
| ------ | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1      | Xoá nền | Otsu thresholding + Gaussian Blur (7×7) trên ảnh grayscale → mask nhị phân, sau đó morphological closing (kernel ellipse 5×5) để lấp lỗ |
| 2      | Tách lá | Áp mask lên ảnh BGR gốc → lá giữ nguyên màu, nền = đen                                                                                    |
| 3      | Resize    | Giữ tỷ lệ khung hình, scale về tối đa 256 pixel (chiều dài nhất), rồi padding đen thành 256×256                                        |

**Mask lá (dùng trong các extractor):** Chuyển ảnh sang grayscale → threshold tại mức **10** → mask nhị phân (lá = 255, nền = 0).
*(File: `features/extractors/base.py` → `_get_leaf_mask()`)*

---

## 2. Nhóm 1 — Đặc trưng Hình dạng (Shape) · 10 tham số

> **File nguồn:** `features/extractors/shape.py` — class `ShapeExtractor`

### 2.1. Aspect Ratio (Tỷ lệ khung hình)

- **Ý nghĩa:** Đo tỷ lệ giữa chiều ngắn và chiều dài của bounding rectangle bao quanh lá. Giá trị gần 1 = lá gần vuông/tròn; gần 0 = lá dài, hẹp.
- **Công thức:**

$$
\text{Aspect Ratio} = \frac{\min(w, h)}{\max(w, h)}
$$

- **Các bước tính trong dự án:**
  1. Chuyển ảnh sang grayscale, threshold tại mức 10 để tạo ảnh nhị phân.
  2. Tìm contour ngoài cùng bằng `cv2.findContours(RETR_EXTERNAL)`.
  3. Lấy contour có diện tích lớn nhất (= lá).
  4. Tính bounding rectangle bằng `cv2.boundingRect(c)` → thu được `(x, y, w, h)`.
  5. Tính `min(w, h) / max(w, h)`.

---

### 2.2. Solidity (Độ đặc)

- **Ý nghĩa:** Tỷ lệ diện tích thực của lá so với diện tích vỏ lồi (convex hull). Giá trị gần 1 = mép lá nhẵn, ít khuyết; thấp = mép lá răng cưa, có nhiều phần lõm.
- **Công thức:**

$$
\text{Solidity} = \frac{A_{\text{contour}}}{A_{\text{convex hull}}}
$$

- **Các bước tính trong dự án:**
  1. Tính diện tích thực của contour lá bằng `cv2.contourArea(c)`.
  2. Tính convex hull bằng `cv2.convexHull(c)` - thông 
  3. Tính diện tích convex hull bằng `cv2.contourArea(hull)`.
  4. Chia diện tích contour cho diện tích convex hull.

---

### 2.3. Circularity (Độ tròn / Compactness)

- **Ý nghĩa:** Đo mức độ hình dạng lá gần giống hình tròn. Giá trị = 1.0 = hình tròn hoàn hảo; nhỏ hơn 1 = hình dạng bất quy tắc/dài.
- **Công thức:**

$$
\text{Circularity} = \frac{4\pi \cdot A}{P^2}
$$

trong đó $A$ = diện tích contour, $P$ = chu vi contour.

- **Các bước tính trong dự án:**
  1. Tính diện tích lá bằng `cv2.contourArea(c)`.
  2. Tính chu vi lá bằng `cv2.arcLength(c, True)`.
  3. Áp dụng công thức `4 * π * A / P²`.

---

### 2.4. Convexity (Độ lồi)

- **Ý nghĩa:** Tỷ lệ chu vi convex hull trên chu vi thực của contour. Giá trị gần 1 = mép lá trơn; nhỏ hơn = mép lá phức tạp, gồ ghề, răng cưa.
- **Công thức:**

$$
\text{Convexity} = \frac{P_{\text{convex hull}}}{P_{\text{contour}}}
$$

- **Các bước tính trong dự án:**
  1. Tính convex hull bằng `cv2.convexHull(c)`.
  2. Tính chu vi convex hull bằng `cv2.arcLength(hull, True)`.
  3. Tính chu vi thực bằng `cv2.arcLength(c, True)`.
  4. Chia chu vi convex hull cho chu vi thực.

---

### 2.5. Extent (Mức phủ hộp bao)

- **Ý nghĩa:** Tỷ lệ diện tích thực của lá so với diện tích bounding rectangle. Đo mức lá "lấp đầy" hộp bao chữ nhật.
- **Công thức:**

$$
\text{Extent} = \frac{A_{\text{contour}}}{w \times h}
$$

- **Các bước tính trong dự án:**
  1. Tính diện tích contour lá.
  2. Tính diện tích bounding rectangle = `w × h` (từ `cv2.boundingRect`).
  3. Chia diện tích contour cho diện tích bounding rectangle.

---

### 2.6. Eccentricity (Độ lệch tâm)

- **Ý nghĩa:** Mô tả mức độ kéo dài của hình ellipse khớp (fitted ellipse) với lá. Giá trị = 0 = hình tròn; gần 1 = hình ellipse rất dẹt (lá dài nhọn).
- **Công thức:**

$$
e = \sqrt{1 - \frac{b^2}{a^2}}
$$

trong đó $a$ = bán trục lớn, $b$ = bán trục nhỏ của fitted ellipse.

- **Các bước tính trong dự án:**
  1. Fit ellipse lên contour lá bằng `cv2.fitEllipse(c)` (yêu cầu ≥ 5 điểm).
  2. Thu được bán trục lớn (`MA/2`) và bán trục nhỏ (`ma/2`).
  3. Sắp xếp `a = max, b = min` rồi áp dụng công thức.

---

### 2.7. Relative Center of Mass (Tâm khối tương đối)

- **Ý nghĩa:** Đo vị trí tương đối của trọng tâm lá dọc theo chiều dài. Giá trị gần 0 = trọng tâm gần cuống/đầu lá; gần 0.5 = trọng tâm ở giữa. Phản ánh sự phân bố diện tích lá dọc theo trục dọc.
- **Công thức:**

$$
\text{RCoM} = \frac{\min(|c_y - y|, \; |y + h - c_y|)}{h}
$$

trong đó $c_y = M_{01} / M_{00}$ (toạ độ y của trọng tâm), $(y, h)$ là toạ độ và chiều cao bounding rectangle.

- **Các bước tính trong dự án:**
  1. Tính moment ảnh bằng `cv2.moments(c)`.
  2. Tính tọa độ trọng tâm: `cy_m = M['m01'] / M['m00']`.
  3. Tính khoảng cách nhỏ nhất từ trọng tâm đến hai mép (trên/dưới) bounding rectangle.
  4. Chia cho chiều cao `h`.

---

### 2.8–2.10. Hu Moments (Hu Moment 1, 2, 3)

- **Ý nghĩa:** Hu Moments là bộ 7 moment bất biến (dự án dùng 3 đầu tiên) — bất biến với phép tịnh tiến, co giãn, xoay ảnh. Dùng để so sánh hình dạng tổng quát giữa các chiếc lá.
  - **Hu 1:** Liên quan đến moment bậc 2, đo "spread" / phân tán tổng thể.
  - **Hu 2:** Đo mức độ bất đối xứng theo hai trục chính.
  - **Hu 3:** Nhạy cảm với các đặc trưng phi đối xứng phức tạp hơn (skewness hình dạng).
- **Công thức gốc (Hu, 1962):**

$$
h_1 = \eta_{20} + \eta_{02}
$$

$$
h_2 = (\eta_{20} - \eta_{02})^2 + 4\eta_{11}^2
$$

$$
h_3 = (\eta_{30} - 3\eta_{12})^2 + (3\eta_{21} - \eta_{03})^2
$$

trong đó $\eta_{pq}$ là normalized central moment.

- **Log transform (trong dự án):**

$$
\tilde{h}_i = -\text{sign}(h_i) \cdot \log_{10}(|h_i|)
$$

- **Các bước tính trong dự án:**
  1. Tính moment bằng `cv2.moments(c)`.
  2. Tính Hu moments bằng `cv2.HuMoments(M)`.
  3. Lấy 3 giá trị đầu tiên.
  4. Áp dụng log transform: `-sign(h) * log10(|h|)` để đưa về thang số dễ so sánh.

---

## 3. Nhóm 2 — Đặc trưng Màu sắc (Color) · 10 tham số

> **File nguồn:** `features/extractors/color.py` — class `ColorExtractor`

### 3.1–3.3. Color Mean (Giá trị trung bình kênh H, S, V)

- **Ý nghĩa:** Giá trị trung bình của mỗi kênh màu (Hue, Saturation, Value) trên vùng lá. Phản ánh tông màu chủ đạo, mức bão hòa và độ sáng trung bình.
- **Công thức:**

$$
\mu_c = \frac{1}{N} \sum_{i=1}^{N} \frac{p_i}{S_c}
$$

trong đó $N$ = số pixel lá, $p_i$ = giá trị pixel kênh $c$, $S_c$ = hệ số scale (H: 180, S: 255, V: 255).

- **Các bước tính trong dự án:**
  1. Chuyển ảnh sang không gian HSV bằng `cv2.cvtColor(BGR2HSV)`.
  2. Tạo mask lá (threshold grayscale tại 10).
  3. Lấy pixel lá: `hsv[mask > 0]`.
  4. Normalize mỗi kênh về [0, 1] bằng chia cho scale tương ứng.
  5. Tính `np.mean()` cho từng kênh.

**Output:** `color_mean_H`, `color_mean_S`, `color_mean_V` (3 tham số)

---

### 3.4–3.6. Color Std (Độ lệch chuẩn kênh H, S, V)

- **Ý nghĩa:** Độ lệch chuẩn mỗi kênh màu. Giá trị cao = màu sắc biến đổi nhiều trên bề mặt lá; thấp = màu đồng nhất.
- **Công thức:**

$$
\sigma_c = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\frac{p_i}{S_c} - \mu_c\right)^2}
$$

- **Các bước tính trong dự án:**
  1. Tương tự như Color Mean, sau khi normalize.
  2. Tính `np.std()` cho từng kênh.

**Output:** `color_std_H`, `color_std_S`, `color_std_V` (3 tham số)

---

### 3.7–3.9. Color Skewness (Độ lệch bậc 3 kênh H, S, V)

- **Ý nghĩa:** Moment bậc 3 dạng căn bậc ba. Đo mức bất đối xứng phân bố màu: dương = lệch phải (nhiều pixel có giá trị thấp hơn trung bình), âm = lệch trái.
- **Công thức:**

$$
\gamma_c = \sqrt[3]{\frac{1}{N} \sum_{i=1}^{N} \left(\frac{p_i}{S_c} - \mu_c\right)^3}
$$

- **Các bước tính trong dự án:**
  1. Tính hiệu `(pixel_normalized - mean)` cho mỗi kênh.
  2. Nâng luỹ thừa bậc 3, tính trung bình.
  3. Lấy căn bậc 3 kết quả bằng `np.cbrt()`.

**Output:** `color_skewness_H`, `color_skewness_S`, `color_skewness_V` (3 tham số)

---

### 3.10. CCV Ratio (Tỷ lệ vùng coherent)

- **Ý nghĩa:** Tỷ lệ pixel thuộc vùng **coherent** (vùng liên thông đủ lớn cùng màu) trên tổng pixel lá. Đo mức độ màu sắc phân bố thành vùng đồng nhất lớn. CCV Ratio cao = màu lá phân bố thành mảng lớn rõ rệt; thấp = màu sắc phân tán, nhiều đốm nhỏ.
- **Công thức:**

$$
\text{CCV Ratio} = \frac{\text{Tổng pixel coherent}}{\text{Tổng pixel lá}}
$$

Một pixel được xem là **coherent** nếu nó thuộc một vùng liên thông (connected component) có diện tích ≥ `ccv_threshold` (mặc định = 25 pixel) trong cùng bin màu.

- **Các bước tính trong dự án:**
  1. Lượng tử hóa HSV thành bins (mặc định 8×4×4 = 128 bins):
     - `h_q = H / 180 * 8`, `s_q = S / 256 * 4`, `v_q = V / 256 * 4`
  2. Gộp thành chỉ số bin duy nhất: `bin = h_q × (4×4) + s_q × 4 + v_q`.
  3. Gán pixel nền = -1 (loại trừ khỏi phân tích).
  4. Với từng bin (0→127), tạo mask nhị phân → chạy `cv2.connectedComponentsWithStats(connectivity=8)`.
  5. Cộng dồn diện tích các component ≥ 25 pixel → `total_coherent`.
  6. Chia cho tổng pixel lá.

**Output:** `ccv_ratio` (1 tham số)

---

## 4. Nhóm 3 — Đặc trưng Kết cấu (Texture) · 10 tham số

> **File nguồn:** `features/extractors/texture.py` — class `TextureExtractor`

### 4.1. GLCM Contrast (Độ tương phản)

- **Ý nghĩa:** Đo mức chênh lệch cường độ giữa pixel và pixel lân cận. Giá trị cao = bề mặt lá có kết cấu thô, rõ nét; thấp = bề mặt mịn.
- **Công thức:**

$$
\text{Contrast} = \sum_{i,j} (i - j)^2 \cdot P(i, j)
$$

trong đó $P(i,j)$ là giá trị GLCM đã chuẩn hóa.

- **Các bước tính trong dự án:**
  1. Crop vùng bounding-box lá để giảm tính toán.
  2. Lượng tử grayscale từ 256 → 64 mức: `gray // 4`.
  3. Tính GLCM bằng `skimage.feature.graycomatrix(distances=[1], angles=[0°, 45°, 90°, 135°], levels=64, symmetric=True, normed=True)`.
  4. Tính contrast bằng `graycoprops(glcm, 'contrast')`.
  5. Lấy trung bình trên 4 hướng.

---

### 4.2. GLCM Energy (Năng lượng / Angular Second Moment)

- **Ý nghĩa:** Đo mức đồng nhất kết cấu. Giá trị cao = kết cấu đồng đều, ít biến đổi; thấp = kết cấu phức tạp, ngẫu nhiên.
- **Công thức:**

$$
\text{Energy} = \sum_{i,j} P(i, j)^2
$$

- **Các bước tính trong dự án:**
  1. Tính GLCM như trên.
  2. Tính energy bằng `graycoprops(glcm, 'energy')`.
  3. Trung bình trên 4 hướng.

---

### 4.3. GLCM Homogeneity (Độ đồng nhất)

- **Ý nghĩa:** Đo mức gần gũi của phân bố GLCM với đường chéo chính. Giá trị cao = cường độ pixel thay đổi nhẹ nhàng (kết cấu mịn).
- **Công thức:**

$$
\text{Homogeneity} = \sum_{i,j} \frac{P(i, j)}{1 + |i - j|}
$$

- **Các bước tính trong dự án:**
  1. Tính GLCM như trên.
  2. Tính homogeneity bằng `graycoprops(glcm, 'homogeneity')`.
  3. Trung bình trên 4 hướng.

---

### 4.4. GLCM Correlation (Độ tương quan)

- **Ý nghĩa:** Đo mức tương quan tuyến tính giữa cường độ pixel lân cận. Giá trị cao = kết cấu có cấu trúc lặp lại rõ rệt.
- **Công thức:**

$$
\text{Correlation} = \sum_{i,j} \frac{(i - \mu_i)(j - \mu_j) \cdot P(i, j)}{\sigma_i \cdot \sigma_j}
$$

- **Các bước tính trong dự án:**
  1. Tính GLCM như trên.
  2. Tính correlation bằng `graycoprops(glcm, 'correlation')`.
  3. Trung bình trên 4 hướng.

**Cấu hình GLCM chung:**

| Tham số             | Giá trị                |
| -------------------- | ------------------------ |
| Số mức lượng tử | 64                       |
| Khoảng cách        | 1 pixel                  |
| Các hướng         | 0°, 45°, 90°, 135°   |
| Đối xứng          | Có (`symmetric=True`) |
| Chuẩn hóa          | Có (`normed=True`)    |

---

### 4.5–4.9. Gabor Filter Responses (Phản hồi bộ lọc Gabor) — 5 tham số

- **Ý nghĩa:** Bộ lọc Gabor bắt kết cấu ở các tần số không gian khác nhau. Mỗi tần số tương ứng với mẫu kết cấu có kích thước khác nhau: tần số thấp = mẫu thô/lớn, tần số cao = mẫu mịn/nhỏ. Mỗi tham số là phản hồi trung bình tại một tần số (đã trung bình qua 4 hướng).
- **Công thức kernel Gabor:**

$$
g(x, y; \lambda, \theta, \sigma, \gamma) = \exp\left(-\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2}\right) \cos\left(\frac{2\pi x'}{\lambda}\right)
$$

trong đó $x' = x\cos\theta + y\sin\theta$, $y' = -x\sin\theta + y\cos\theta$, $\lambda = 1/f$.

- **Giá trị đặc trưng cho mỗi tần số:**

$$
\text{gabor\_freq\_k} = \frac{1}{4} \sum_{\theta \in \Theta} \text{mean}\left(|G_{\theta, f_k} * I|\right)_{\text{leaf}}
$$

- **Các bước tính trong dự án:**
  1. Với mỗi tần số $f \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$ cycle/pixel:
     - Với mỗi hướng $\theta \in \{0°, 45°, 90°, 135°\}$:
       - Tạo Gabor kernel: `cv2.getGaborKernel(ksize=(21,21), sigma=4.0, theta, lambd=1/f, gamma=0.5, psi=0)`.
       - Lọc ảnh: `cv2.filter2D(gray, kernel)`.
       - Lấy giá trị trung bình pixel vùng lá (`filtered[mask > 0]`).
     - Trung bình 4 hướng → 1 giá trị cho tần số đó.

**Cấu hình Gabor:**

| Tham số             | Giá trị                            |
| -------------------- | ------------------------------------ |
| Kích thước kernel | 21 × 21                             |
| Sigma                | 4.0                                  |
| Gamma (aspect ratio) | 0.5                                  |
| Psi (phase offset)   | 0.0                                  |
| Tần số             | 0.1, 0.2, 0.3, 0.4, 0.5 cycles/pixel |
| Hướng              | 0°, 45°, 90°, 135°               |

**Output:** `gabor_freq_0`, `gabor_freq_1`, `gabor_freq_2`, `gabor_freq_3`, `gabor_freq_4` (5 tham số)

---

### 4.10. LBP Entropy (Entropy mẫu nhị phân cục bộ)

- **Ý nghĩa:** Shannon entropy của histogram LBP. Đo mức độ đa dạng micro-texture (kết cấu vi mô) bề mặt lá. Entropy cao = bề mặt có nhiều loại micro-pattern; thấp = kết cấu đơn điệu/đồng nhất.
- **Công thức LBP (uniform, P=8, R=1):**

$$
\text{LBP}_{P,R}(x_c, y_c) = \sum_{p=0}^{P-1} s(g_p - g_c) \cdot 2^p
$$

trong đó $s(x) = 1$ nếu $x \geq 0$, ngược lại $s(x) = 0$; $g_c$ = cường độ pixel trung tâm, $g_p$ = cường độ pixel lân cận.

- **Shannon Entropy:**

$$
H = -\sum_{k} p_k \cdot \log_2(p_k)
$$

trong đó $p_k$ = tỷ lệ pixel rơi vào bin $k$ (histogram density).

- **Các bước tính trong dự án:**
  1. Tính LBP bằng `skimage.feature.local_binary_pattern(gray, P=8, R=1, method='uniform')`.
  2. Lấy giá trị LBP tại vùng lá: `lbp[mask > 0]`.
  3. Tạo histogram density với 10 bins (P + 2 = 10 cho uniform LBP).
  4. Tính Shannon entropy: `-Σ(p * log₂(p))`, bỏ qua bins có p = 0.

**Output:** `lbp_entropy` (1 tham số)

---

## 5. Nhóm 4 — Đặc trưng Đối xứng (Symmetry) · 5 tham số

> **File nguồn:** `features/extractors/symmetry_extractor.py` — class `SymmetryExtractor`

**Chuẩn bị chung (PCA tìm trục tự nhiên):**

1. Tạo mask lá bằng Otsu trên kênh Saturation (HSV) + morphological closing (kernel ellipse 7×7, 3 lần lặp).
2. Thu tập hợp tọa độ pixel lá → centroid hình học.
3. Tính ma trận hiệp phương sai → phân tích eigenvalue/eigenvector (`np.linalg.eigh`).
4. Eigenvector có eigenvalue lớn nhất = **trục dọc** (trục chính/principal axis) → vuông góc = **trục ngang**.
5. Chiếu pixel lên trục dọc và ngang.

---

### 5.1. Longitudinal Asymmetry (Bất đối xứng dọc)

- **Ý nghĩa:** Mức chênh lệch diện tích giữa nửa trái và nửa phải lá (chia theo trục dọc). Giá trị gần 0 = lá đối xứng; cao = lá lệch trái/phải.
- **Công thức:**

$$
\text{Long. Asymmetry} = \frac{|A_{\text{trái}} - A_{\text{phải}}|}{A_{\text{tổng}}}
$$

- **Các bước tính:**
  1. Chiếu pixel lên trục ngang (perp_axis): `proj_transverse = pts_centered @ perp_axis`.
  2. Đếm pixel có giá trị chiếu > 0 (bên trái) và < 0 (bên phải).
  3. Tính độ chênh tuyệt đối chia cho tổng pixel lá.

---

### 5.2. Transverse Asymmetry (Bất đối xứng ngang)

- **Ý nghĩa:** Mức chênh lệch diện tích giữa nửa ngọn và nửa gốc lá (chia theo trục ngang). Giá trị gần 0 = phần trên/dưới lá cân bằng.
- **Công thức:**

$$
\text{Trans. Asymmetry} = \frac{|A_{\text{ngọn}} - A_{\text{gốc}}|}{A_{\text{tổng}}}
$$

- **Các bước tính:**
  1. Chiếu pixel lên trục dọc: `proj_longitudinal = pts_centered @ principal_axis`.
  2. Đếm pixel có giá trị chiếu > 0 (ngọn) và < 0 (gốc).
  3. Tính độ chênh tuyệt đối chia cho tổng.

---

### 5.3. Center of Mass Shift (Độ lệch trọng tâm)

- **Ý nghĩa:** Khoảng cách Euclidean giữa trọng tâm hình học (geometric centroid) và trọng tâm khối (center of mass từ image moments). Giá trị lớn = phân bố pixel lá không đều, trọng tâm khối bị lệch.
- **Công thức:**

$$
\text{CMS} = \sqrt{(x_{\text{geom}} - x_{\text{mass}})^2 + (y_{\text{geom}} - y_{\text{mass}})^2}
$$

trong đó $(x_{\text{mass}}, y_{\text{mass}}) = (M_{10}/M_{00}, \; M_{01}/M_{00})$.

- **Các bước tính:**
  1. Tính centroid hình học = trung bình cộng tọa độ pixel lá.
  2. Tính moments bằng `cv2.moments(mask)` → center of mass $(M_{10}/M_{00}, M_{01}/M_{00})$.
  3. Tính khoảng cách Euclidean giữa hai trọng tâm.

---

### 5.4. Length Asymmetry (Bất đối xứng chiều dài)

- **Ý nghĩa:** Độ chênh khoảng cách từ trọng tâm đến ngọn lá so với từ trọng tâm đến gốc lá. Giá trị gần 0 = trọng tâm nằm giữa lá; cao = trọng tâm lệch về phía ngọn hoặc gốc.
- **Công thức:**

$$
\text{Length Asym.} = \frac{|d_{\text{max}} - d_{\text{min}}|}{d_{\text{max}} + d_{\text{min}}}
$$

trong đó $d_{\text{max}}$ = khoảng chiếu cực đại dọc trục chính > 0, $d_{\text{min}}$ = |giá trị cực tiểu|.

- **Các bước tính:**
  1. Lấy giá trị chiếu cực đại và cực tiểu trên trục dọc.
  2. Tính độ chênh tuyệt đối chia cho tổng chiều dài.

---

### 5.5. Width Asymmetry (Bất đối xứng chiều rộng)

- **Ý nghĩa:** Độ chênh giữa sải cánh trái và cánh phải lá. Giá trị gần 0 = hai cánh lá đều; cao = một bên rộng hơn bên kia.
- **Công thức:**

$$
\text{Width Asym.} = \frac{|w_{\text{trái}} - w_{\text{phải}}|}{w_{\text{trái}} + w_{\text{phải}}}
$$

- **Các bước tính:**
  1. Lấy giá trị chiếu cực đại (trái) và |cực tiểu| (phải) trên trục ngang.
  2. Tính độ chênh tuyệt đối chia cho tổng chiều rộng.

---

## 6. Nhóm 5 — Đặc trưng Gân lá (Vein) · 5 tham số

> **File nguồn:** `features/extractors/vein_extractor.py` — class `VeinExtractor`

**Chuẩn bị chung (trích xuất vùng gân):**

1. Tạo mask lá bằng Otsu trên kênh Saturation + morphological closing (kernel ellipse 7×7, 3 lần lặp).
2. Trích gân bằng **Blackhat morphology**: `cv2.MORPH_BLACKHAT` với kernel rect 5×5 → bắt chi tiết tối (gân) trên nền sáng (thịt lá).
3. Threshold blackhat tại mức 15 → mask gân (`vein_mask`).
4. Phát hiện cạnh bằng **Canny** (threshold 50–150) → `edges`.
5. Cắt gân và cạnh chỉ trong vùng lá: `bitwise_and` với leaf_mask.

---

### 6.1. Vein Density (Mật độ gân)

- **Ý nghĩa:** Tỷ lệ diện tích gân so với diện tích phiến lá. Giá trị cao = mạng gân dày đặc; thấp = gân thưa.
- **Công thức:**

$$
\text{Vein Density} = \frac{A_{\text{vein}}}{A_{\text{leaf}}}
$$

- **Các bước tính:**
  1. Đếm pixel gân trong vùng lá: `np.count_nonzero(veins_inside)`.
  2. Đếm tổng pixel lá: `np.count_nonzero(leaf_mask)`.
  3. Chia lấy tỷ lệ.

---

### 6.2. Vein Edge Density (Mật độ cạnh gân)

- **Ý nghĩa:** Tỷ lệ pixel cạnh (phát hiện bằng Canny) trong vùng lá. Phản ánh mức độ phức tạp/chi tiết của mạng gân.
- **Công thức:**

$$
\text{Vein Edge Density} = \frac{A_{\text{edges}}}{A_{\text{leaf}}}
$$

- **Các bước tính:**
  1. Đếm pixel cạnh trong vùng lá: `np.count_nonzero(edges_inside)`.
  2. Chia cho tổng pixel lá.

---

### 6.3. Vein Thickness (Độ dày gân)

- **Ý nghĩa:** Ước lượng độ dày trung bình của sợi gân bằng tỷ số diện tích gân trên chiều dài cạnh bao. Giá trị cao = gân dày; thấp = gân mảnh.
- **Công thức:**

$$
\text{Vein Thickness} = \frac{A_{\text{vein}}}{A_{\text{edges}} + \epsilon}
$$

trong đó $\epsilon = 10^{-6}$ để tránh chia cho 0.

- **Các bước tính:**
  1. Lấy diện tích gân (`vein_area`) và diện tích cạnh (`edge_area`).
  2. Chia: `vein_area / (edge_area + 1e-6)`.

---

### 6.4. Vein Contrast (Độ tương phản gân)

- **Ý nghĩa:** Cường độ pixel trung bình vùng gân trong ảnh blackhat. Giá trị cao = gân nổi bật, tương phản mạnh so với thịt lá; thấp = gân mờ.
- **Công thức:**

$$
\text{Vein Contrast} = \frac{1}{N_v} \sum_{i \in \text{veins}} B(i)
$$

trong đó $B(i)$ = cường độ pixel blackhat tại vị trí gân, $N_v$ = số pixel gân.

- **Các bước tính:**
  1. Áp mask lá lên ảnh blackhat: `cv2.bitwise_and(blackhat, leaf_mask)`.
  2. Lấy giá trị pixel tại vị trí gân: `blackhat_inside[veins_inside > 0]`.
  3. Tính `np.mean()`.

---

### 6.5. Vein Uniformity (Độ đồng đều gân)

- **Ý nghĩa:** Độ lệch chuẩn cường độ pixel vùng gân. Giá trị thấp = gân phân bố đều, đồng nhất; cao = có gân đậm nhạt xen kẽ.
- **Công thức:**

$$
\text{Vein Uniformity} = \sqrt{\frac{1}{N_v} \sum_{i \in \text{veins}} (B(i) - \bar{B})^2}
$$

- **Các bước tính:**
  1. Lấy giá trị pixel vùng gân (như Vein Contrast).
  2. Tính `np.std()`.

---

## 7. Bảng tóm tắt tổng hợp

| #  | Nhóm    | Tham số                | Key trong code              | Miền giá trị      |
| -- | -------- | ----------------------- | --------------------------- | -------------------- |
| 1  | Shape    | Aspect Ratio            | `aspect_ratio`            | [0, 1]               |
| 2  | Shape    | Solidity                | `solidity`                | [0, 1]               |
| 3  | Shape    | Circularity             | `circularity`             | [0, 1]               |
| 4  | Shape    | Convexity               | `convexity`               | [0, 1]               |
| 5  | Shape    | Extent                  | `extent`                  | [0, 1]               |
| 6  | Shape    | Eccentricity            | `eccentricity`            | [0, 1)               |
| 7  | Shape    | Relative Center of Mass | `relative_center_of_mass` | [0, 0.5]             |
| 8  | Shape    | Hu Moment 1             | `hu_moment_1`             | ℝ (log-transformed) |
| 9  | Shape    | Hu Moment 2             | `hu_moment_2`             | ℝ (log-transformed) |
| 10 | Shape    | Hu Moment 3             | `hu_moment_3`             | ℝ (log-transformed) |
| 11 | Color    | Mean Hue                | `color_mean_H`            | [0, 1]               |
| 12 | Color    | Mean Saturation         | `color_mean_S`            | [0, 1]               |
| 13 | Color    | Mean Value              | `color_mean_V`            | [0, 1]               |
| 14 | Color    | Std Hue                 | `color_std_H`             | [0, 1]               |
| 15 | Color    | Std Saturation          | `color_std_S`             | [0, 1]               |
| 16 | Color    | Std Value               | `color_std_V`             | [0, 1]               |
| 17 | Color    | Skewness Hue            | `color_skewness_H`        | ℝ                   |
| 18 | Color    | Skewness Saturation     | `color_skewness_S`        | ℝ                   |
| 19 | Color    | Skewness Value          | `color_skewness_V`        | ℝ                   |
| 20 | Color    | CCV Ratio               | `ccv_ratio`               | [0, 1]               |
| 21 | Texture  | GLCM Contrast           | `glcm_contrast`           | [0, +∞)             |
| 22 | Texture  | GLCM Energy             | `glcm_energy`             | [0, 1]               |
| 23 | Texture  | GLCM Homogeneity        | `glcm_homogeneity`        | [0, 1]               |
| 24 | Texture  | GLCM Correlation        | `glcm_correlation`        | [-1, 1]              |
| 25 | Texture  | Gabor Freq 0 (f=0.1)    | `gabor_freq_0`            | ℝ                   |
| 26 | Texture  | Gabor Freq 1 (f=0.2)    | `gabor_freq_1`            | ℝ                   |
| 27 | Texture  | Gabor Freq 2 (f=0.3)    | `gabor_freq_2`            | ℝ                   |
| 28 | Texture  | Gabor Freq 3 (f=0.4)    | `gabor_freq_3`            | ℝ                   |
| 29 | Texture  | Gabor Freq 4 (f=0.5)    | `gabor_freq_4`            | ℝ                   |
| 30 | Texture  | LBP Entropy             | `lbp_entropy`             | [0, log₂(10)]       |
| 31 | Symmetry | Longitudinal Asymmetry  | `longitudinal_asymmetry`  | [0, 1]               |
| 32 | Symmetry | Transverse Asymmetry    | `transverse_asymmetry`    | [0, 1]               |
| 33 | Symmetry | Center of Mass Shift    | `center_of_mass_shift`    | [0, +∞) (pixel)     |
| 34 | Symmetry | Length Asymmetry        | `length_asymmetry`        | [0, 1]               |
| 35 | Symmetry | Width Asymmetry         | `width_asymmetry`         | [0, 1]               |
| 36 | Vein     | Vein Density            | `vein_density`            | [0, 1]               |
| 37 | Vein     | Vein Edge Density       | `vein_edge_density`       | [0, 1]               |
| 38 | Vein     | Vein Thickness          | `vein_thickness`          | [0, +∞)             |
| 39 | Vein     | Vein Contrast           | `vein_contrast`           | [0, 255]             |
| 40 | Vein     | Vein Uniformity         | `vein_uniformity`         | [0, 255]             |

---

> **Tổng cộng: 40 tham số** = Shape(10) + Color(10) + Texture(10) + Symmetry(5) + Vein(5)

*Tài liệu được tạo tự động từ mã nguồn dự án LeafSearch CBIR.*
