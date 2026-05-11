# CHI TIẾT ĐẶC TRƯNG GÂN LÁ VÀ ĐỘ ĐỐI XỨNG

Tài liệu này mô tả chi tiết cách tính toán, công thức và ý nghĩa của các đặc trưng Gân lá và Độ đối xứng được triển khai trong hệ thống Leaf-CBIR.

---

## 1. ĐẶC TRƯNG TÍNH ĐỐI XỨNG (SYMMETRY FEATURES)

Tính đối xứng phản ánh sự cân đối của phiến lá qua các trục sinh học. Hệ thống sử dụng Phân tích thành phần chính (PCA) để xác định trục dọc tự nhiên của lá trước khi tính toán.

### 1.1. Cách tính toán chung
1.  **Xác định trục:** Sử dụng ma trận hiệp phương sai của các pixel thuộc lá để tìm các Vector riêng (Eigenvectors). Vector có trị riêng lớn nhất xác định **trục dọc** (chiều dài), vector còn lại xác định **trục ngang** (chiều rộng).
2.  **Chia vùng:** Trục dọc chia lá thành nửa trái và nửa phải. Trục ngang chia lá thành nửa ngọn (top) và nửa đáy (bottom).

### 1.2. Các đại lượng chi tiết

#### a. Bất đối xứng dọc (Longitudinal Asymmetry)
-   **Công thức:** $S_{long} = \frac{|Area_{left} - Area_{right}|}{Area_{total}}$
-   **Ý nghĩa:** Đo mức độ chênh lệch diện tích giữa hai cánh lá bên trái và bên phải trục gân chính. Giá trị càng gần 0 thì lá càng đối xứng trái-phải.

#### b. Bất đối xứng ngang (Transverse Asymmetry)
-   **Công thức:** $S_{trans} = \frac{|Area_{top} - Area_{bottom}|}{Area_{total}}$
-   **Ý nghĩa:** Đo sự khác biệt giữa phần ngọn lá và phần gốc lá. Đặc trưng này giúp phân biệt các loại lá hình tim (gốc to) với các loại lá hình thoi.

#### c. Độ lệch trọng tâm (Center of Mass Shift)
-   **Công thức:** $CMS = \sqrt{(x_{geom} - x_{mass})^2 + (y_{geom} - y_{mass})^2}$
-   **Ý nghĩa:** Tính khoảng cách giữa tâm hình học (trung bình tọa độ các pixel) và trọng tâm vật lý (tính theo mô-men ảnh). Nếu lá phân bổ mô-men không đều, hai tâm này sẽ lệch nhau.

#### d. Bất đối xứng chiều dài (Length Asymmetry)
-   **Công thức:** $L_{asym} = \frac{|h_{max} - h_{min}|}{h_{max} + h_{min}}$
-   **Ý nghĩa:** So sánh khoảng cách từ trọng tâm đến đỉnh lá so với khoảng cách đến cuống lá dọc theo trục chính.

#### e. Bất đối xứng chiều rộng (Width Asymmetry)
-   **Công thức:** $W_{asym} = \frac{|w_{left} - w_{right}|}{w_{left} + w_{right}}$
-   **Ý nghĩa:** Đo độ "sải" của hai bên cánh lá. Một bên cánh lá to hơn bên kia sẽ làm tăng giá trị này.

---

## 2. ĐẶC TRƯNG GÂN LÁ (VEIN FEATURES)

Gân lá được trích xuất bằng các kỹ thuật hình thái học (Morphology) để tách các đường nét tinh vi trên bề mặt lá.

### 2.1. Cách tính toán chung
1.  **Lọc Blackhat:** Sử dụng phép toán $Blackhat = Close(Img) - Img$ để làm nổi bật các chi tiết tối màu (gân lá) trên nền phiến lá sáng hơn.
2.  **Phát hiện biên Canny:** Sử dụng thuật toán Canny để bắt các đường cạnh chằng chịt của mạng lưới gân.

### 2.2. Các đại lượng chi tiết

#### a. Mật độ gân lá (Vein Density)
-   **Công thức:** $V_{dens} = \frac{Pixels_{vein}}{Pixels_{leaf}}$
-   **Ý nghĩa:** Tỷ lệ diện tích bề mặt gân so với toàn bộ diện tích phiến lá. Lá có hệ thống gân dày đặc sẽ có mật độ cao.

#### b. Mật độ cạnh gân (Vein Edge Density)
-   **Công thức:** $V_{edge} = \frac{Pixels_{edge}}{Pixels_{leaf}}$
-   **Ý nghĩa:** Đo độ chằng chịt của mạng lưới gân. Càng nhiều gân nhỏ li ti, tổng chu vi (cạnh) của gân càng lớn.

#### c. Độ dày sợi gân (Vein Thickness)
-   **Công thức:** $V_{thick} = \frac{Pixels_{vein}}{Pixels_{edge}}$
-   **Ý nghĩa:** Tỉ lệ giữa diện tích gân và chu vi biên gân. Giá trị này giúp phân biệt gân lá to, rõ (như lá bàng) với gân lá mảnh (như lá liễu).

#### d. Độ tương phản gân (Vein Contrast)
-   **Công thức:** $V_{cont} = \text{Mean}(\text{Blackhat\_Intensity})$
-   **Ý nghĩa:** Đo độ "nổi" của gân lá so với bề mặt phiến lá. Gân càng hằn rõ và sâu thì độ tương phản màu sắc càng cao.

#### e. Độ đồng đều gân (Vein Uniformity)
-   **Công thức:** $V_{unif} = \text{StdDev}(\text{Blackhat\_Intensity})$
-   **Ý nghĩa:** Đo mức độ phân bổ của gân. Giá trị này thấp cho thấy gân lá phân bổ đều khắp bề mặt, giá trị cao cho thấy gân chỉ tập trung ở một số vùng nhất định (như gân chính).
