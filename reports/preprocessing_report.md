# Báo cáo: Quy trình tiền xử lý ảnh lá (Image Preprocessing)

Quy trình tiền xử lý ảnh trong dự án được thiết kế nhằm mục đích tách biệt phần lá ra khỏi nền, đồng thời chuẩn hóa kích thước ảnh để đảm bảo tính nhất quán trước khi đưa vào các thuật toán trích xuất đặc trưng (hình dáng, màu sắc, kết cấu và gân lá). 

Toàn bộ logic tiền xử lý được triển khai trong tập tin `features/preprocess.py`. Dưới đây là tổng hợp chi tiết các bước thực hiện.

## Bước 1: Đọc ảnh đầu vào
- **Phương pháp**: Sử dụng `numpy.fromfile` kết hợp `cv2.imdecode` để tải ảnh từ bộ nhớ đệm (buffer).
- **Mục đích**: Cách tiếp cận này giúp xử lý an toàn các đường dẫn (paths) chứa ký tự Unicode (như tiếng Việt), tránh các lỗi không mong muốn từ hàm `cv2.imread` mặc định của OpenCV.

## Bước 2: Phân đoạn và tách nền lá (`segment_leaf`)
Quá trình này nhằm xác định vị trí của lá và loại bỏ các chi tiết nền không cần thiết.
1. **Chuyển đổi ảnh xám**: Ảnh BGR ban đầu được chuyển về thang độ xám (Grayscale).
2. **Làm mờ (Gaussian Blur)**: Áp dụng bộ lọc Gaussian với kích thước kernel `7x7`. Thao tác này giúp làm mịn ảnh, giảm nhiễu hạt để cải thiện chất lượng khi phân ngưỡng.
3. **Phân ngưỡng nhị phân (Otsu's Thresholding)**: Hệ thống sử dụng phương pháp Otsu (`cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU`) để tự động tìm ra ngưỡng cắt tối ưu nhất. Kết quả trả về là một "mặt nạ" (mask) nhị phân nơi chiếc lá là vùng sáng và nền là vùng tối.
4. **Hình thái học (Morphology Ex - Close)**: Sử dụng phép đóng hình thái học (`MORPH_CLOSE`) với kernel hình elip `5x5` để lấp kín các lỗ hổng nhỏ, các chấm đen xuất hiện bên trong thân lá trên mask nhị phân.
5. **Lọc nhiễu (Giữ lại thành phần lớn nhất)**: Bằng cách tìm đường viền (`cv2.findContours`), hệ thống chỉ giữ lại vùng liên thông có diện tích lớn nhất (chính là chiếc lá) và xóa bỏ các đốm sáng nhỏ lẻ (nhiễu rác từ nền).
6. **Tách nền**: Áp dụng phép toán `cv2.bitwise_and` giữa ảnh gốc và mask nhị phân. Các pixel nằm ngoài vùng lá bị gán màu đen hoàn toàn (0, 0, 0), trong khi bản thân chiếc lá vẫn giữ được nguyên vẹn màu sắc.

## Bước 3: Chuẩn hóa kích thước (`resize_and_pad`)
Hệ thống cần một đầu vào có kích thước cố định (`256x256`) nhưng tuyệt đối không được làm biến dạng hình dáng thật của chiếc lá.
1. **Tính tỷ lệ thu phóng**: Xác định kích thước ảnh hiện tại. Tính tỷ lệ dựa vào cạnh lớn nhất để đưa độ dài cạnh này về chuẩn `256` pixel.
2. **Thu phóng nguyên tỷ lệ (Resize)**: Ảnh lá đã tách nền được thu nhỏ (hoặc phóng to) theo đúng tỷ lệ trên. Thuật toán `cv2.INTER_AREA` được dùng để có chất lượng hình ảnh tốt nhất khi thu nhỏ.
3. **Thêm viền đen (Padding)**: Vì ảnh sau khi thu phóng sẽ bị hụt một cạnh (do lá thường không phải hình vuông hoàn hảo), thuật toán tự động tính toán phần thiếu và đệm thêm viền đen (`cv2.copyMakeBorder`) đều ra xung quanh. Kết quả cuối cùng là một bức ảnh vuông `256x256` hoàn chỉnh với chiếc lá nằm ở chính giữa trung tâm.

## Kết luận
Sau 3 bước chính này, ảnh đầu ra là một ma trận vuông 256x256, chỉ chứa phần lá ở giữa (giữ nguyên tỷ lệ, màu sắc thực) với phông nền đen tuyệt đối. Dữ liệu này ở trạng thái tốt nhất, sẵn sàng phục vụ cho các bước trích xuất các vector đặc trưng tiếp theo.
