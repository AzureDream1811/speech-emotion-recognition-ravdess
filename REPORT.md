
# Báo cáo: Nhận dạng cảm xúc trong giọng nói sử dụng tập dữ liệu CREMA-D

## Giới thiệu

Tài liệu này trình bày chi tiết quy trình xây dựng một mô hình học sâu để nhận dạng cảm xúc trong giọng nói. Quy trình bao gồm các bước từ tiền xử lý dữ liệu âm thanh, trích xuất đặc trưng, đến huấn luyện và đánh giá mô hình. Tập dữ liệu được sử dụng là CREMA-D, một bộ dữ liệu phổ biến cho các tác vụ nhận dạng cảm xúc. Mô hình được xây dựng dựa trên kiến trúc ResNet-50 thông qua kỹ thuật học chuyển giao (transfer learning).

## 1. Chuẩn bị môi trường và Nhập thư viện

Bước đầu tiên là thiết lập môi trường làm việc và nhập các thư viện Python cần thiết.

- **`os`**: Tương tác với hệ điều hành, dùng để quản lý đường dẫn và tệp tin.
- **`pandas`**: Đọc, ghi và xử lý dữ liệu dạng bảng (DataFrame).
- **`numpy`**: Thực hiện các phép toán số học hiệu suất cao, đặc biệt là với mảng đa chiều.
- **`seaborn` và `matplotlib.pyplot`**: Trực quan hóa dữ liệu, vẽ biểu đồ và đồ thị.
- **`librosa`**: Một thư viện mạnh mẽ chuyên dụng cho phân tích và xử lý âm thanh.
- **`soundfile` (`sf`)**: Đọc và ghi các tệp âm thanh.
- **`pydub`**: Thư viện cấp cao để xử lý âm thanh, dùng để loại bỏ khoảng lặng.
- **`torch` và `torchvision`**: Nền tảng học sâu của PyTorch, cung cấp các công cụ để xây dựng, huấn luyện và triển khai mô hình mạng nơ-ron.
- **`sklearn.model_selection`**: Cung cấp công cụ để chia tập dữ liệu.
- **`tqdm`**: Tạo thanh tiến trình (progress bar) để theo dõi các vòng lặp.
- **Các thư viện khác**: `PIL` (xử lý ảnh), `cv2` (OpenCV), `warnings`, `copy` cho các tác vụ phụ trợ.

## 2. Tiền xử lý dữ liệu

Đây là giai đoạn quan trọng nhất, quyết định phần lớn đến hiệu suất của mô hình. Dữ liệu âm thanh thô được xử lý qua nhiều bước để trở nên phù hợp cho việc huấn luyện.

### 2.1. Tải và khám phá tập dữ liệu CREMA-D

- **Tải dữ liệu**: Quét qua thư mục chứa tập dữ liệu CREMA-D (`dataset/CREMA-D/AudioWAV/`). Tên mỗi tệp âm thanh chứa thông tin về người nói, câu nói và cảm xúc.
- **Trích xuất thông tin**: Từ tên tệp, các thông tin như `speaker` (người nói) và `emotion` (cảm xúc) được trích xuất. Các mã cảm xúc (ví dụ: "ANG" cho "Anger") được ánh xạ sang tên cảm xúc đầy đủ.
- **Tạo DataFrame**: Toàn bộ thông tin được lưu vào một DataFrame của `pandas` để dễ dàng quản lý và truy xuất. DataFrame này bao gồm các cột: `speaker`, `path` (đường dẫn đến tệp âm thanh), và `emotion`.
- **Phân tích phân phối**: Sử dụng `seaborn.countplot` để vẽ biểu đồ phân phối số lượng mẫu cho mỗi loại cảm xúc. Điều này giúp kiểm tra xem tập dữ liệu có bị mất cân bằng hay không.

### 2.2. Loại bỏ khoảng lặng

Các khoảng lặng ở đầu và cuối mỗi đoạn ghi âm không chứa thông tin hữu ích về cảm xúc và có thể gây nhiễu cho mô hình.

- **Sử dụng `pydub`**: Thư viện `pydub` được dùng để phát hiện và cắt bỏ các khoảng lặng.
- **Cơ chế hoạt động**: Một hàm `detect_leading_silence` được định nghĩa để quét qua đoạn âm thanh theo từng đoạn nhỏ (chunk). Nếu mức âm lượng (dBFS) của một chunk thấp hơn một ngưỡng (`-50.0 dBFS`), nó được coi là khoảng lặng. Quá trình này được thực hiện từ cả đầu và cuối của file âm thanh.
- **Lưu kết quả**: Các tệp âm thanh sau khi đã được cắt bỏ khoảng lặng được lưu vào một thư mục mới (`dataset_silenced/`). Một DataFrame mới (`dataset_silenced`) được tạo để theo dõi các tệp đã xử lý này.

### 2.3. Phân chia tập dữ liệu

Để đánh giá mô hình một cách khách quan, tập dữ liệu được chia thành ba phần: huấn luyện (train), kiểm định (validation), và kiểm thử (test).

- **`GroupShuffleSplit`**: Công cụ này từ `scikit-learn` được sử dụng để đảm bảo rằng tất cả các mẫu âm thanh từ cùng một người nói (`speaker`) chỉ thuộc về một trong ba tập (train, val, hoặc test). Điều này ngăn chặn "rò rỉ dữ liệu", giúp mô hình tổng quát hóa tốt hơn trên những người nói mà nó chưa từng gặp.
- **Tỷ lệ phân chia**:
    1.  80% dữ liệu được dùng cho tập huấn luyện (`train_df`).
    2.  20% còn lại được chia đều thành 10% cho tập kiểm định (`val_df`) và 10% cho tập kiểm thử (`test_df`).
- **Lưu trữ**: Các DataFrame tương ứng với mỗi tập được lưu thành các tệp CSV riêng biệt (`train.csv`, `val.csv`, `test.csv`).

### 2.4. Tăng cường dữ liệu (Data Augmentation)

Tăng cường dữ liệu là một kỹ thuật quan trọng để tăng kích thước và sự đa dạng của tập huấn luyện, giúp mô hình chống lại hiện tượng học vẹt (overfitting). Chỉ tập huấn luyện được tăng cường.

- **Các kỹ thuật được áp dụng**:
    - **Thêm nhiễu (Noise)**: Thêm nhiễu ngẫu nhiên vào tín hiệu âm thanh.
    - **Giãn/Nén thời gian (Stretch)**: Thay đổi tốc độ của âm thanh một cách ngẫu nhiên.
    - **Thay đổi cao độ (Pitch)**: Dịch chuyển cao độ của âm thanh lên hoặc xuống.
- **Quy trình**: Với mỗi mẫu trong tập huấn luyện, một bản sao gốc được giữ lại. Sau đó, một trong ba kỹ thuật tăng cường trên được chọn ngẫu nhiên và áp dụng để tạo ra một mẫu mới.
- **Kết quả**: Tập huấn luyện được mở rộng, với mỗi mẫu gốc có thêm một phiên bản tăng cường. Dữ liệu mới được lưu vào thư mục `dataset_augmented/` và được quản lý bởi `dataset_augmented.csv`.

## 3. Trích xuất đặc trưng: Mel Spectrogram

Mạng nơ-ron tích chập (CNN), kiến trúc được sử dụng trong dự án này, được thiết kế để làm việc với dữ liệu dạng hình ảnh. Do đó, các tín hiệu âm thanh một chiều cần được chuyển đổi thành một biểu diễn hai chiều giống như hình ảnh. Mel Spectrogram là một lựa chọn phổ biến và hiệu quả cho việc này.

### 3.1. Chuyển đổi âm thanh thành Mel Spectrogram

- **Cố định độ dài**: Các đoạn âm thanh có độ dài khác nhau. Để đảm bảo đầu vào cho mô hình có kích thước đồng nhất, tất cả các đoạn âm thanh được cắt hoặc đệm (pad) để có cùng độ dài (3 giây).
- **Tạo Mel Spectrogram**:
    1.  **Short-Time Fourier Transform (STFT)**: Tín hiệu âm thanh được chia thành các khung (frame) ngắn chồng chéo nhau, và biến đổi Fourier được áp dụng trên từng khung để phân tích phổ tần số.
    2.  **Mel Scale**: Phổ tần số sau đó được chuyển đổi sang thang đo Mel, một thang đo mô phỏng cách tai người cảm nhận tần số.
    3.  **Logarithmic Scale**: Cuối cùng, biên độ được chuyển đổi sang thang đo decibel (dB), gần với cách con người cảm nhận âm lượng.
- **Lưu đặc trưng**: Các Mel Spectrogram (dưới dạng mảng NumPy) được lưu vào thư mục `features/mels/` trong các thư mục con `train`, `val`, `test`.

### 3.2. Chuyển đổi `.npy` thành `.png`

Để mô hình ResNet-50 có thể sử dụng, các mảng Mel Spectrogram được chuyển đổi thành tệp hình ảnh.

- **Quy trình**:
    1.  Tải tệp `.npy` chứa Mel Spectrogram.
    2.  Sử dụng `librosa.display.specshow` để vẽ Mel Spectrogram lên một biểu đồ `matplotlib`. Các tham số như `cmap='inferno'` được sử dụng để tạo ra hình ảnh có màu sắc.
    3.  Hình ảnh được lưu vào bộ nhớ đệm (buffer) dưới định dạng PNG.
    4.  Sử dụng `PIL` và `OpenCV` để đọc lại hình ảnh từ bộ nhớ đệm và lưu nó dưới dạng tệp `.png` vào thư mục `features/images/`.

## 4. Xây dựng và Huấn luyện mô hình

### 4.1. Chuẩn bị cho PyTorch

- **`EmotionDataset`**: Một lớp Dataset tùy chỉnh được tạo để tải các hình ảnh Mel Spectrogram và nhãn tương ứng. Lớp này kế thừa từ `torch.utils.data.Dataset` và triển khai các phương thức `__len__` và `__getitem__`.
- **`transforms`**: Các phép biến đổi hình ảnh được áp dụng trong quá trình tải dữ liệu.
    - **Tập huấn luyện**: Bao gồm thay đổi kích thước, lật ngang ngẫu nhiên, xoay ngẫu nhiên và chuẩn hóa. Các phép biến đổi này cũng là một dạng tăng cường dữ liệu ở cấp độ hình ảnh.
    - **Tập kiểm định**: Chỉ bao gồm thay đổi kích thước và chuẩn hóa để đảm bảo dữ liệu nhất quán.
- **`DataLoader`**: Tạo các trình tải dữ liệu cho tập huấn luyện và kiểm định. `DataLoader` quản lý việc tạo các lô (batch) dữ liệu, xáo trộn dữ liệu (shuffle) cho tập huấn luyện, và sử dụng đa luồng để tăng tốc độ tải dữ liệu.

### 4.2. Thiết lập mô hình (ResNet-50)

- **Học chuyển giao (Transfer Learning)**: Thay vì xây dựng một mô hình từ đầu, chúng ta sử dụng mô hình **ResNet-50** đã được huấn luyện trước trên tập dữ liệu ImageNet. Kiến trúc này đã học được các đặc trưng hình ảnh cấp thấp và trung bình rất tốt.
- **Đóng băng các lớp**: Tất cả các trọng số của các lớp tích chập trong ResNet-50 được "đóng băng" (`param.requires_grad = False`). Điều này có nghĩa là chúng sẽ không được cập nhật trong quá trình huấn luyện.
- **Thay thế lớp cuối cùng**: Lớp phân loại cuối cùng (fully connected layer) của ResNet-50, vốn được thiết kế cho 1000 lớp của ImageNet, được thay thế bằng một lớp `nn.Linear` mới. Lớp này có đầu ra bằng với số lượng cảm xúc cần nhận dạng trong bài toán của chúng ta.
- **Chỉ huấn luyện lớp mới**: Chỉ có các tham số của lớp phân loại mới này được huấn luyện. Điều này giúp mô hình thích nghi nhanh chóng với tác vụ mới mà không làm mất đi các kiến thức đã học.

### 4.3. Vòng lặp huấn luyện

- **Hàm `train_model`**: Hàm này chứa logic chính cho việc huấn luyện và kiểm định mô hình qua nhiều kỷ nguyên (epoch).
- **Các giai đoạn (Phase)**: Trong mỗi epoch, mô hình trải qua hai giai đoạn: `train` và `val`.
    - **Giai đoạn `train`**: Mô hình được đặt ở chế độ `model.train()`. Quá trình lan truyền ngược (backpropagation) và cập nhật trọng số được kích hoạt.
    - **Giai đoạn `val`**: Mô hình được đặt ở chế độ `model.eval()`. Việc tính toán gradient bị tắt để tiết kiệm bộ nhớ và tăng tốc độ.
- **Quy trình trong mỗi giai đoạn**:
    1.  Lặp qua các batch dữ liệu từ `DataLoader`.
    2.  Đưa dữ liệu và nhãn lên thiết bị tính toán (CPU hoặc GPU).
    3.  Dự đoán đầu ra với mô hình.
    4.  Tính toán hàm mất mát (loss) bằng `nn.CrossEntropyLoss`.
    5.  (Chỉ trong giai đoạn train) Thực hiện lan truyền ngược và cập nhật trọng số bằng `optimizer.step()`.
    6.  Tính toán và tích lũy loss và độ chính xác (accuracy).
- **Lưu mô hình tốt nhất**: Sau mỗi epoch, độ chính xác trên tập kiểm định được so sánh với độ chính xác tốt nhất đã đạt được. Nếu mô hình hiện tại tốt hơn, trọng số của nó sẽ được lưu lại vào tệp `best_model.pth`.

## 5. Đánh giá mô hình

Sau khi quá trình huấn luyện hoàn tất, mô hình với hiệu suất tốt nhất trên tập kiểm định được tải lại và đánh giá trên cả tập kiểm định và tập kiểm thử.

- **Hàm `evaluate_model`**: Hàm này lấy đầu vào là mô hình và một `DataLoader`, sau đó trả về tất cả các nhãn thực tế và nhãn dự đoán.
- **Các chỉ số đánh giá**:
    - **`classification_report`**: Cung cấp các chỉ số chi tiết cho từng lớp cảm xúc, bao gồm Precision, Recall, và F1-score.
    - **`confusion_matrix` (Ma trận nhầm lẫn)**: Một bảng trực quan hóa hiệu suất của mô hình. Các hàng đại diện cho các lớp thực tế, các cột đại diện cho các lớp dự đoán. Nó cho thấy mô hình thường nhầm lẫn giữa các cặp cảm xúc nào.
- **Đánh giá trên tập Test**: Đây là bước cuối cùng và quan trọng nhất, cho thấy mô hình hoạt động tốt như thế nào trên dữ liệu hoàn toàn mới mà nó chưa từng thấy. Kết quả trên tập này là thước đo cuối cùng về hiệu suất của mô hình.

## Kết luận

Notebook này đã trình bày một quy trình hoàn chỉnh và chi tiết để giải quyết bài toán nhận dạng cảm xúc trong giọng nói. Bằng cách kết hợp các kỹ thuật tiền xử lý âm thanh tiên tiến, trích xuất đặc trưng bằng Mel Spectrogram, và sức mạnh của học chuyển giao với kiến trúc ResNet-50, mô hình đã được xây dựng và đánh giá một cách có hệ thống. Các kết quả đánh giá trên tập kiểm thử cung cấp một cái nhìn khách quan về khả năng của mô hình trong việc nhận dạng cảm xúc từ các mẫu giọng nói thực tế.
