# Báo cáo Dự án: Nhận dạng Cảm xúc qua Giọng nói (Speech Emotion Recognition)

## 1. Giới thiệu

### 1.1. Mục tiêu

Dự án này nhằm xây dựng một hệ thống có khả năng nhận dạng cảm xúc của con người thông qua tín hiệu giọng nói. Hệ thống sẽ phân loại các đoạn âm thanh thành các lớp cảm xúc cơ bản như: Giận dữ (Anger), Ghê tởm (Disgust), Sợ hãi (Fear), Vui vẻ (Happy), Trung tính (Neutral), và Buồn (Sad).

### 1.2. Tập dữ liệu

Dự án sử dụng tập dữ liệu **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)**. Đây là một bộ dữ liệu đa phương thức chứa các bản ghi âm và video của các diễn viên thể hiện cảm xúc. Trong dự án này, chúng ta chỉ tập trung vào phần âm thanh (AudioWAV).

### 1.3. Hướng tiếp cận

Hướng tiếp cận chính của dự án là chuyển đổi bài toán từ phân loại chuỗi thời gian (tín hiệu âm thanh) sang bài toán phân loại hình ảnh. Quy trình tổng thể như sau:

1. **Tiền xử lý âm thanh**: Tín hiệu âm thanh thô được xử lý để loại bỏ nhiễu và các phần không chứa thông tin.
2. **Trích xuất đặc trưng**: Mỗi tệp âm thanh được chuyển đổi thành một ảnh **Mel Spectrogram**. Mel Spectrogram là một biểu đồ biểu diễn phổ năng lượng của tín hiệu âm thanh theo thang Mel, mô phỏng gần hơn với cách tai người cảm nhận âm thanh.
3. **Huấn luyện mô hình**: Sử dụng các kiến trúc mạng nơ-ron tích chập (Convolutional Neural Network - CNN) đã được huấn luyện trước (pre-trained) trên tập dữ liệu ImageNet để huấn luyện mô hình phân loại các ảnh Mel Spectrogram.
4. **Đánh giá**: Đánh giá hiệu suất của mô hình trên tập dữ liệu thử nghiệm.

Lý do chọn hướng tiếp cận này là vì các mô hình CNN pre-trained (như ResNet, DenseNet, VGG, EfficientNet) đã học được các đặc trưng hình ảnh rất mạnh mẽ từ tập ImageNet. Bằng cách chuyển âm thanh thành ảnh, chúng ta có thể tận dụng sức mạnh của học chuyển giao (Transfer Learning), giúp mô hình học nhanh hơn, hiệu quả hơn và yêu cầu ít dữ liệu hơn so với việc xây dựng một mô hình từ đầu.

## 2. Tiền xử lý dữ liệu

### 2.1. Tải và Khám phá Dữ liệu

- **Tải dữ liệu**: Các tệp âm thanh từ thư mục `dataset/CREMA-D/AudioWAV/` được quét và thông tin về `speaker` (người nói), `path` (đường dẫn tệp), và `emotion` (cảm xúc) được trích xuất từ tên tệp.
- **Phân tích**: Dữ liệu được lưu vào một `DataFrame` của Pandas. Biểu đồ phân phối cảm xúc cho thấy dữ liệu tương đối cân bằng giữa các lớp, đây là một điều kiện thuận lợi cho việc huấn luyện mô hình.

### 2.2. Loại bỏ Khoảng lặng (Silence Removal)

- **Vấn đề**: Các tệp âm thanh thường chứa các khoảng lặng ở đầu và cuối. Những khoảng lặng này không mang thông tin về cảm xúc và có thể được xem là nhiễu, làm giảm hiệu suất của mô hình.
- **Giải pháp**: Sử dụng thư viện `pydub` để phát hiện và cắt bỏ các khoảng lặng này. Một ngưỡng âm lượng (`-50.0 dBFS`) được sử dụng để xác định đâu là khoảng lặng.
- **Kết quả**: Các tệp âm thanh đã được xử lý được lưu vào thư mục `dataset_silenced/`, và một tệp CSV mới (`CSVs/dataset_silenced.csv`) được tạo ra để theo dõi.

### 2.3. Phân chia Tập dữ liệu (Train/Validation/Test Split)

- **Sự cần thiết**: Để huấn luyện và đánh giá mô hình một cách khách quan, dữ liệu cần được chia thành 3 tập riêng biệt:
  - **Tập huấn luyện (Train set)**: Dùng để huấn luyện mô hình.
  - **Tập kiểm định (Validation set)**: Dùng để tinh chỉnh các siêu tham số (hyperparameters) và theo dõi quá trình huấn luyện, tránh tình trạng học vẹt (overfitting).
  - **Tập thử nghiệm (Test set)**: Dùng để đánh giá hiệu suất cuối cùng của mô hình trên dữ liệu mà nó chưa từng thấy.
- **Phương pháp**: `GroupShuffleSplit` từ `scikit-learn` được sử dụng.
- **Lý do chọn `GroupShuffleSplit`**: Trong tập dữ liệu này, mỗi người nói có nhiều bản ghi âm. Nếu chỉ chia ngẫu nhiên, các bản ghi của cùng một người nói có thể xuất hiện trong cả ba tập dữ liệu. Điều này có thể khiến mô hình "nhớ" giọng của người nói thay vì học các đặc trưng cảm xúc thực sự. `GroupShuffleSplit` đảm bảo rằng tất cả các bản ghi của một người nói chỉ thuộc về một tập duy nhất (train, val, hoặc test). Điều này giúp mô hình có khả năng tổng quát hóa tốt hơn với giọng nói của những người mới.
- **Tỷ lệ phân chia**: 80% cho tập huấn luyện, 10% cho tập kiểm định, và 10% cho tập thử nghiệm.

### 2.4. Tăng cường Dữ liệu (Data Augmentation)

- **Vấn đề**: Để mô hình có khả năng chống nhiễu và tổng quát hóa tốt hơn, chúng ta cần làm cho dữ liệu huấn luyện đa dạng hơn.
- **Giải pháp**: Áp dụng các kỹ thuật tăng cường dữ liệu âm thanh một cách ngẫu nhiên cho tập huấn luyện:
  - **Thêm nhiễu (Noise)**: Thêm nhiễu ngẫu nhiên vào tín hiệu.
  - **Kéo dài/Co giãn thời gian (Time Stretch)**: Thay đổi tốc độ của âm thanh mà không làm thay đổi cao độ.
  - **Thay đổi cao độ (Pitch Shift)**: Thay đổi cao độ của âm thanh mà không làm thay đổi tốc độ.
- **Thực hiện**: Mỗi tệp âm thanh trong tập huấn luyện được giữ lại bản gốc và tạo thêm một phiên bản tăng cường. Dữ liệu tăng cường được lưu vào `dataset_augmented/`.

## 3. Chuyển đổi sang Mel Spectrogram

Đây là bước cốt lõi của phương pháp tiếp cận.

- **Mel Spectrogram là gì?**: Nó là một biểu đồ 2D biểu diễn sự thay đổi của phổ năng lượng tín hiệu âm thanh theo thời gian. Trục hoành là thời gian, trục tung là tần số (theo thang Mel), và màu sắc biểu thị cường độ (biên độ) tại mỗi điểm thời gian-tần số.
- **Tại sao lại dùng Mel Spectrogram?**:
  - Nó mô phỏng cách tai người cảm nhận tần số, tập trung nhiều hơn vào các tần số thấp, nơi chứa nhiều thông tin quan trọng của giọng nói.
  - Nó biến đổi tín hiệu 1D (âm thanh) thành biểu diễn 2D (ảnh), cho phép chúng ta áp dụng các mô hình CNN mạnh mẽ.
- **Quá trình thực hiện**:
  1. Sử dụng thư viện `librosa` để tải tệp âm thanh.
  2. Tính toán Mel Spectrogram từ tín hiệu âm thanh. Các tham số quan trọng (`N_MELS`, `N_FFT`, `HOP_LENGTH`) quyết định độ phân giải về tần số và thời gian của ảnh.
  3. Chuyển đổi biên độ sang thang decibel (dB) để nén dải động và làm nổi bật các đặc trưng.
  4. Vẽ spectrogram và lưu dưới dạng tệp ảnh PNG (`.png`) không có trục và viền.
- **Lưu trữ**: Các ảnh được tạo ra được lưu vào `features/images/`. Dữ liệu ảnh (dưới dạng mảng NumPy) và nhãn tương ứng được lưu vào các tệp `.npy` (`features/*.npy`) để tăng tốc độ tải dữ liệu trong các lần chạy sau.

## 4. Tải dữ liệu và Tạo Data Loader

- **Tải dữ liệu ảnh**: Các tệp `.npy` chứa mảng ảnh và nhãn được tải vào bộ nhớ.
- **One-Hot Encoding**: Nhãn cảm xúc (dạng chuỗi) được chuyển đổi thành vector one-hot. Ví dụ: `Happy` -> `[0, 1, 0, 0, 0, 0]`. Đây là định dạng đầu ra mà mô hình cần để tính toán hàm mất mát `CrossEntropyLoss`.
- **Tạo `Dataset` và `DataLoader`**:
  - Một lớp `EmotionDataset` tùy chỉnh được tạo ra để quản lý việc truy xuất ảnh và nhãn.
  - `DataLoader` được sử dụng để tạo các lô (batch) dữ liệu từ `Dataset`. Nó giúp quản lý việc xáo trộn dữ liệu (shuffle), tải dữ liệu song song, và tự động hóa quá trình đưa dữ liệu vào mô hình.
- **Biến đổi ảnh (Transforms)**:
  - **Resize**: Đồng bộ hóa kích thước tất cả các ảnh về `224x224`, kích thước đầu vào tiêu chuẩn của nhiều mô hình pre-trained.
  - **ToTensor**: Chuyển đổi ảnh từ định dạng `PIL Image` (hoặc mảng NumPy) sang `Tensor` của PyTorch.
  - **Normalize**: Chuẩn hóa các giá trị pixel của ảnh bằng cách sử dụng trung bình (mean) và độ lệch chuẩn (std) của tập dữ liệu ImageNet. Đây là một bước bắt buộc khi sử dụng các mô hình pre-trained trên ImageNet, vì nó đảm bảo rằng dữ liệu đầu vào của chúng ta có cùng phân phối với dữ liệu mà mô hình đã được huấn luyện.

## 5. Xây dựng Mô hình (Model Architecture)

Dự án thử nghiệm với nhiều kiến trúc CNN pre-trained khác nhau để tìm ra mô hình tốt nhất. Nguyên tắc chung là **Học chuyển giao (Transfer Learning)**.

- **Nguyên tắc**:
  1. Tải một mô hình đã được huấn luyện trên ImageNet (ví dụ: `ResNet18`).
  2. **Đóng băng (Freeze)** hầu hết các lớp của mô hình. Các lớp này đã học được các đặc trưng hình ảnh tổng quát (cạnh, góc, kết cấu...). Chúng ta giữ lại các trọng số này.
  3. **Mở băng (Unfreeze)** một vài lớp cuối. Các lớp này học các đặc trưng phức tạp và chuyên biệt hơn. Chúng ta cho phép chúng được cập nhật trong quá trình huấn luyện để thích nghi với dữ liệu Mel Spectrogram.
  4. **Thay thế lớp phân loại (Classifier)**: Lớp cuối cùng của mô hình gốc (thường có 1000 đầu ra cho ImageNet) được thay thế bằng một lớp phân loại mới, tùy chỉnh cho bài toán của chúng ta (6 đầu ra cho 6 cảm xúc). Lớp này sẽ được huấn luyện từ đầu.

- **Các mô hình được sử dụng**:
  - `ResNet18`
  - `DenseNet121`
  - `VGG16`
  - `EfficientNet-B0`

- **Lý do lựa chọn**: Các mô hình này có kiến trúc đa dạng, đại diện cho các trường phái thiết kế CNN khác nhau và đã chứng tỏ hiệu quả cao trên nhiều bài toán thị giác máy tính.

## 6. Huấn luyện và Đánh giá

### 6.1. Thiết lập Huấn luyện

- **Hàm mất mát (Loss Function)**: `CrossEntropyLoss` với `label_smoothing=0.1`.
  - `CrossEntropyLoss` là lựa chọn tiêu chuẩn cho bài toán phân loại đa lớp.
  - `Label Smoothing` là một kỹ thuật điều chuẩn (regularization) giúp mô hình bớt "tự tin" một cách thái quá vào dự đoán của mình, từ đó giảm overfitting và tăng khả năng tổng quát hóa.
- **Trình tối ưu hóa (Optimizer)**: `AdamW`.
  - `AdamW` là một biến thể của Adam optimizer, cải thiện cách xử lý suy giảm trọng số (weight decay), thường cho kết quả tốt hơn.
  - Một **tỷ lệ học khác biệt (differential learning rate)** được áp dụng: các lớp được mở băng (gần đầu ra hơn) có tỷ lệ học cao hơn (`5e-5`), trong khi các lớp sâu hơn có tỷ lệ học thấp hơn (`1e-5`). Lý do là các lớp sâu hơn chỉ cần tinh chỉnh nhẹ, trong khi các lớp mới cần học nhanh hơn.
- **Bộ lập lịch Tỷ lệ học (Learning Rate Scheduler)**: `ReduceLROnPlateau`.
  - Cơ chế này sẽ tự động giảm tỷ lệ học khi hàm mất mát trên tập kiểm định (`val_loss`) không cải thiện sau một số `patience` epoch nhất định. Điều này giúp mô hình hội tụ tốt hơn ở giai đoạn cuối của quá trình huấn luyện.
- **Dừng sớm (Early Stopping)**:
  - Quá trình huấn luyện sẽ dừng lại nếu `val_loss` không cải thiện trong một số `PATIENCE` epoch liên tiếp (ở đây là 15).
  - Điều này giúp tiết kiệm thời gian tính toán và ngăn mô hình bắt đầu học vẹt (overfitting) khi nó không còn học được điều gì hữu ích nữa.
  - Mô hình có `val_loss` tốt nhất sẽ được lưu lại.

### 6.2. Quá trình Huấn luyện và Kiểm định

- Trong mỗi epoch, mô hình thực hiện hai pha:
  1. **Pha Huấn luyện (Train Phase)**: Mô hình học từ tập `train_loader`. Trọng số được cập nhật.
  2. **Pha Kiểm định (Validation Phase)**: Mô hình được đánh giá trên tập `val_loader`. Trọng số không được cập nhật. Kết quả ở pha này được dùng để theo dõi hiệu suất và ra quyết định (giảm learning rate, dừng sớm, lưu mô hình).
- Các chỉ số `Loss`, `Accuracy`, `F1-score`, `Recall`, `Precision` được ghi lại cho cả hai tập để theo dõi.

### 6.3. Đánh giá trên Tập Thử nghiệm (Test)

- Sau khi quá trình huấn luyện kết thúc, mô hình có hiệu suất tốt nhất trên tập kiểm định được tải lại.
- Mô hình được đánh giá lần cuối trên tập `test_loader`. Đây là kết quả cuối cùng, phản ánh hiệu suất của mô hình trên dữ liệu hoàn toàn mới.
- **Classification Report**: Cung cấp một báo cáo chi tiết về `precision`, `recall`, `f1-score` cho từng lớp cảm xúc.
- **Confusion Matrix (Ma trận nhầm lẫn)**: Trực quan hóa hiệu suất của mô hình. Nó cho thấy mô hình dự đoán đúng bao nhiêu mẫu cho mỗi lớp và thường nhầm lẫn giữa các lớp nào.

## 7. Kết luận

Dự án đã xây dựng thành công một quy trình hoàn chỉnh để nhận dạng cảm xúc qua giọng nói bằng cách sử dụng phương pháp chuyển đổi sang Mel Spectrogram và học chuyển giao. Việc thử nghiệm với nhiều kiến trúc CNN khác nhau cho phép so sánh và lựa chọn mô hình phù hợp nhất cho bài toán. Các kỹ thuật như phân chia dữ liệu theo nhóm, tăng cường dữ liệu, và các chiến lược huấn luyện nâng cao (learning rate scheduler, early stopping) đều đóng vai trò quan trọng trong việc xây dựng một mô hình mạnh mẽ và có khả năng tổng quát hóa tốt.
