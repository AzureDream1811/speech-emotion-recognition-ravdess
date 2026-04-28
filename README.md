# Nhận Dạng Cảm Xúc Qua Giọng Nói

Dự án này xây dựng một quy trình học sâu (deep learning) để nhận dạng cảm xúc của con người từ giọng nói. Hệ thống xử lý các file âm thanh thô, chuyển đổi chúng thành dạng biểu diễn hình ảnh (Mel Spectrogram), và sau đó sử dụng một Mạng Nơ-ron Tích chập (CNN) để phân loại cảm xúc.

## 🚀 Tính Năng

- **Quy Trình Toàn Diện**: Từ âm thanh thô đến phân loại cảm xúc.
- **Tiền Xử Lý Dữ Liệu**: Bao gồm loại bỏ khoảng lặng và tăng cường dữ liệu (thêm nhiễu, thay đổi cao độ, co giãn thời gian) để tạo ra một bộ dữ liệu mạnh mẽ.
- **Trích Xuất Đặc Trưng**: Chuyển đổi tín hiệu âm thanh thành hình ảnh Mel Spectrogram, phù hợp cho các mô hình CNN.
- **Mô Hình Học Sâu**: Sử dụng mô hình ResNet đã được huấn luyện trước (pre-trained) và tinh chỉnh (fine-tuning) để đạt độ chính xác cao.
- **Kỹ Thuật Huấn Luyện Nâng Cao**: Áp dụng scheduler `OneCycleLR`, trình tối ưu hóa `AdamW`, và Test Time Augmentation (TTA) để cải thiện hiệu suất.
- **Quy Trình Có Cấu Trúc**: Toàn bộ quy trình được tổ chức thành một chuỗi các file Jupyter Notebook.

## 📂 Cấu Trúc Dự Án

```
.
├── 1_Preparedataset.ipynb      # Notebook để làm sạch, tăng cường và chia dữ liệu.
├── 2_FeatureExtraction.ipynb   # Notebook để chuyển đổi âm thanh thành Mel Spectrogram.
├── 3_CNN-classification.ipynb  # Notebook để huấn luyện và đánh giá mô hình CNN.
├── requirements.txt            # Các gói Python cần thiết.
├── dataset/                    # Thư mục chứa các bộ dữ liệu âm thanh thô (RAVDESS, CREMA-D).
├── CSVs/                       # Lưu các file CSV chứa đường dẫn file và nhãn.
├── features/                   # Lưu các đặc trưng đã trích xuất (ảnh Mel Spectrogram).
└── ...
```

## ⚙️ Cài Đặt

Thực hiện theo các bước sau để thiết lập môi trường cho dự án.

### 1. Clone Repository

```bash
git clone https://github.com/AzureDream1811/speech-emotion-recognition-ravdess.git
cd speech-emotion-recognition-ravdess
```

### 2. Tạo Môi Trường Ảo

Rất khuyến khích sử dụng môi trường ảo để quản lý các gói phụ thuộc.

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường
# Trên Windows
.venv\Scripts\activate
# Trên macOS/Linux
source .venv/bin/activate
```

### 3. Cài Đặt Các Gói Phụ Thuộc

Cài đặt tất cả các gói cần thiết từ file `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Cài Đặt FFmpeg (Để Loại Bỏ Khoảng Lặng)

Bước loại bỏ khoảng lặng trong `1_Preparedataset.ipynb` yêu cầu FFmpeg.

- **Windows**: Tải file thực thi của FFmpeg, sau đó thêm đường dẫn đến thư mục `bin` vào biến môi trường PATH của hệ thống.
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg`

### 5. Tải Dữ Liệu

Dự án này sử dụng bộ dữ liệu **CREMA-D** và **RAVDESS**.

1.  Tải các bộ dữ liệu từ nguồn chính thức của chúng.
2.  Tạo một thư mục có tên `dataset` trong thư mục gốc của dự án.
3.  Đặt các file âm thanh theo cấu trúc sau:
    ```
    dataset/
    ├── CREMA-D/
    │   └── AudioWAV/
    │       ├── 1001_DFA_ANG_XX.wav
    │       └── ...
    └── ravdess/
        ├── Actor_01/
        │   ├── 03-01-01-01-01-01-01.wav
        │   └── ...
        └── Actor_02/
            └── ...
    ```

## 📈 Quy Trình - Hướng Dẫn Chạy

Dự án được chia thành ba notebook chính. Hãy chạy chúng theo thứ tự sau.

### Bước 1: Chuẩn Bị Dữ Liệu

**Notebook:** [1_Preparedataset.ipynb](1_Preparedataset.ipynb)

Notebook này xử lý tất cả các bước chuẩn bị dữ liệu ban đầu cho bộ dữ liệu CREMA-D.

- **Tải đường dẫn âm thanh** và trích xuất nhãn (cảm xúc, người nói).
- **Chia dữ liệu** thành các tập huấn luyện và kiểm thử, đảm bảo rằng người nói trong tập huấn luyện không xuất hiện trong tập kiểm thử (sử dụng `GroupShuffleSplit`).
- **Loại bỏ khoảng lặng** ở đầu và cuối của các file âm thanh.
- **Thực hiện tăng cường dữ liệu** trên tập huấn luyện bằng cách thêm nhiễu, co giãn thời gian, hoặc thay đổi cao độ.
- **Lưu danh sách file cuối cùng** vào các file CSV trong thư mục `CSVs/`.

> **Chạy tất cả các ô (cell) trong notebook này từ trên xuống dưới.**

### Bước 2: Trích Xuất Đặc Trưng (Mel Spectrograms)

**Notebook:** [2_FeatureExtraction.ipynb](2_FeatureExtraction.ipynb)

Notebook này chuyển đổi các file âm thanh đã được tiền xử lý thành hình ảnh Mel Spectrogram, đây sẽ là đầu vào cho mô hình CNN của chúng ta.

- **Tải các file CSV** được tạo ở bước trước.
- **Lặp qua từng file âm thanh**, chuẩn hóa độ dài của nó về một khoảng thời gian tiêu chuẩn (3 giây).
- **Tạo một Mel Spectrogram** cho mỗi file âm thanh.
- **Lưu các spectrogram dưới dạng ảnh PNG** vào các thư mục `features/images/train` và `features/images/test`.
- **Tạo các file CSV mới** (`train_images.csv`, `test_images.csv`) để ánh xạ các ảnh này với cảm xúc và người nói tương ứng.

> **Chạy tất cả các ô trong notebook này từ trên xuống dưới.**

### Bước 3: Huấn Luyện Mô Hình CNN

**Notebook:** [3_CNN-classification.ipynb](3_CNN-classification.ipynb)

Đây là bước cuối cùng, nơi chúng ta huấn luyện và đánh giá mô hình nhận dạng cảm xúc.

- **Định nghĩa các phép biến đổi** và tăng cường dữ liệu cho ảnh spectrogram.
- **Tạo các đối tượng `Dataset` và `DataLoader`** cho việc huấn luyện, xác thực và kiểm thử.
- **Xây dựng mô hình ResNet** sử dụng học chuyển giao (transfer learning), tinh chỉnh nó cho tác vụ cụ thể của chúng ta.
- **Huấn luyện mô hình** bằng tập huấn luyện và đánh giá nó trên tập xác thực sau mỗi epoch.
- **Áp dụng Early Stopping** để ngăn chặn overfitting và lưu lại mô hình tốt nhất dựa trên điểm F1-score.
- **Đánh giá mô hình cuối cùng** trên tập kiểm thử chưa từng thấy bằng cách sử dụng **Test Time Augmentation (TTA)** để tăng cường độ chính xác.

> **Chạy tất cả các ô trong notebook này để huấn luyện mô hình và xem các chỉ số hiệu suất cuối cùng.**

## 📊 Kết Quả

Hiệu suất của mô hình được đánh giá bằng các chỉ số Accuracy, F1-Score, Precision và Recall. Kết quả cuối cùng trên tập kiểm thử được in ra ở cuối notebook `3_CNN-classification.ipynb`.

*(Bạn có thể thêm bảng kết quả cuối cùng của mình vào đây sau khi chạy toàn bộ quy trình)*

| Model    | Accuracy | F1-Score |
| -------- | -------- | -------- |
| ResNet34 | ...      | ...      |
