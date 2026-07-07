# Speech Emotion Recognition trên RAVDESS + SAVEE: So sánh MFCC+SVM, ResNet34 và DenseNet121

---

## Tóm tắt

Project thực nghiệm ba phương pháp cho bài toán Speech Emotion Recognition (SER) trên tập dữ liệu kết hợp RAVDESS +
SAVEE (1920 mẫu, 8 lớp cảm xúc): **(1)** SVM với đặc trưng MFCC (mean+std, 40 chiều), **(2)** ResNet34 transfer learning
trên log-mel spectrogram, **(3)** DenseNet121 transfer learning trên log-mel spectrogram. Cả ba dùng chung split
80/10/10 (stratified, seed=42). Kết quả trên test set: MFCC+SVM đạt 66.67% accuracy (F1-weighted 0.6669), ResNet34 đạt
76.04% (F1-macro 0.7588) — tốt nhất, DenseNet121 đạt 73.96% (F1-macro 0.7386). ResNet34 vượt trội các lớp *angry*,
*surprise*; SVM overfit nặng (train 99.48% vs test 66.67%); cả hai CNN đều overfit gần như tuyệt đối (train ~100%). Hạn
chế lớn nhất là speaker leakage do split ngẫu nhiên theo file, không theo diễn viên.

---

## 1. Giới thiệu

### 1.1 Bối cảnh

Nhận dạng cảm xúc từ giọng nói (SER) là bài toán phân loại một đoạn âm thanh chứa lời nói thành một trong các nhãn cảm
xúc (vui, buồn, giận, sợ, ...). Đây là bài toán khó do cảm xúc con người mang tính chủ quan, biểu hiện âm học của cùng
một cảm xúc khác nhau giữa người nói, ngôn ngữ, văn hóa, và các bộ dữ liệu công khai đều có kích thước nhỏ.

### 1.2 Ứng dụng thực tế

- **Tương tác người–máy**: trợ lý ảo, chatbot thoại điều chỉnh phản hồi theo trạng thái cảm xúc người dùng.
- **Chăm sóc khách hàng**: tổng đài tự động phát hiện khách hàng bức xúc để chuyển tiếp nhân viên hỗ trợ.
- **Y tế/sức khỏe tâm thần**: theo dõi trạng thái cảm xúc bệnh nhân qua giọng nói.
- **An toàn giao thông, giám sát**: phát hiện trạng thái căng thẳng, mệt mỏi của người lái xe.

### 1.3 Tóm tắt phương pháp

Project trích xuất hai loại đặc trưng âm học — MFCC (đặc trưng thống kê, không có chiều thời gian) và log-mel
spectrogram (biểu diễn ảnh 2D giữ thông tin thời gian–tần số) — rồi áp dụng ba mô hình: SVM (kernel RBF, tối ưu qua
GridSearchCV) trên MFCC, và ResNet34/DenseNet121 (transfer learning từ ImageNet) trên log-mel spectrogram. Kết quả được
so sánh trên cùng tập test để đánh giá đặc trưng và kiến trúc nào phù hợp hơn cho bài toán.

---

## 2. Các công trình liên quan

**DeepEmoNet (Vu, 2025)** thử nghiệm ba mô hình trên cùng RAVDESS+SAVEE: SVM với đặc trưng MFCC (chỉ lấy mean, 20
chiều), BiLSTM, và ResNet34 (log-mel spectrogram, có data augmentation), đạt tốt nhất 66.7% accuracy trên **validation
set**. Đây là công trình được project này tái hiện và mở rộng trực tiếp: project dùng thêm std của MFCC (40 chiều thay
vì 20), bổ sung DenseNet121, và báo cáo trên **test set** để tránh thiên vị do tối ưu trên validation.

**Sareen et al. (2025), "Speech Emotion Recognition using Mel Spectrogram and CNN"** chuyển âm thanh thành ảnh
mel-spectrogram rồi huấn luyện một CNN 2D từ đầu (không transfer learning), đánh giá riêng trên RAVDESS (70%), SAVEE (
60%), TESS (99.89%) và tập kết hợp (87%), có áp dụng data augmentation để tăng số mẫu. So với project này, hướng tiếp
cận đặc trưng (log-mel spectrogram dạng ảnh) giống nhau, nhưng project sử dụng transfer learning từ ImageNet thay vì
huấn luyện CNN từ đầu — ưu điểm là tận dụng được đặc trưng thị giác đã học sẵn trên dữ liệu nhỏ, nhược điểm là log-mel
spectrogram không phải ảnh tự nhiên nên pretraining không hoàn toàn phù hợp về lý thuyết.

**Nghiên cứu 1D-CNN feature-fusion (MDPI, 2025)** không dùng ảnh spectrogram mà tính trực tiếp chuỗi đặc trưng theo thời
gian (MFCC, mel-spectrogram, Chroma) rồi đưa vào CNN 1D, đạt 91.9% trên RAVDESS. Điểm mạnh của hướng này là giữ được
thông tin thời gian mà vẫn nhẹ hơn CNN 2D; đây là gợi ý cải tiến khả thi cho hướng SVM của project (vốn xóa bỏ thông tin
thời gian khi lấy mean/std).

Ba công trình cho thấy một mạch phát triển chung: từ đặc trưng thống kê không có thời gian (MFCC mean/std + SVM) → biểu
diễn ảnh 2D (log-mel spectrogram + CNN, có/không transfer learning) → biểu diễn chuỗi giữ thời gian (feature fusion +
CNN1D/LSTM). Project này nằm ở hai bước đầu của mạch đó và so sánh trực tiếp hiệu quả giữa chúng.

---

## 3. Phát biểu bài toán

### 3.1 Bài toán

**Input:** một file âm thanh (.wav) chứa câu nói của một diễn viên.
**Output:** một trong 8 nhãn cảm xúc: `neutral, calm, happy, sad, angry, fear, disgust, surprise`.

Đây là bài toán **phân loại đa lớp (multi-class classification)**, thuộc nhóm **supervised learning**.

### 3.2 Thuật toán

#### 3.2.1 MFCC + SVM

**Trích xuất đặc trưng:**

1. Load audio (`sr=16000`).
2. Tính 20 hệ số MFCC theo thời gian → ma trận `(20, T)`.
3. Lấy **mean** và **std** theo trục thời gian, ghép lại → vector 40 chiều/mẫu.
4. Chuẩn hóa bằng `StandardScaler` (fit trên train).

**Mô hình:** SVM kernel RBF. Quyết định phân loại dựa trên khoảng cách tới siêu phẳng tối ưu trong không gian đặc trưng
đã ánh xạ bởi kernel Gaussian:

```
K(x, x') = exp(-γ‖x - x'‖²)
```

Siêu tham số `C` (mức phạt lỗi) và `γ` được tìm bằng `GridSearchCV` (5-fold CV, tối ưu F1-weighted) trên lưới
`C∈{0.1,1,10,100}`, `γ∈{'scale','auto',0.001,0.01}`.

**Flow:** `audio → MFCC(20,T) → mean+std(40,) → StandardScaler → SVM-RBF → nhãn`

#### 3.2.2 Log-Mel Spectrogram + CNN (ResNet34 / DenseNet121)

**Trích xuất đặc trưng:**

1. Load audio (`sr=16000`).
2. Tính mel spectrogram (`n_mels=128, n_fft=1024, hop_length=512`) → chuyển sang thang dB (`power_to_db`).
3. Pad/trim trục thời gian về `128` frame → ảnh `(128, 128)`.
4. Chuẩn hóa min-max về `[0,1]` theo từng mẫu, nhân bản thành 3 kênh (giả RGB) để khớp input của CNN pretrained.
5. Chuẩn hóa tiếp theo mean/std **tính trên tập train** (không dùng thống kê ImageNet).

**Mô hình:** ResNet34 và DenseNet121 pretrained trên ImageNet, thay lớp phân loại cuối bằng `Dropout(0.5) → Linear(→8)`.
Huấn luyện toàn bộ mạng (fine-tune) với `Adam (lr=1e-3)`, `CosineAnnealingLR`, `CrossEntropyLoss`, `batch_size=64`,
`30 epoch`; checkpoint lưu tại epoch có validation loss thấp nhất.

**Flow:**
`audio → log-mel(128,128) → [0,1] norm → 3-channel → train-set norm → CNN pretrained (fc/classifier thay mới) → nhãn`

---

## 4. Thực nghiệm

### 4.1 Dữ liệu

| Tập dữ liệu | Số mẫu   | Nguồn                        | Số lớp                                   |
|-------------|----------|------------------------------|------------------------------------------|
| RAVDESS     | 1440     | 24 diễn viên (12 nam, 12 nữ) | 8 lớp (neutral: 96 mẫu, còn lại 192/lớp) |
| SAVEE       | 480      | 4 diễn viên nam              | 7 lớp (không có *calm*; neutral=120)     |
| **Tổng**    | **1920** | —                            | **8 lớp**                                |

**Thuộc tính dữ liệu:** file âm thanh `.wav`, mono, câu nói ngắn diễn theo kịch bản cố định, nhãn cảm xúc gán sẵn theo
tên file (RAVDESS: mã số trong filename; SAVEE: tiền tố ký tự, vd. `a`=angry).

**Vấn đề của dữ liệu:**

- Mất cân bằng lớp nhẹ: *neutral* của RAVDESS chỉ có 96 mẫu so với 192 của các lớp khác; SAVEE không có lớp *calm*.
- Kích thước nhỏ (1920 mẫu) → dễ overfitting với mô hình tham số lớn (CNN pretrained ~7–21M tham số).
- Dữ liệu acted speech (diễn viên đóng), không phải cảm xúc tự nhiên.

**Tiền xử lý:**

- MFCC: mean+std theo thời gian → 40 chiều, `StandardScaler`.
- Log-mel: chuẩn hóa `[0,1]` rồi chuẩn hóa lại theo mean/std của tập train, không dùng data augmentation (ảnh hoặc âm
  thanh).
- Không xử lý riêng imbalanced data (không oversampling/class weight); mất cân bằng ở mức nhẹ nên không bắt buộc.

**Chia dữ liệu:** Stratified split, `random_state=42`, tỉ lệ 80/10/10 (Train 1536 / Val 192 / Test 192).

> ⚠️ **Speaker leakage:** split thực hiện ngẫu nhiên theo *file*, không phải theo diễn viên (speaker-independent). Cùng
> một diễn viên có thể xuất hiện ở cả train/val/test → mô hình có thể học đặc trưng giọng người nói thay vì cảm xúc, khiến
> accuracy bị thổi phồng. Đây là hạn chế lớn nhất của thực nghiệm (xem mục 5).

### 4.2 Cấu hình thực nghiệm

| Thành phần   | MFCC + SVM             | ResNet34                   | DenseNet121                |
|--------------|------------------------|----------------------------|----------------------------|
| Đặc trưng    | MFCC mean+std (40-dim) | Log-mel (128×128×3)        | Log-mel (128×128×3)        |
| Chuẩn hóa    | StandardScaler         | [0,1] + train-set mean/std | [0,1] + train-set mean/std |
| Augmentation | —                      | —                          | —                          |
| Mô hình      | SVM RBF (GridSearchCV) | ResNet34 (pretrained)      | DenseNet121 (pretrained)   |
| Optimizer    | —                      | Adam + CosineAnnealingLR   | Adam + CosineAnnealingLR   |
| Epochs       | —                      | 30                         | 30                         |
| Batch size   | —                      | 64                         | 64                         |
| Số tham số   | ~vài nghìn (SVM)       | ~21.3M                     | ~7.0M                      |

### 4.3 Kết quả

| Mô hình                         | Accuracy (Test) | F1 (Test)             | Ghi chú             |
|---------------------------------|-----------------|-----------------------|---------------------|
| MFCC + SVM (baseline, mặc định) | 56.8%           | 0.5636 (weighted)     | trước GridSearch    |
| MFCC + SVM (tuned)              | **66.67%**      | **0.6669** (weighted) | C=10, gamma='scale' |
| Log-Mel + ResNet34              | **76.04%**      | **0.7588** (macro)    | seed=42, 30 epoch   |
| Log-Mel + DenseNet121           | **73.96%**      | **0.7386** (macro)    | seed=42, 30 epoch   |

**Per-class F1 — MFCC+SVM:** yếu nhất *sad* (0.56), *happy* (0.59); tốt nhất *surprise* (0.79), *calm* (0.76).
**Per-class F1 — ResNet34:** yếu nhất *disgust* (0.62); tốt nhất *surprise* (0.87), *angry* (0.86).
**Per-class F1 — DenseNet121:** yếu nhất *fear* (0.67), *sad* (0.68); tốt nhất *calm* (0.84, recall 0.95).

**Phân tích:**

- **SVM overfit nghiêm trọng nhất theo tỉ lệ tham số**: train 99.48% vs test 66.67%, do lấy mean/std xóa bỏ động học
  thời gian, mô hình dễ học thuộc dữ liệu train.
- **ResNet34 tốt nhất trong 3 mô hình** (accuracy + F1-macro cao nhất), vượt trội ở *angry*, *surprise*, nhưng cũng
  overfit gần tuyệt đối (train ~100% vs test 76.04%) và nặng tham số nhất (~21.3M).
- **DenseNet121** ít tham số hơn ResNet34 (~7M) nhưng kết quả thấp hơn ở seed này (73.96% vs 76.04%); tốt nhất ở *calm*
  nhờ dense connections tái sử dụng đặc trưng tốt cho lớp có ít mẫu.
- Log-mel + CNN pretrained vượt SVM ~7–9 điểm % accuracy, cho thấy giữ thông tin thời gian–tần số dạng ảnh 2D giúp ích
  hơn đặc trưng thống kê phẳng, dù phải đánh đổi bằng chi phí tính toán và overfitting nặng hơn.
- Thứ hạng ResNet34 vs DenseNet121 **nhạy với seed**: lần chạy khác (không cố định seed) DenseNet121 thắng (77.08% vs
  75.00%); kết luận "model nào tốt hơn" ở đây chỉ đúng cho seed=42.

---

## 5. Kết luận

Ba mô hình cho bài toán SER trên RAVDESS+SAVEE (1920 mẫu, 8 lớp): MFCC+SVM (66.67% test, đơn giản, không cần GPU, nhưng
overfit nặng và mất thông tin thời gian), ResNet34 transfer learning (76.04% test — tốt nhất ở seed=42, nhưng nặng tham
số nhất và overfit gần tuyệt đối), DenseNet121 transfer learning (73.96% test, ít tham số hơn nhưng kết quả thấp hơn
ResNet34 ở seed này).

**Ưu điểm chung:** log-mel spectrogram + CNN pretrained cho kết quả tốt hơn MFCC+SVM đáng kể mà không cần thiết kế đặc
trưng thủ công phức tạp.
**Nhược điểm chung:** cả ba mô hình đều overfit (train ≫ test), dataset nhỏ, và quan trọng nhất — split hiện tại **không
speaker-independent** nên accuracy báo cáo có khả năng bị thổi phồng.

**Hướng phát triển:**

- Speaker-independent split (leave-speakers-out) để đánh giá đúng khả năng tổng quát hóa.
- Chạy nhiều seed (≥5), báo cáo mean ± std thay vì một lần chạy.
- Fine-tune pretrained speech model (wav2vec 2.0, HuBERT) thay cho vision model trên log-mel.
- Thêm delta/delta-delta MFCC hoặc chuyển sang biểu diễn giữ thời gian (feature fusion + CNN1D/LSTM) cho nhánh SVM.
- Data augmentation âm thanh (pitch shift, time stretch, noise injection) để giảm overfitting.

---

## Tài liệu tham khảo

- Vu, T., DeepEmoNet: Building Machine Learning Models for Automatic Emotion Recognition in Human Speeches, arXiv:
  2509.00025, 2025.
- Sareen, V., et al., Speech Emotion Recognition using Mel Spectrogram and Convolutional Neural Networks (CNN), Procedia
  Computer Science, vol. 258, 2025, pp. 3693-3702.
- Speech Emotion Recognition on MELD and RAVDESS Datasets Using CNN (feature-fusion 1D-CNN), Information, MDPI, vol. 16,
  no. 7, 2025, p. 518.
- Livingstone, S. R., Russo, F. A., The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), 2018.
- Jackson, P., Haq, S., Surrey Audio-Visual Expressed Emotion (SAVEE) Database, 2014.
- Huang, G., et al., Densely Connected Convolutional Networks, CVPR 2017.
- He, K., et al., Deep Residual Learning for Image Recognition, CVPR 2016.
- McFee, B., et al., librosa: Audio and music signal analysis in Python, SciPy 2015.

## Phân công công việc

- Phạm Tấn Đức (23130068):  Log-Mel + ResNet34/DenseNet121 (trích xuất đặc trưng, huấn luyện, đánh giá) · Viết mục 2 (Công trình liên quan), mục 3.2.2 (thuật toán CNN) · Mục 4.3 (Kết quả)
- Trần Lê Công Hiếu (23130108): MFCC + SVM (trích xuất đặc trưng, GridSearchCV, đánh giá) · Viết mục 1 (Giới thiệu), mục 3.1+3.2.1 (bài toán + thuật toán SVM) · Mục 4.1 (Dữ liệu)

Link code: https://github.com/AzureDream1811/SER_ravdess_savee