# Speech Emotion Recognition trên RAVDESS + SAVEE: So sánh MFCC+SVM, BiLSTM và ResNet34

---

## Abstract

Bài báo cáo này trình bày kết quả thực nghiệm của ba mô hình học máy áp dụng cho bài toán nhận dạng cảm xúc từ giọng nói (Speech Emotion Recognition — SER): **(1)** SVM với đặc trưng MFCC, **(2)** BiLSTM với Log-Mel Spectrogram, và **(3)** ResNet34 (transfer learning từ ImageNet) với Log-Mel Spectrogram. Cả ba mô hình được huấn luyện và đánh giá trên cùng một tập dữ liệu kết hợp RAVDESS và SAVEE (1920 mẫu, 8 lớp cảm xúc), với cùng chiến lược chia dữ liệu 80/10/10. Mô hình ResNet34 đạt kết quả tốt nhất với **accuracy 75.00%** và **F1-macro 0.7503** trên tập test, vượt qua kết quả tốt nhất được báo cáo trong DeepEmoNet (Vu, 2025) là 66.7% trên tập validation.

---

## 1. Giới thiệu

Nhận dạng cảm xúc từ giọng nói là một bài toán quan trọng trong xử lý ngôn ngữ nói, với ứng dụng trong giao tiếp người–máy, y tế, và giám sát cảm xúc. Bài toán gặp khó khăn do sự mơ hồ và phức tạp của cảm xúc con người, cũng như sự thiếu hụt dữ liệu huấn luyện.

Báo cáo này tái hiện và mở rộng phương pháp của DeepEmoNet (Vu, 2025), so sánh ba pipeline xử lý âm thanh khác nhau về đặc trưng đầu vào và kiến trúc mô hình. Điểm khác biệt chính so với DeepEmoNet:

- Sử dụng **mean + std** của MFCC (40 chiều) thay vì chỉ mean (20 chiều)
- BiLSTM được cải tiến với **Attention Pooling** và **SpecAugment**
- Đánh giá trên **test set** thay vì validation set, đảm bảo ước lượng không thiên vị

---

## 2. Related Works

Các nghiên cứu trước đây về SER đã sử dụng nhiều phương pháp khác nhau. Schuller et al. (2003) dùng Hidden Markov Model để phát hiện cảm xúc từ các đặc trưng giọng nói. Demircan và Kahramanli (2018) kết hợp MFCC với fuzzy C-means và kNN. Lim et al. (2016) áp dụng CNN và LSTM trên biểu diễn STFT, cho kết quả tốt hơn các phương pháp truyền thống. DeepEmoNet (Vu, 2025) sử dụng cùng dataset RAVDESS+SAVEE, thử nghiệm SVM, LSTM, và ResNet34 với transfer learning, đạt 66.7% accuracy (validation set).

---

## 3. Phương pháp

### 3.1 Mô hình 1 — MFCC + SVM

**Đặc trưng:** Với mỗi file âm thanh, trích xuất 20 MFCC coefficients, sau đó tính **mean** và **std** theo trục thời gian, ghép thành vector **40 chiều**. Chuẩn hóa bằng `StandardScaler` (fit trên train, transform val/test).

**Mô hình:** SVM nhân RBF, tối ưu siêu tham số bằng `GridSearchCV` với 5-fold cross-validation trên tập train:

| Siêu tham số | Lưới tìm kiếm |
|---|---|
| C | {0.1, 1, 10, 100} |
| gamma | {'scale', 'auto', 0.001, 0.01} |

Metric tối ưu: F1-weighted. Kết quả tốt nhất: **C=10, gamma='scale'**.

---

### 3.2 Mô hình 2 — Log-Mel Spectrogram + BiLSTM (v2)

**Đặc trưng:** Log-Mel spectrogram với N_MELS=128, n_fft=1024, hop_length=512. Mỗi file được pad/trim về độ dài cố định MAX_STEPS=128 (~4.1s). Chuẩn hóa log-dB [-80, 0] → [0, 1].

**Kiến trúc:**

```
Input (B, 128, 128)
  → BiLSTM × 2 (hidden=128, dropout=0.3, bidirectional) → (B, 128, 256)
  → AttentionPool (learnable soft-attention) → (B, 256)
  → BatchNorm1d(256)
  → Dropout(0.4)
  → Linear(256 → 8)
```

Tổng số tham số: **662,281**.

**Kỹ thuật:**
- **SpecAugment**: T_mask=20, F_mask=20, 2 masks mỗi loại (chỉ train)
- **Class-weighted CrossEntropyLoss**: bù cho sự mất cân bằng lớp (neutral có ít mẫu hơn)
- **Adam** lr=1e-3, weight_decay=1e-4
- **ReduceLROnPlateau**: factor=0.5, patience=5
- **Early stopping**: patience=15, dừng tại epoch 99

---

### 3.3 Mô hình 3 — Log-Mel Spectrogram + ResNet34

**Đặc trưng:** Log-Mel spectrogram (N_MELS=128, IMG_SIZE=128×128), chuẩn hóa [-80,0]→[0,1], nhân lên 3 kênh để phù hợp đầu vào RGB của ResNet34.

**Kiến trúc:** ResNet34 pretrained trên ImageNet. Thay thế layer `fc` cuối: Linear(512 → 8).

**Augmentation (chỉ train):**
- RandomHorizontalFlip
- RandomAffine(degrees=10, scale=0.9–1.1) — xoay và zoom
- ColorJitter(brightness=0.3, contrast=0.3) — thay đổi độ sáng
- **Mixup** (α=0.6): tạo tổ hợp lồi của cặp mẫu

**Huấn luyện:**
- **Adam** lr=1e-3
- **ExponentialLR**: gamma=0.9/epoch
- **CrossEntropyLoss** (soft labels qua Mixup)
- **30 epochs** (DeepEmoNet protocol cho pretrained model)
- Checkpoint tại epoch có val loss thấp nhất

---

## 4. Thực nghiệm

### 4.1 Dữ liệu

| Tập dữ liệu | Mẫu | Giới tính | Cảm xúc |
|---|---|---|---|
| RAVDESS | 1440 | 24 diễn viên (12 nam, 12 nữ) | 8 lớp (neutral có 96 mẫu, còn lại 192) |
| SAVEE | 480 | 4 diễn viên nam | 7 lớp (không có calm; neutral=120) |
| **Tổng** | **1920** | — | **8 lớp** |

> **[CẦN HÌNH: Biểu đồ phân phối cảm xúc (bar chart) của combined dataset — lấy từ cell "combine" trong cả 3 notebook]**

**Chia dữ liệu:** Stratified split, random_state=42.

| Tập | Mẫu | Tỉ lệ |
|---|---|---|
| Train | 1536 | 80% |
| Validation | 192 | 10% |
| Test | 192 | 10% |

So với DeepEmoNet sử dụng split 90/5/5 (1728/96/96), cách chia này giảm training data nhưng cho test set lớn hơn và đáng tin cậy hơn.

---

### 4.2 Chi tiết thực nghiệm

| Thành phần | MFCC + SVM | BiLSTM v2 | ResNet34 |
|---|---|---|---|
| **Đặc trưng** | MFCC mean+std (40-dim) | Log-mel (128×128) | Log-mel (128×128 × 3ch) |
| **Chuẩn hóa** | StandardScaler | [−80,0]→[0,1] | [0,1] + ImageNet norm |
| **Augmentation** | — | SpecAugment | Rotate/Zoom/Brightness + Mixup |
| **Loss** | — | Weighted CrossEntropy | CrossEntropy (soft) |
| **Optimizer** | GridSearch (SVM) | Adam + ReduceLROnPlateau | Adam + ExponentialLR |
| **Epochs** | — | ≤200, stopped @99 | 30 |
| **Batch size** | — | 64 | 64 |
| **Pretrained** | ✗ | ✗ | ✓ ImageNet |
| **Tham số** | ~few K (SVM) | 662,281 | 21.3M |

---

### 4.3 Phương pháp đánh giá

Sử dụng **accuracy** và **F1-score** để đánh giá hiệu suất mô hình. Tất cả kết quả được báo cáo trên **test set** — tập dữ liệu chưa được sử dụng trong bất kỳ quá trình huấn luyện hay chọn hyperparameter nào.

- SVM: F1 **weighted** (phù hợp với class imbalance)
- BiLSTM, ResNet34: F1 **macro** (trọng số bằng nhau cho mọi lớp)

---

### 4.4 Kết quả

#### Bảng tổng hợp

| Mô hình | Accuracy (Test) | F1 (Test) | Ghi chú |
|---|---|---|---|
| MFCC + SVM (baseline) | 56.8% | 0.5636 | Trước GridSearch |
| MFCC + SVM (tuned) | **66.67%** | **0.6669** (weighted) | C=10, gamma='scale' |
| Log-Mel + BiLSTM v2 | **63.54%** | **0.6394** (macro) | Stopped @epoch 99 |
| Log-Mel + ResNet34 | **75.00%** | **0.7503** (macro) | 30 epochs fine-tune |

#### So sánh với DeepEmoNet

| Mô hình | Accuracy | F1 | Tập đo | Nguồn |
|---|---|---|---|---|
| SVM (DeepEmoNet) | 51.7% | 0.509 | Validation | Vu, 2025 |
| LSTM (DeepEmoNet) | 52.8% | 0.497 | Validation | Vu, 2025 |
| CNN + TL (DeepEmoNet) | 57.3% | 0.528 | Validation | Vu, 2025 |
| CNN + TL + Aug (DeepEmoNet) | 66.7% | 0.631 | Validation | Vu, 2025 |
| **SVM (dự án này)** | **66.67%** | **0.6669** | **Test** | — |
| **BiLSTM v2 (dự án này)** | **63.54%** | **0.6394** | **Test** | — |
| **ResNet34 (dự án này)** | **75.00%** | **0.7503** | **Test** | — |

> ⚠️ **Lưu ý so sánh:** DeepEmoNet báo cáo trên validation set; dự án này báo cáo trên test set. Validation accuracy thường cao hơn test accuracy do model được lựa chọn dựa trên val loss. Nếu tính cùng điều kiện, kết quả của dự án này có khả năng cao hơn thực tế được trình bày.

---

#### Per-class Performance — MFCC + SVM (Test Set)

| Cảm xúc | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| neutral | 0.65 | 0.59 | 0.62 | 22 |
| calm | 0.65 | 0.89 | **0.76** | 19 |
| happy | 0.55 | 0.64 | 0.59 | 25 |
| sad | 0.56 | 0.56 | 0.56 | 25 |
| angry | 0.71 | 0.68 | 0.69 | 25 |
| fear | 0.65 | 0.68 | 0.67 | 25 |
| disgust | 0.75 | 0.60 | 0.67 | 25 |
| surprise | **0.86** | 0.73 | **0.79** | 26 |
| **macro avg** | 0.67 | 0.67 | 0.67 | 192 |

> **[CẦN HÌNH: Confusion matrix của SVM — từ cell "conf-matrix" trong notebook mfcc_SVM]**

---

#### Per-class Performance — BiLSTM v2 (Test Set)

> **[CẦN HÌNH / SỐ: Classification report của BiLSTM v2 — chạy lại notebook và thêm cell `classification_report`]**

> **[CẦN HÌNH: Confusion matrix của BiLSTM v2]**

> **[CẦN HÌNH: Training curves của BiLSTM v2 (Loss / Accuracy / LR) — từ cell "plots" trong notebook logmelspec_LSTM]**

---

#### Per-class Performance — ResNet34 (Test Set)

> **[CẦN HÌNH / SỐ: Classification report của ResNet34 — từ cell "class-report" trong notebook logmelspec_CNN_resnet34]**

> **[CẦN HÌNH: Confusion matrix của ResNet34 — từ cell "conf-matrix"]**

> **[CẦN HÌNH: Training curves của ResNet34 (Loss / Accuracy / LR) — đã có tại `result/training_curves_resnet34.png`]**

---

#### Tổng hợp kết quả theo lớp cảm xúc

> **[CẦN HÌNH: Bar chart so sánh F1 per-class của 3 mô hình — vẽ bằng seaborn/matplotlib từ 3 classification report]**

---

## 5. Phân tích

### 5.1 MFCC + SVM

**Điểm mạnh:**
- Đơn giản, nhanh, không cần GPU
- Test accuracy (66.67%) vượt xa SVM của DeepEmoNet (51.7% val) nhờ dùng mean+std thay vì chỉ mean
- Kết quả tốt trên *calm* (F1=0.76) và *surprise* (F1=0.79)

**Điểm yếu:**
- Train accuracy 99.48% vs Test 66.67% → **overfitting nghiêm trọng** (SVM memorize training data)
- Mất thông tin thời gian: lấy mean/std xóa bỏ temporal dynamics của giọng nói
- Khó cải thiện thêm mà không đổi sang đặc trưng tốt hơn

### 5.2 BiLSTM v2

**Điểm mạnh:**
- Xử lý được chuỗi thời gian (temporal modeling)
- Attention Pooling cải thiện đáng kể so với last-timestep (không overfitting như LSTM gốc của DeepEmoNet)
- Chỉ 662K tham số, nhẹ nhàng, không cần pretrained

**Điểm yếu:**
- Test accuracy (63.54%) thấp hơn SVM → cần nhiều dữ liệu hơn để BiLSTM thể hiện ưu thế
- Training lâu (99 epochs), không ổn định trong giai đoạn đầu

### 5.3 ResNet34 (Transfer Learning)

**Điểm mạnh:**
- Kết quả tốt nhất: **75.0% accuracy**, **F1=0.7503**
- Transfer learning từ ImageNet cho phép học được low-level visual patterns từ spectrogram
- Mixup + image augmentation giảm overfitting hiệu quả (train 94.5% vs test 75.0%, gap hợp lý)
- Val ≈ Test (77% vs 75%) → model generalizes tốt

**Điểm yếu:**
- 21.3M tham số, cần GPU
- Overfitting nhẹ: train 94.5% >> test 75.0%
- Log-mel spectrogram ≠ ảnh tự nhiên → ImageNet pretraining không hoàn toàn phù hợp về mặt lý thuyết, dù thực tế hoạt động tốt

---

## 6. Kết luận

Ba mô hình được phát triển cho bài toán SER trên RAVDESS+SAVEE:

1. **MFCC + SVM**: baseline đơn giản nhưng hiệu quả (66.67% test), phù hợp khi không có GPU
2. **BiLSTM v2**: mô hình sequence learning với attention, kết quả trung bình (63.54% test) nhưng kiến trúc có khả năng mở rộng tốt
3. **ResNet34 + Transfer Learning**: mô hình tốt nhất (75.00% test), vượt kết quả được báo cáo trong DeepEmoNet trên cùng dataset

Bước tiếp theo để cải thiện:
- Fine-tune pretrained speech model (wav2vec 2.0, HuBERT) thay vì vision model
- Kết hợp CNN và LSTM (CNN để trích xuất đặc trưng, LSTM để mô hình chuỗi)
- Thêm delta/delta-delta MFCC cho pipeline SVM
- Tăng kích thước dataset (data augmentation âm thanh: pitch shift, time stretch, noise injection)

---

## Tài liệu tham khảo

- Vu, T. (2025). *DeepEmoNet: Building Machine Learning Models for Automatic Emotion Recognition in Human Speeches*. arXiv:2509.00025.
- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).
- Jackson, P., & Haq, S. (2014). Surrey Audio-Visual Expressed Emotion (SAVEE) Database.
- Park, D. S., et al. (2019). SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. *Interspeech 2019*.
- Zhang, H., et al. (2018). mixup: Beyond Empirical Risk Minimization.
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
- McFee, B., et al. (2015). librosa: Audio and music signal analysis in Python. *SciPy 2015*.
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*.
