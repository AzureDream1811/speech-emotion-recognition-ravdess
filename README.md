# Speech Emotion Recognition trên RAVDESS + SAVEE: So sánh MFCC+SVM, ResNet34 và DenseNet121

---

## Abstract

Bài báo cáo này trình bày kết quả thực nghiệm của ba mô hình học máy áp dụng cho bài toán nhận dạng cảm xúc từ giọng nói (Speech Emotion Recognition — SER): **(1)** SVM với đặc trưng MFCC, **(2)** ResNet34 (transfer learning từ ImageNet) với Log-Mel Spectrogram, và **(3)** DenseNet121 (transfer learning từ ImageNet) với Log-Mel Spectrogram. Cả ba mô hình được huấn luyện và đánh giá trên cùng một tập dữ liệu kết hợp RAVDESS và SAVEE (1920 mẫu, 8 lớp cảm xúc), với cùng chiến lược chia dữ liệu 80/10/10, seed cố định (`SEED=42`). Mô hình ResNet34 đạt kết quả tốt nhất với **accuracy 76.04%** và **F1-macro 0.7588** trên tập test.

---

## 1. Giới thiệu

Nhận dạng cảm xúc từ giọng nói là một bài toán quan trọng trong xử lý ngôn ngữ nói, với ứng dụng trong giao tiếp người–máy, y tế, và giám sát cảm xúc. Bài toán gặp khó khăn do sự mơ hồ và phức tạp của cảm xúc con người, cũng như sự thiếu hụt dữ liệu huấn luyện.

Báo cáo này tái hiện và mở rộng phương pháp của DeepEmoNet (Vu, 2025), so sánh ba pipeline xử lý âm thanh khác nhau về đặc trưng đầu vào và kiến trúc mô hình. Điểm khác biệt chính so với DeepEmoNet:

- Sử dụng **mean + std** của MFCC (40 chiều) thay vì chỉ mean (20 chiều)
- So sánh hai kiến trúc CNN: **ResNet34** và **DenseNet121** với transfer learning
- Đánh giá trên **test set** thay vì validation set, đảm bảo ước lượng không thiên vị

---

## 2. Related Works

DeepEmoNet (Vu, 2025) sử dụng cùng dataset RAVDESS+SAVEE, thử nghiệm SVM, LSTM, và ResNet34 với transfer learning, đạt 66.7% accuracy (validation set). Đây là công trình duy nhất được báo cáo này trực tiếp tái hiện và mở rộng (xem mục 1).

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

**Dữ liệu:** RAVDESS + SAVEE (1920 mẫu), chia 80/10/10.

---

### 3.2 Mô hình 2 — Log-Mel Spectrogram + ResNet34

**Đặc trưng:** Log-Mel spectrogram (N_MELS=128, IMG_SIZE=128×128), chuẩn hóa [-80,0]→[0,1], sau đó Normalize theo mean/std tính trực tiếp trên tập train (không dùng ImageNet stats), nhân lên 3 kênh để phù hợp đầu vào RGB của ResNet34.

**Kiến trúc:** ResNet34 pretrained trên ImageNet. Thay thế layer `fc` cuối:

```
Dropout(p=0.5) → Linear(512 → 8)
```

Tổng số tham số: **~21.3M**.

**Augmentation:** Không dùng (không augment ảnh, không Mixup).

**Huấn luyện:**
- **Adam** lr=1e-3
- **CosineAnnealingLR**
- **CrossEntropyLoss** (hard labels)
- **30 epochs**
- Checkpoint tại epoch có val loss thấp nhất

**Dữ liệu:** RAVDESS + SAVEE (1920 mẫu), chia 80/10/10.

---

### 3.3 Mô hình 3 — Log-Mel Spectrogram + DenseNet121

**Đặc trưng:** Giống ResNet34 — Log-Mel spectrogram (N_MELS=128, IMG_SIZE=128×128), chuẩn hóa [-80,0]→[0,1] rồi Normalize theo mean/std tính trên tập train, nhân lên 3 kênh.

**Kiến trúc:** DenseNet121 pretrained trên ImageNet. Thay thế layer `classifier` cuối:

```
Dropout(p=0.5) → Linear(1024 → 8)
```

**Augmentation & Huấn luyện:** Giống ResNet34 (xem mục 3.2) — không augment, không Mixup.

**Dữ liệu:** RAVDESS + SAVEE (1920 mẫu), chia 80/10/10.

---

## 4. Thực nghiệm

### 4.1 Dữ liệu

| Tập dữ liệu | Mẫu | Giới tính | Cảm xúc |
|---|---|---|---|
| RAVDESS | 1440 | 24 diễn viên (12 nam, 12 nữ) | 8 lớp (neutral có 96 mẫu, còn lại 192) |
| SAVEE | 480 | 4 diễn viên nam | 7 lớp (không có calm; neutral=120) |
| **Tổng** | **1920** | — | **8 lớp** |

Cả ba mô hình (SVM, ResNet34, DenseNet121) đều sử dụng cùng tập dữ liệu kết hợp RAVDESS + SAVEE (**1920 mẫu**).

**Chia dữ liệu:** Stratified split, random_state=42.

> ⚠️ **Cảnh báo speaker leakage:** split thực hiện ngẫu nhiên theo *file*, **không phải speaker-independent**. Cùng một diễn viên xuất hiện ở cả train, val và test → mô hình có thể học đặc trưng giọng người nói thay vì cảm xúc, làm accuracy bị thổi phồng. SER chuẩn cần chia theo diễn viên (leave-speakers-out). Đây là hạn chế lớn nhất (xem mục 5.4).

| Tập | Mẫu | Tỉ lệ |
|---|---|---|
| Train | 1536 | 80% |
| Validation | 192 | 10% |
| Test | 192 | 10% |

---

### 4.2 Chi tiết thực nghiệm

| Thành phần | MFCC + SVM | ResNet34 | DenseNet121 |
|---|---|---|---|
| **Dữ liệu** | RAVDESS + SAVEE (1920) | RAVDESS + SAVEE (1920) | RAVDESS + SAVEE (1920) |
| **Đặc trưng** | MFCC mean+std (40-dim) | Log-mel (128×128 × 3ch) | Log-mel (128×128 × 3ch) |
| **Chuẩn hóa** | StandardScaler | [0,1] + train-set mean/std | [0,1] + train-set mean/std |
| **Augmentation** | — | — | — |
| **Loss** | — | CrossEntropy (hard labels) | CrossEntropy (hard labels) |
| **Optimizer** | GridSearch (SVM) | Adam + CosineAnnealingLR | Adam + CosineAnnealingLR |
| **Epochs** | — | 30 | 30 |
| **Batch size** | — | 64 | 64 |
| **Pretrained** | ✗ | ✓ ImageNet | ✓ ImageNet |
| **Tham số** | ~few K (SVM) | ~21.3M | ~7.0M |

---

### 4.3 Phương pháp đánh giá

Sử dụng **accuracy** và **F1-score** để đánh giá hiệu suất mô hình. Tất cả kết quả được báo cáo trên **test set** — tập dữ liệu chưa được sử dụng trong bất kỳ quá trình huấn luyện hay chọn hyperparameter nào.

- SVM: F1 **weighted** (phù hợp với class imbalance)
- ResNet34, DenseNet121: F1 **macro** (trọng số bằng nhau cho mọi lớp)

---

### 4.4 Kết quả

#### Bảng tổng hợp

| Mô hình | Dữ liệu | Accuracy (Test) | F1 (Test) | Ghi chú |
|---|---|---|---|---|
| MFCC + SVM (baseline) | RAVDESS+SAVEE | 56.8% | 0.5636 | Trước GridSearch |
| MFCC + SVM (tuned) | RAVDESS+SAVEE | **66.67%** | **0.6669** (weighted) | C=10, gamma='scale' |
| Log-Mel + ResNet34 | RAVDESS+SAVEE | **76.04%** | **0.7588** (macro) | 30 epochs fine-tune, seed=42 |
| Log-Mel + DenseNet121 | RAVDESS+SAVEE | **73.96%** | **0.7386** (macro) | 30 epochs fine-tune, seed=42 |

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

---

#### Per-class Performance — ResNet34 (Test Set)

| Cảm xúc | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| neutral | 0.70 | 0.73 | 0.71 | 22 |
| calm | 0.84 | 0.84 | 0.84 | 19 |
| happy | 0.74 | 0.68 | 0.71 | 25 |
| sad | 0.70 | 0.76 | 0.73 | 25 |
| angry | **0.85** | **0.88** | **0.86** | 25 |
| fear | 0.72 | 0.72 | 0.72 | 25 |
| disgust | 0.70 | 0.56 | 0.62 | 25 |
| surprise | 0.83 | **0.92** | **0.87** | 26 |
| **macro avg** | 0.76 | 0.76 | 0.76 | 192 |

---

#### Per-class Performance — DenseNet121 (Test Set)

| Cảm xúc | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| neutral | 0.67 | 0.73 | 0.70 | 22 |
| calm | 0.75 | **0.95** | **0.84** | 19 |
| happy | 0.76 | 0.64 | 0.70 | 25 |
| sad | 0.68 | 0.68 | 0.68 | 25 |
| angry | **0.81** | 0.84 | 0.82 | 25 |
| fear | 0.70 | 0.64 | 0.67 | 25 |
| disgust | 0.76 | 0.64 | 0.70 | 25 |
| surprise | 0.79 | 0.85 | 0.81 | 26 |
| **macro avg** | 0.74 | 0.75 | 0.74 | 192 |

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

### 5.2 ResNet34 (Transfer Learning)

**Điểm mạnh:**
- Transfer learning từ ImageNet cho phép học được low-level visual patterns từ spectrogram
- Kiến trúc residual giúp huấn luyện ổn định
- Vượt SVM về accuracy (76.04% vs 66.67%), cân bằng tốt giữa các lớp — tốt nhất ở *angry* (F1=0.86) và *surprise* (F1=0.87)
- **Kết quả tốt nhất trong 3 model** (accuracy + F1 macro cao nhất trên test), dù ít tham số hơn không phải lý do — ngược lại, 21.3M tham số (nhiều hơn DenseNet121) nhưng vẫn overfit ít bị ảnh hưởng hơn ở seed này

**Điểm yếu:**
- Overfitting nghiêm trọng: train 100% vs test 76.04%
- Yếu nhất ở *disgust* (F1=0.62, recall chỉ 0.56)
- 21.3M tham số — nặng hơn đáng kể so với DenseNet121 (~7M) dù kết quả tốt hơn, không hiệu quả về mặt tham số

### 5.3 DenseNet121 (Transfer Learning)

**Điểm mạnh:**
- Ít tham số hơn ResNet34 (~7M vs ~21.3M)
- Dense connections giúp tái sử dụng đặc trưng
- Tốt nhất ở *calm* (recall 0.95, F1=0.84)

**Điểm yếu:**
- Overfitting: train 100% >> test 73.96%
- **Kết quả thấp hơn ResNet34** ở seed này (73.96% vs 76.04% accuracy, 0.7386 vs 0.7588 F1 macro) — đảo ngược so với lần chạy chưa cố định seed trước đó
- Log-mel spectrogram ≠ ảnh tự nhiên → ImageNet pretraining không hoàn toàn phù hợp về mặt lý thuyết
- Yếu ở *happy*, *fear*, *disgust* (F1 quanh 0.67-0.70)

### 5.4 Hạn chế nghiên cứu

Các kết quả trên cần được diễn giải thận trọng vì những hạn chế sau:

- **Speaker leakage (nghiêm trọng nhất):** dữ liệu chia ngẫu nhiên theo file, cùng diễn viên nằm ở cả train và test. Vì vậy accuracy nhiều khả năng bị thổi phồng; đánh giá SER đúng chuẩn phải dùng **speaker-independent split** (leave-speakers-out).
- **Test set nhỏ (192 mẫu) + chỉ chạy 1 seed:** đã cố định `SEED=42` cho `random/numpy/torch/cuda`, nhưng **kết quả rất nhạy với seed** — lần chạy trước (chưa cố định seed) DenseNet121 thắng ResNet34 (77.08% vs 75.00%), lần chạy này (seed cố định) thì ngược lại (73.96% vs 76.04%). Kết luận "model nào tốt hơn" không đáng tin nếu chỉ dựa trên 1 lần chạy — cần chạy nhiều seed và báo cáo mean ± std hoặc k-fold.
- **Phiên bản notebook:** dự án hiện chỉ dùng **`logmelspec_CNN_v2`** cho các mô hình CNN (v1/v3 chỉ là bản thử nghiệm trung gian, không dùng cho số liệu báo cáo).

---

## 6. Kết luận

Ba mô hình được phát triển cho bài toán SER trên RAVDESS+SAVEE:

1. **MFCC + SVM**: baseline đơn giản nhưng hiệu quả (66.67% test), phù hợp khi không có GPU
2. **ResNet34 + Transfer Learning**: mô hình tốt nhất trong lần chạy có seed cố định (76.04% test), vượt SVM và DenseNet121, nhưng vẫn overfitting nghiêm trọng và nặng tham số nhất (~21.3M)
3. **DenseNet121 + Transfer Learning**: kết quả thấp hơn ResNet34 ở seed này (73.96% test), dù ít tham số hơn (~7M)

> Lưu ý: thứ hạng ResNet34 vs DenseNet121 đã đảo ngược giữa 2 lần chạy (có/không cố định seed) — xem mục 5.4. Kết luận "model nào tốt hơn" ở đây chỉ đúng cho seed=42, chưa đủ cơ sở để khái quát.

Bước tiếp theo để cải thiện:
- Chạy nhiều seed (≥5) và báo cáo mean ± std để so sánh model công bằng, đáng tin hơn
- Fine-tune pretrained speech model (wav2vec 2.0, HuBERT) thay vì vision model
- Sử dụng speaker-independent split để đánh giá chính xác hơn
- Thêm delta/delta-delta MFCC cho pipeline SVM
- Tăng kích thước dataset (data augmentation âm thanh: pitch shift, time stretch, noise injection)

---

## Tài liệu tham khảo

- Vu, T. (2025). *DeepEmoNet: Building Machine Learning Models for Automatic Emotion Recognition in Human Speeches*. arXiv:2509.00025.
- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).
- Jackson, P., & Haq, S. (2014). Surrey Audio-Visual Expressed Emotion (SAVEE) Database.
- Huang, G., et al. (2017). Densely Connected Convolutional Networks. *CVPR 2017*.
- He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
- McFee, B., et al. (2015). librosa: Audio and music signal analysis in Python. *SciPy 2015*.