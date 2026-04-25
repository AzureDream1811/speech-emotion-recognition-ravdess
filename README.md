
# Speech Emotion Recognition (RAVDESS)

## Models

- ResNet18
- DenseNet
- VGG16

## Dataset

- RAVDESS

## Pipeline

Audio → mel Spectrogram → Model → Emotion

## Kết quả

| Model    | Accuracy |
| -------- | -------- |
| Resnet   | ...      |
| DenseNet | ...      |
| VGG16    | ...      |

## Vấn đề khi tiền xử lý dữ liệu

### 1. Time Duration khác nhau

Các audio có độ dài thời gian khác nhau dẫn đến mel spectrogram cũng có trục thời gian khác nhau:

| Audio     | Shape       |
| -----     | -----       |
| Audio 3s  | (128, 94)   |
| Audio 7s  | (128, 219)  |
| Audio 10s | (128, 313)  |

Các model CNN yêu cầu input phải có shape cố định để có thể batch training. Giải pháp là **Padding + Truncation** — chọn `target_time` dựa trên percentile 90 của dataset, sau đó:

- Nếu `time >= target_time` → cắt ở giữa để giữ phần trọng tâm
- Nếu `time < target_time` → pad zeros 2 bên cho cân đối

```python
def pad_or_truncate(mel, target_time=128):
    time = mel.shape[1]
    if time >= target_time:
        start = (time - target_time) // 2
        return mel[:, start:start + target_time]
    else:
        pad_left = (target_time - time) // 2
        pad_right = target_time - time - pad_left
        return np.pad(mel, ((0, 0), (pad_left, pad_right)))
```

### 2. Normalize mel spectrogram

Mel spectrogram sau khi convert sang dB có giá trị nằm trong khoảng âm (thường từ -80 đến 0 dB). Nếu không normalize, model sẽ khó hội tụ vì:

- Các giá trị quá lớn về magnitude làm gradient không ổn định
- Phân phối giá trị khác nhau giữa các file audio

Giải pháp dùng **Z-score normalization** (mean=0, std=1):
S_dB = (S_dB - np.mean(S_dB)) / (np.std(S_dB) + 1e-6)

```python
> Thêm `1e-6` vào mẫu số để tránh chia cho 0 khi std = 0.
```

### 3. Số channel không tương thích

Các model CNN (ResNet18, DenseNet, VGG16) mặc định nhận input **3 channel (RGB)**, trong khi mel spectrogram là ảnh **1 channel (grayscale)**. cách xử lý:

**Sửa lớp conv1 của model nhận 1 channel**:

```python
model.conv1 = nn.Conv2d(
    in_channels=1, out_channels=64,
    kernel_size=7, stride=2, padding=3, bias=False
)
```
