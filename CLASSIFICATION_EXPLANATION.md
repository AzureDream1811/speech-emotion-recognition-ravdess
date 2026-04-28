# Giải Thích Chi Tiết Quy Trình Huấn Luyện Mô Hình CNN Phân Loại Cảm Xúc

Đây là tài liệu giải thích chi tiết về mã nguồn trong file `3_CNN-classification.ipynb`. Mục tiêu là làm rõ từng bước, từ chuẩn bị dữ liệu đến huấn luyện và đánh giá mô hình, giúp những người mới bắt đầu có thể hiểu được quy trình.

## 1. Import Thư Viện

Bước đầu tiên là import tất cả các thư viện cần thiết.

```python
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torchvision import models
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import ttach as tta
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
```

- **`pandas`**: Dùng để đọc và xử lý dữ liệu từ file CSV.
- **`torch`, `torch.nn`, `optim`**: Là các thành phần cốt lõi của PyTorch, dùng để xây dựng, huấn luyện và tối ưu hóa mô hình mạng neural.
- **`torchvision`**: Cung cấp các mô hình, bộ dữ liệu và các phép biến đổi hình ảnh phổ biến.
- **`Dataset`, `DataLoader`**: Các lớp của PyTorch giúp quản lý và tải dữ liệu một cách hiệu quả.
- **`PIL.Image`**: Thư viện `Pillow` dùng để xử lý hình ảnh.
- **`sklearn.model_selection.train_test_split`**: Dùng để chia bộ dữ liệu thành các tập huấn luyện và kiểm thử.
- **`sklearn.metrics`**: Cung cấp các hàm để đánh giá hiệu suất mô hình (độ chính xác, F1-score, ...).
- **`ttach` (Test Time Augmentation)**: Một thư viện giúp cải thiện độ chính xác khi dự đoán trên tập kiểm thử.

## 2. Chuẩn Bị Dữ Liệu và Biến Đổi Hình Ảnh (Image Transformation)

### Cấu Hình (Config)

Chúng ta định nghĩa các tham số quan trọng.

```python
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
emotion_to_idx = { ... }
```

- **`IMG_SIZE`**: Kích thước ảnh đầu vào cho mô hình (224x224 pixels).
- **`BATCH_SIZE`**: Số lượng mẫu dữ liệu được xử lý trong một lần lặp huấn luyện.
- **`NUM_WORKERS`**: Số luồng xử lý song song để tải dữ liệu.
- **`emotion_to_idx`**: Một dictionary để ánh xạ tên các cảm xúc (nhãn dạng chữ) sang dạng số mà mô hình có thể hiểu được.

### Các Phép Biến Đổi (Transforms)

Đây là một bước cực kỳ quan trọng, nơi chúng ta định nghĩa cách xử lý và tăng cường dữ liệu hình ảnh.

```python
train_transform = transforms.Compose([...])
val_transform = transforms.Compose([...])
```

- **`transforms.Compose`**: Gộp nhiều phép biến đổi lại thành một chuỗi thực thi tuần tự.

#### `train_transform` (Dành cho tập huấn luyện)
Tập huấn luyện được áp dụng các kỹ thuật **Data Augmentation** (Tăng cường dữ liệu) để làm cho mô hình có khả năng tổng quát hóa tốt hơn, tránh học vẹt (overfitting).
- **`transforms.Resize`**: Thay đổi kích thước ảnh về `IMG_SIZE`.
- **`transforms.RandomAffine`**: Áp dụng các phép biến đổi hình học ngẫu nhiên như dịch chuyển (`translate`), co giãn (`scale`).
- **`transforms.ToTensor`**: Chuyển đổi ảnh (định dạng PIL) thành Tensor, là cấu trúc dữ liệu cơ bản của PyTorch.
- **`transforms.RandomErasing`**: Xóa một vùng hình chữ nhật ngẫu nhiên trên ảnh. Kỹ thuật này buộc mô hình phải học các đặc trưng từ nhiều phần khác nhau của ảnh.
- **`transforms.Normalize`**: Chuẩn hóa giá trị các pixel về một khoảng phân phối nhất định (thường là giá trị trung bình 0 và độ lệch chuẩn 1). Các giá trị `mean` và `std` này là giá trị chuẩn hóa phổ biến cho các mô hình được huấn luyện trước trên bộ dữ liệu ImageNet.

#### `val_transform` (Dành cho tập xác thực và kiểm thử)
Đối với dữ liệu xác thực và kiểm thử, chúng ta chỉ thực hiện các bước biến đổi cơ bản cần thiết để đưa ảnh về đúng định dạng đầu vào cho mô hình, **không** áp dụng các phép tăng cường ngẫu nhiên. Điều này đảm bảo rằng chúng ta đang đánh giá mô hình trên dữ liệu gốc một cách nhất quán.

## 3. Lớp `ImageDataset`

PyTorch yêu cầu chúng ta tạo một lớp `Dataset` tùy chỉnh để định nghĩa cách tải và truy xuất từng mẫu dữ liệu.

```python
class ImageDataset(Dataset):
    # ...
```

- **`__init__`**: Hàm khởi tạo, nhận vào DataFrame chứa đường dẫn ảnh và nhãn, cùng với các phép biến đổi.
- **`__len__`**: Trả về tổng số lượng mẫu trong bộ dữ liệu.
- **`__getitem__`**: Định nghĩa cách lấy một mẫu dữ liệu tại một chỉ số (`idx`) cụ thể. Nó sẽ đọc ảnh từ đường dẫn, chuyển đổi sang định dạng RGB, áp dụng các phép biến đổi đã định nghĩa, và trả về cặp `(image, label)`.

## 4. Tải Dữ Liệu (Load Data)

Tại đây, chúng ta chia dữ liệu và tạo các đối tượng `DataLoader`.

```python
# Chia dữ liệu
train_indices, val_indices = train_test_split(...)
train_subset = train_df.iloc[train_indices]
val_subset = train_df.iloc[val_indices]

# Tạo Dataset
train_dataset = ImageDataset(train_subset, ...)
val_dataset = ImageDataset(val_subset, ...)
test_dataset = ImageDataset(test_df, ...)

# Tạo DataLoader
train_loader = DataLoader(train_dataset, ...)
val_loader = DataLoader(val_dataset, ...)
test_loader = DataLoader(test_dataset, ...)
```

- **`train_test_split`**: Chúng ta chia tập `train_df` ban đầu thành hai phần: một tập huấn luyện mới (`train_subset`) và một tập xác thực (`val_subset`) với tỉ lệ 80:20. Tập xác thực rất quan trọng để theo dõi hiệu suất của mô hình trong quá trình huấn luyện và phát hiện sớm overfitting.
- **`DataLoader`**: Đây là một trình vòng lặp (iterator) mạnh mẽ của PyTorch. Nó nhận vào một `Dataset` và tự động tạo ra các "lô" (batch) dữ liệu, xáo trộn dữ liệu (với `shuffle=True` cho tập huấn luyện), và có thể sử dụng nhiều tiến trình để tải dữ liệu song song, giúp tăng tốc độ huấn luyện.

## 5. Xây Dựng Mô Hình CNN (ResNet)

### Chọn Thiết Bị (Device)

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
Dòng mã này tự động kiểm tra xem máy tính có GPU (với CUDA) hay không. Nếu có, `device` sẽ là "cuda", và mọi tính toán sẽ được thực hiện trên GPU để tăng tốc. Nếu không, `device` sẽ là "cpu".

### Kiến Trúc Mô Hình `resnet34`

Chúng ta sử dụng một kỹ thuật gọi là **Transfer Learning** (Học chuyển giao).

```python
def resnet34(num_classes=6, pretrained=True):
    # Tải mô hình ResNet-34 đã được huấn luyện trước trên ImageNet
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

    # Đóng băng các lớp đầu
    for param in model.parameters():
        param.requires_grad = False

    # Mở băng các lớp sau để tinh chỉnh (fine-tuning)
    for layer in [model.layer2, model.layer3, model.layer4]:
        for param in layer.parameters():
            param.requires_grad = True

    # Thay thế lớp phân loại cuối cùng
    in_features = model.fc.in_features
    model.fc = nn.Sequential(...)

    return model

model = resnet34().to(device)
```

- **Transfer Learning**: Thay vì huấn luyện một mô hình từ đầu (mất rất nhiều thời gian và dữ liệu), chúng ta tận dụng một mô hình `ResNet-34` đã được huấn luyện trên bộ dữ liệu khổng lồ ImageNet. Mô hình này đã học được các đặc trưng hình ảnh rất tốt (như cạnh, góc, kết cấu).
- **Đóng băng (Freeze)**: Chúng ta "đóng băng" các lớp đầu của mô hình (`param.requires_grad = False`). Điều này có nghĩa là trọng số của các lớp này sẽ không được cập nhật trong quá trình huấn luyện. Chúng ta giữ lại các đặc trưng cơ bản mà mô hình đã học.
- **Tinh chỉnh (Fine-tuning)**: Chúng ta "mở băng" các lớp sau (`param.requires_grad = True`) để chúng có thể được "tinh chỉnh" lại cho phù hợp với bộ dữ liệu cảm xúc của chúng ta.
- **Thay thế lớp phân loại**: Lớp `fc` (fully connected) cuối cùng của ResNet gốc được thiết kế để phân loại 1000 lớp của ImageNet. Chúng ta thay thế nó bằng một chuỗi các lớp `nn.Sequential` mới, được thiết kế riêng cho bài toán của mình (phân loại 6 loại cảm xúc). Lớp này bao gồm:
    - `nn.Linear`: Các lớp tuyến tính.
    - `nn.ReLU`: Hàm kích hoạt phi tuyến.
    - `nn.BatchNorm1d`: Chuẩn hóa batch, giúp ổn định quá trình huấn luyện.
    - `nn.Dropout`: Một kỹ thuật chính quy hóa khác, ngẫu nhiên "tắt" một vài nơ-ron trong quá trình huấn luyện để tránh overfitting.
- **`.to(device)`**: Di chuyển toàn bộ mô hình lên thiết bị đã chọn (GPU hoặc CPU).

### Cấu Hình Huấn Luyện

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(...)
scheduler = optim.lr_scheduler.OneCycleLR(...)
```

- **`criterion` (Hàm mất mát)**: `CrossEntropyLoss` là hàm mất mát tiêu chuẩn cho bài toán phân loại đa lớp. `label_smoothing` là một kỹ thuật giúp mô hình bớt "tự tin" thái quá và tổng quát hóa tốt hơn.
- **`optimizer` (Trình tối ưu hóa)**: `AdamW` là một biến thể cải tiến của trình tối ưu hóa Adam, thường cho kết quả tốt. Nó sẽ cập nhật trọng số của mô hình dựa trên gradient tính được từ hàm mất mát.
- **`scheduler` (Bộ điều chỉnh tốc độ học)**: `OneCycleLR` là một bộ điều chỉnh tốc-độ-học (learning rate) hiện đại. Nó sẽ tự động tăng và giảm learning rate theo một chu kỳ trong quá trình huấn luyện, giúp mô hình hội tụ nhanh hơn và đạt hiệu suất tốt hơn.

## 6. Huấn Luyện và Đánh Giá

### Hàm `evaluate`

Hàm này dùng để đánh giá hiệu suất của mô hình trên một bộ dữ liệu (xác thực hoặc kiểm thử).

```python
def evaluate(loader, eval_model=None):
    eval_model.eval() # Chuyển mô hình sang chế độ đánh giá
    y_true = [] # Nhãn thật
    y_pred = [] # Nhãn dự đoán
    with torch.no_grad(): # Không tính toán gradient
        for images, labels in loader:
            # ... tính toán output, loss ...
            preds = torch.argmax(outputs, dim=1) # Lấy nhãn dự đoán
            # ... lưu lại nhãn ...
    # ... tính các chỉ số accuracy, f1, precision, recall ...
    return avg_loss, acc, f1, prec, rec
```
- **`model.eval()`**: Rất quan trọng! Lệnh này chuyển mô hình sang chế độ đánh giá. Ở chế độ này, các lớp như `Dropout` và `BatchNorm` sẽ hoạt động khác so với khi huấn luyện, đảm bảo kết quả đánh giá nhất quán.
- **`with torch.no_grad()`**: Tắt việc tính toán gradient, giúp tiết kiệm bộ nhớ và tăng tốc độ tính toán vì chúng ta không cần cập nhật trọng số ở bước này.

### Vòng Lặp Huấn Luyện

Đây là nơi quá trình học thực sự diễn ra.

```python
for epoch in range(EPOCHS):
    model.train() # Chuyển mô hình sang chế độ huấn luyện
    for images, labels in train_loader:
        # 1. Đưa dữ liệu lên device
        images = images.to(device)
        labels = labels.to(device)

        # 2. Xóa gradient cũ
        optimizer.zero_grad()

        # 3. Forward pass: Đưa ảnh qua mô hình để nhận output
        outputs = model(images)

        # 4. Tính loss
        loss = criterion(outputs, labels)

        # 5. Backward pass: Lan truyền ngược loss để tính gradient
        loss.backward()

        # 6. Cập nhật trọng số
        optimizer.step()
        scheduler.step() # Cập nhật learning rate

    # Đánh giá trên tập validation sau mỗi epoch
    val_loss, val_acc, ... = evaluate(val_loader)

    # Lưu lại mô hình tốt nhất
    if val_f1 > best_f1:
        torch.save(model.state_dict(), "best_resnet34_cremad.pth")

    # Early Stopping
    if no_improve >= PATIENCE:
        print("Early stopping triggered.")
        break
```
- **`model.train()`**: Chuyển mô hình về chế độ huấn luyện.
- **Vòng lặp `epoch`**: Một epoch là một lần mô hình duyệt qua toàn bộ tập dữ liệu huấn luyện.
- **Vòng lặp `batch`**: Xử lý từng lô dữ liệu.
- **Early Stopping**: Một kỹ thuật hữu ích để dừng quá trình huấn luyện nếu hiệu suất trên tập xác thực không được cải thiện sau một số lượng `PATIENCE` epoch nhất định. Điều này giúp tiết kiệm thời gian và tránh overfitting.

### Đánh Giá Cuối Cùng với TTA (Test Time Augmentation)

Sau khi huấn luyện xong, chúng ta tải lại mô hình tốt nhất đã lưu và đánh giá nó trên tập kiểm thử (test set) - là tập dữ liệu mà mô hình chưa từng thấy.

```python
# Tải mô hình tốt nhất
model.load_state_dict(torch.load("best_resnet34_cremad.pth"))

# Tạo mô hình TTA
tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.five_crop_transform(200, 200))

# Đánh giá trên tập test
test_loss, test_acc, ... = evaluate(test_loader, tta_model)
```
- **Test Time Augmentation (TTA)**: Thay vì dự đoán trên một ảnh gốc duy nhất, TTA tạo ra nhiều phiên bản biến đổi của ảnh đó (ví dụ: cắt 5 vùng khác nhau - 4 góc và trung tâm), sau đó đưa tất cả qua mô hình và lấy kết quả trung bình (hoặc theo đa số). Kỹ thuật này thường giúp tăng độ chính xác và độ ổn định của dự đoán.

Hy vọng lời giải thích chi tiết này sẽ giúp bạn hiểu rõ hơn về dự án!