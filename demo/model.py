import torch
import torch.nn as nn
from torchvision.models import resnet34

NUM_CLASSES = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_resnet34(num_classes=NUM_CLASSES, dropout_p=0.5):
    model = resnet34(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


model = build_resnet34()
model.load_state_dict(
    torch.load("../src/cnn/checkpoints/ResNet34/best_model.pth", map_location=DEVICE)
)
model.to(DEVICE)
model.eval()