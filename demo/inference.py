import numpy as np
import librosa
import torch
import torchvision.transforms as T

from model import model, DEVICE

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "surprise"]

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
IMG_SIZE = 128

# from training set (see logmelspec_CNN_v2.ipynb, cell 13)
NORM_MEAN = [0.2975, 0.2975, 0.2975]
NORM_STD = [0.3306, 0.3306, 0.3306]

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


def extract_logmelspec(path, sample_rate=SAMPLE_RATE, n_mels=N_MELS,
                        n_fft=N_FFT, hop_length=HOP_LENGTH):
    audio, sr = librosa.load(path, sr=sample_rate)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    return librosa.power_to_db(mel, ref=np.max)


def pad_or_trim(spec, max_steps=IMG_SIZE):
    t = spec.shape[1]
    if t >= max_steps:
        return spec[:, :max_steps]
    return np.pad(spec, ((0, 0), (0, max_steps - t)), mode="constant")


def spec_to_tensor(spec):
    spec_min, spec_max = spec.min(), spec.max()
    img = (spec - spec_min) / (spec_max - spec_min + 1e-8)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img.repeat(3, 1, 1)


def predict_emotion(audio_path):
    spec = extract_logmelspec(audio_path)
    spec = pad_or_trim(spec)
    tensor = spec_to_tensor(spec)
    tensor = transform(tensor).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)

    pred = output.argmax(1).item()
    return EMOTIONS[pred]