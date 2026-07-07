from io import BytesIO
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from torchvision.models import resnet34

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pth"

LABELS = ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "surprise"]
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
IMG_SIZE = 128

app = FastAPI(title="Speech Emotion Recognition Demo")


def build_model() -> torch.nn.Module:
    model = resnet34(weights=None)
    model.fc = nn.Sequential(  # type: ignore[assignment]
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, len(LABELS)),
    )
    return model


def load_model() -> torch.nn.Module:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {MODEL_PATH}")

    model = build_model()
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


MODEL = load_model()


def pad_or_trim(spec: torch.Tensor, max_steps: int = IMG_SIZE) -> torch.Tensor:
    if spec.shape[1] >= max_steps:
        return spec[:, :max_steps]
    pad_width = max_steps - spec.shape[1]
    return F.pad(spec, (0, pad_width))


def preprocess(audio_bytes: bytes) -> torch.Tensor:
    audio, sr = librosa.load(BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    if audio.size == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    spec = torch.tensor(log_mel, dtype=torch.float32)
    spec = pad_or_trim(spec, IMG_SIZE)

    spec_min = spec.amin()
    spec_max = spec.amax()
    spec = (spec - spec_min) / (spec_max - spec_min + 1e-8)
    spec = spec.unsqueeze(0).repeat(3, 1, 1)
    spec = (spec - 0.5) / 0.5
    return spec.unsqueeze(0)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>SER Demo</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 880px; margin: 40px auto; padding: 0 16px; line-height: 1.5; }
    .card { border: 1px solid #ddd; border-radius: 14px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,.04); }
    button { padding: 10px 16px; border: 0; border-radius: 10px; background: #2563eb; color: #fff; cursor: pointer; }
    button:disabled { background: #94a3b8; cursor: not-allowed; }
    pre { background: #0f172a; color: #e2e8f0; padding: 16px; border-radius: 12px; overflow: auto; }
    .muted { color: #64748b; }
  </style>
</head>
<body>
  <h1>Speech Emotion Recognition Demo</h1>
  <p class="muted">Upload a short audio file and get the predicted emotion.</p>
  <div class="card">
    <input id="file" type="file" accept="audio/*" />
    <div style="margin-top: 16px;">
      <button id="submit">Predict</button>
    </div>
    <p id="status" class="muted"></p>
    <pre id="output">No prediction yet.</pre>
  </div>
  <script>
    const fileInput = document.getElementById('file');
    const submitBtn = document.getElementById('submit');
    const statusEl = document.getElementById('status');
    const outputEl = document.getElementById('output');

    submitBtn.addEventListener('click', async () => {
      if (!fileInput.files.length) {
        statusEl.textContent = 'Please choose an audio file first.';
        return;
      }

      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      submitBtn.disabled = true;
      statusEl.textContent = 'Predicting...';

      try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || 'Prediction failed');
        }

        const entries = Object.entries(data);
        entries.sort((a, b) => b[1] - a[1]);
        const [label, score] = entries[0];
        statusEl.textContent = `Prediction: ${label} (${(score * 100).toFixed(2)}%)`;
        outputEl.textContent = JSON.stringify(data, null, 2);
      } catch (error) {
        statusEl.textContent = error.message;
        outputEl.textContent = 'Prediction failed.';
      } finally {
        submitBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    x = preprocess(audio_bytes)
    with torch.no_grad():
        logits = MODEL(x)
        probs = F.softmax(logits, dim=1).squeeze(0)

    return {label: round(float(prob), 4) for label, prob in zip(LABELS, probs.tolist())}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
