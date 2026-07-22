"""
=====================================================================
 Skin Cancer Classification — single-file Flask app
=====================================================================
All-in-one version: backend + HTML/CSS/JS UI combined into one file.
Built from skin-cancer-classification-project-with-pytorch.ipynb

Model : timm "rexnet_150", fine-tuned on 7 skin-lesion classes.
Weights: put your trained file at  saved_models/cancer_best_model.pth
         (this is the file produced by trainer.save_best_model()
         in the notebook).

Setup:
    pip install flask torch torchvision timm pillow
    python app.py
    -> open http://localhost:5000

API:
    POST /predict   multipart form, field name "file" (jpg/png)
    GET  /health     model load status
=====================================================================
"""

import os
import io

import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms as T
from flask import Flask, request, jsonify, render_template_string

# ----------------------------------------------------------------------
# Config — must match the notebook
# ----------------------------------------------------------------------
MODEL_NAME = "rexnet_150"
IM_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Class order must match the `classes` dict produced by CustomDataset:
# {'mel': 0, 'vasc': 1, 'df': 2, 'nv': 3, 'bkl': 4, 'akiec': 5, 'bcc': 6}
CLASSES = ["mel", "vasc", "df", "nv", "bkl", "akiec", "bcc"]

CLASS_INFO = {
    "mel":   {"name": "Melanoma", "risk": "high"},
    "vasc":  {"name": "Vascular lesion", "risk": "low"},
    "df":    {"name": "Dermatofibroma", "risk": "low"},
    "nv":    {"name": "Melanocytic nevus", "risk": "low"},
    "bkl":   {"name": "Benign keratosis-like lesion", "risk": "low"},
    "akiec": {"name": "Actinic keratosis / intraepithelial carcinoma", "risk": "medium"},
    "bcc":   {"name": "Basal cell carcinoma", "risk": "high"},
}

MODEL_PATH = os.path.join("saved_models", "cancer_best_model.pth")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH_MB = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------------------
# HTML / CSS / JS — embedded as one template string
# ----------------------------------------------------------------------
PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Skin Cancer Classification</title>
<style>
  :root {
    --bg: #0f1115;
    --card: #171a21;
    --border: #2a2e37;
    --text: #e8e9ec;
    --muted: #9aa1ac;
    --accent: #4f8cff;
    --high: #ff5c5c;
    --medium: #ffb74f;
    --low: #4fd18b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    justify-content: center;
    padding: 40px 20px;
  }
  .wrap { width: 100%; max-width: 640px; }
  h1 { font-size: 1.5rem; margin-bottom: 4px; }
  p.subtitle { color: var(--muted); margin-top: 0; margin-bottom: 24px; }
  .warning {
    background: #3a2320; border: 1px solid #7a3b32; color: #ffb4a8;
    padding: 12px 16px; border-radius: 10px; margin-bottom: 20px; font-size: 0.9rem;
  }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 24px;
  }
  #dropzone {
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 32px;
    text-align: center;
    cursor: pointer;
    transition: border-color .15s ease, background .15s ease;
  }
  #dropzone.dragover { border-color: var(--accent); background: #14213a33; }
  #dropzone p { margin: 8px 0 0; color: var(--muted); font-size: 0.9rem; }
  #fileInput { display: none; }
  #preview { max-width: 100%; max-height: 260px; border-radius: 10px; display: none; margin: 0 auto 16px; }
  button#submitBtn {
    width: 100%; margin-top: 16px; padding: 12px; border: none; border-radius: 10px;
    background: var(--accent); color: white; font-size: 1rem; font-weight: 600;
    cursor: pointer; transition: opacity .15s ease;
  }
  button#submitBtn:disabled { opacity: 0.5; cursor: not-allowed; }
  #results { margin-top: 24px; display: none; }
  .result-row {
    display: flex; align-items: center; gap: 12px; padding: 10px 0;
    border-bottom: 1px solid var(--border);
  }
  .result-row:last-child { border-bottom: none; }
  .result-label { flex: 0 0 190px; font-size: 0.9rem; }
  .result-code { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; }
  .bar-track { flex: 1; height: 8px; background: #2a2e37; border-radius: 4px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; }
  .risk-high .bar-fill { background: var(--high); }
  .risk-medium .bar-fill { background: var(--medium); }
  .risk-low .bar-fill { background: var(--low); }
  .result-pct { flex: 0 0 54px; text-align: right; font-variant-numeric: tabular-nums; }
  .top-banner {
    padding: 14px 16px; border-radius: 10px; margin-bottom: 20px; font-weight: 600;
  }
  .top-banner.risk-high { background: #3a1f1f; color: #ff8a80; }
  .top-banner.risk-medium { background: #3a2f1a; color: #ffcc80; }
  .top-banner.risk-low { background: #1a3324; color: #7de3a6; }
  .disclaimer { color: var(--muted); font-size: 0.78rem; margin-top: 20px; line-height: 1.4; }
  #status { text-align: center; color: var(--muted); margin-top: 12px; font-size: 0.9rem; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Skin Lesion Classifier</h1>
  <p class="subtitle">Upload a dermoscopic image to classify it into one of 7 lesion types (rexnet_150).</p>

  {% if model_error %}
  <div class="warning">⚠️ {{ model_error }}</div>
  {% endif %}

  <div class="card">
    <div id="dropzone">
      <img id="preview" alt="preview">
      <div id="dropzoneText">
        <strong>Click to choose an image</strong>
        <p>or drag and drop &middot; JPG / PNG</p>
      </div>
    </div>
    <input type="file" id="fileInput" accept=".jpg,.jpeg,.png">
    <button id="submitBtn" disabled>Classify Image</button>
    <div id="status"></div>

    <div id="results">
      <div id="topBanner" class="top-banner"></div>
      <div id="resultList"></div>
      <div class="disclaimer">
        This tool is for educational/demo purposes only and is not a medical device.
        It does not provide a diagnosis. Always consult a qualified dermatologist
        for evaluation of skin lesions.
      </div>
    </div>
  </div>
</div>

<script>
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const dropzoneText = document.getElementById('dropzoneText');
const submitBtn = document.getElementById('submitBtn');
const statusEl = document.getElementById('status');
const results = document.getElementById('results');
const topBanner = document.getElementById('topBanner');
const resultList = document.getElementById('resultList');

let selectedFile = null;

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.style.display = 'block';
    dropzoneText.style.display = 'none';
  };
  reader.readAsDataURL(file);
  submitBtn.disabled = false;
  results.style.display = 'none';
  statusEl.textContent = '';
}

submitBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  submitBtn.disabled = true;
  statusEl.textContent = 'Analyzing image...';
  results.style.display = 'none';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      statusEl.textContent = 'Error: ' + (data.error || 'Something went wrong.');
      submitBtn.disabled = false;
      return;
    }

    statusEl.textContent = '';
    renderResults(data);
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
  }
  submitBtn.disabled = false;
});

function renderResults(data) {
  const top = data.top_prediction;
  topBanner.className = 'top-banner risk-' + top.risk;
  topBanner.textContent = `Most likely: ${top.name} (${top.code}) — ${top.probability}%`;

  resultList.innerHTML = '';
  data.predictions.forEach(p => {
    const row = document.createElement('div');
    row.className = 'result-row risk-' + p.risk;
    row.innerHTML = `
      <div class="result-label">${p.name}<div class="result-code">${p.code}</div></div>
      <div class="bar-track"><div class="bar-fill" style="width:${p.probability}%"></div></div>
      <div class="result-pct">${p.probability}%</div>
    `;
    resultList.appendChild(row);
  });

  results.style.display = 'block';
}
</script>
</body>
</html>
"""

# ----------------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024

# ----------------------------------------------------------------------
# Load model once at startup
# ----------------------------------------------------------------------
model = None
model_load_error = None

transform = T.Compose([
    T.Resize((IM_SIZE, IM_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


def load_model():
    global model, model_load_error
    try:
        m = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            m.load_state_dict(state_dict)
        else:
            model_load_error = (
                f"Weights file not found at '{MODEL_PATH}'. "
                f"Copy 'cancer_best_model.pth' from the notebook's saved_models/ folder "
                f"into this app's saved_models/ folder."
            )
        m.to(DEVICE)
        m.eval()
        model = m
    except Exception as e:
        model_load_error = f"Failed to load model: {e}"


load_model()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image: Image.Image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu()

    results = []
    for idx, code in enumerate(CLASSES):
        info = CLASS_INFO.get(code, {"name": code, "risk": "unknown"})
        results.append({
            "code": code,
            "name": info["name"],
            "risk": info["risk"],
            "probability": round(float(probs[idx]) * 100, 2),
        })

    results.sort(key=lambda r: r["probability"], reverse=True)
    return results


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(PAGE_TEMPLATE, model_error=model_load_error)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if model is not None and model_load_error is None else "error",
        "device": DEVICE,
        "model_name": MODEL_NAME,
        "weights_loaded": model_load_error is None,
        "error": model_load_error,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or model_load_error is not None:
        return jsonify({"error": model_load_error or "Model not loaded"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file part in request. Use form field name 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        results = predict_image(image)
    except Exception as e:
        return jsonify({"error": f"Could not process image: {e}"}), 400

    return jsonify({
        "predictions": results,
        "top_prediction": results[0],
    })


if __name__ == "__main__":
    # debug=True is convenient for local development; turn off in production
    app.run(host="0.0.0.0", port=5000, debug=True)
