# ============================================================
# FastAPI Vision AI Microservice — Track A: AgriVision
# ============================================================

import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse

app = FastAPI(title="AgriVision — Vision AI Microservice")

# ---- Configuration ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "vision_model.pth"
CONFIDENCE_THRESHOLD = 0.75  # Uncertainty Rule threshold

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgriVision AI Classifier</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .container { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.1); max-width: 500px; width: 100%; text-align: center; }
        h1 { color: #2c3e50; margin-bottom: 5px; }
        p.subtitle { color: #7f8c8d; font-size: 0.9em; margin-bottom: 25px; }
        .upload-area { border: 2px dashed #3498db; border-radius: 8px; padding: 30px; cursor: pointer; transition: background 0.3s; margin-bottom: 20px; }
        .upload-area:hover { background: #ebf5fb; }
        .upload-area p { color: #3498db; font-weight: bold; margin: 0; }
        input[type="file"] { display: none; }
        #preview { max-width: 100%; max-height: 300px; border-radius: 8px; display: none; margin: 15px auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        button { background: #2ecc71; color: white; border: none; padding: 12px 25px; font-size: 16px; border-radius: 25px; cursor: pointer; font-weight: bold; transition: background 0.3s; width: 100%; }
        button:hover { background: #27ae60; }
        button:disabled { background: #95a5a6; cursor: not-allowed; }
        #result { margin-top: 25px; padding: 15px; border-radius: 8px; display: none; }
        .success { background: #d4efdf; color: #1e8449; border: 1px solid #2ecc71; }
        .uncertain { background: #fdebd0; color: #d35400; border: 1px solid #f39c12; }
        .error { background: #fadbd8; color: #c0392b; border: 1px solid #e74c3c; }
        .prob-bar-container { display: flex; align-items: center; margin-top: 8px; font-size: 0.85em; }
        .prob-label { width: 60px; text-align: left; font-weight: bold; color: #34495e; text-transform: capitalize; }
        .prob-bar-bg { flex-grow: 1; background: #ecf0f1; height: 10px; border-radius: 5px; margin: 0 10px; overflow: hidden; }
        .prob-bar-fill { height: 100%; background: #3498db; transition: width 0.5s ease-out; }
        .prob-val { width: 40px; text-align: right; color: #7f8c8d; }
    </style>
</head>
<body>
<div class="container">
    <h1>🌱 AgriVision AI</h1>
    <p class="subtitle">Upload a plant leaf image to detect Blight, Spot, or Healthy status.</p>
    <div class="upload-area" id="drop-zone" onclick="document.getElementById('file-input').click()">
        <p>Drag & Drop or Click to Upload</p>
        <input type="file" id="file-input" accept="image/*" onchange="previewImage(event)">
    </div>
    <img id="preview" alt="Image Preview">
    <button id="predict-btn" onclick="uploadAndPredict()" disabled>Analyze Leaf</button>
    <div id="result">
        <h2 id="pred-text" style="margin-top:0; margin-bottom: 5px;"></h2>
        <p id="conf-text" style="margin: 0; font-size: 0.9em; margin-bottom: 15px;"></p>
        <div id="prob-container"></div>
    </div>
</div>
<script>
    let currentFile = null;
    function previewImage(event) {
        const file = event.target.files[0];
        if (file) {
            currentFile = file;
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.getElementById('preview');
                img.src = e.target.result;
                img.style.display = 'block';
                document.getElementById('predict-btn').disabled = false;
                document.getElementById('result').style.display = 'none';
            }
            reader.readAsDataURL(file);
        }
    }
    const dropZone = document.getElementById('drop-zone');
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.background = '#ebf5fb'; });
    dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.style.background = 'transparent'; });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.background = 'transparent';
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            document.getElementById('file-input').files = e.dataTransfer.files;
            previewImage({ target: { files: e.dataTransfer.files } });
        }
    });
    async function uploadAndPredict() {
        if (!currentFile) return;
        const btn = document.getElementById('predict-btn');
        const resDiv = document.getElementById('result');
        const predText = document.getElementById('pred-text');
        const confText = document.getElementById('conf-text');
        const probContainer = document.getElementById('prob-container');
        btn.innerText = 'Analyzing...';
        btn.disabled = true;
        resDiv.style.display = 'none';
        probContainer.innerHTML = '';
        const formData = new FormData();
        formData.append('file', currentFile);
        try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();
            resDiv.style.display = 'block';
            resDiv.className = '';
            if (data.error) {
                resDiv.classList.add('error');
                predText.innerText = 'Error';
                confText.innerText = data.error;
            } else {
                predText.innerText = data.prediction.toUpperCase();
                confText.innerText = `Confidence: ${data.confidence}%`;
                if (data.prediction === 'Uncertain') {
                    resDiv.classList.add('uncertain');
                    if (data.note) confText.innerText += ` (${data.note})`;
                } else {
                    resDiv.classList.add('success');
                }
                for (const [cls, prob] of Object.entries(data.probabilities)) {
                    const percent = (prob * 100).toFixed(1);
                    const isMax = cls === data.prediction.toLowerCase() || (data.prediction === 'Uncertain' && parseFloat(percent) === data.confidence);
                    const color = isMax ? (data.prediction === 'Uncertain' ? '#f39c12' : '#2ecc71') : '#95a5a6';
                    probContainer.innerHTML += `
                        <div class="prob-bar-container">
                            <div class="prob-label">${cls}</div>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill" style="width: ${percent}%; background: ${color}"></div>
                            </div>
                            <div class="prob-val">${percent}%</div>
                        </div>
                    `;
                }
            }
        } catch (err) {
            resDiv.style.display = 'block';
            resDiv.classList.add('error');
            predText.innerText = 'Network Error';
            confText.innerText = err.message;
        } finally {
            btn.innerText = 'Analyze Leaf';
            btn.disabled = false;
        }
    }
</script>
</body>
</html>
"""

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    classes = checkpoint['classes']
    num_classes = checkpoint['num_classes']
    img_size = checkpoint['img_size']
    imagenet_mean = checkpoint['imagenet_mean']
    imagenet_std = checkpoint['imagenet_std']
    cfg = checkpoint['classifier_config']

    model = models.efficientnet_b0(weights=None)
    in_features = cfg['in_features']
    model.classifier = nn.Sequential(
        nn.Dropout(p=cfg['dropout'], inplace=True),
        nn.Linear(in_features, cfg['hidden_dim']),
        nn.SiLU(inplace=True),
        nn.Dropout(p=cfg['dropout'] * 0.5),
        nn.Linear(cfg['hidden_dim'], num_classes),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    return model, classes, preprocess

# Load model at startup
model, CLASSES, preprocess = load_model()

@app.get('/', response_class=HTMLResponse)
async def serve_frontend():
    return HTML_CONTENT

@app.get('/health')
async def health():
    return {
        'status': 'healthy',
        'model': 'EfficientNet-B0',
        'track': 'C - AgriVision',
        'classes': list(CLASSES),
    }

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        confidence_val = confidence.item()
        predicted_class = CLASSES[predicted_idx.item()]
        probs_dict = {cls: round(float(p), 4) for cls, p in zip(CLASSES, probabilities[0].cpu().tolist())}

        if confidence_val < CONFIDENCE_THRESHOLD:
            return JSONResponse(content={
                'prediction': 'Uncertain',
                'confidence': round(confidence_val * 100, 2),
                'note': 'Confidence below 75% threshold',
                'probabilities': probs_dict,
            })

        return JSONResponse(content={
            'prediction': predicted_class,
            'confidence': round(confidence_val * 100, 2),
            'probabilities': probs_dict,
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})

