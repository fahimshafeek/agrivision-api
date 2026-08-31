# ============================================================
# FastAPI Vision AI Microservice — Track A: AgriVision
# ============================================================
# CONSTRAINT #2: Uncertainty Rule
#   If top confidence < 75%, return {"prediction": "Uncertain"}
# ============================================================

import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="AgriVision — Vision AI Microservice")

# ---- Configuration ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "vision_model.pth"
CONFIDENCE_THRESHOLD = 0.75  # Uncertainty Rule threshold


def load_model():
    """Load the trained EfficientNet-B0 model from checkpoint."""
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    classes = checkpoint['classes']
    num_classes = checkpoint['num_classes']
    img_size = checkpoint['img_size']
    imagenet_mean = checkpoint['imagenet_mean']
    imagenet_std = checkpoint['imagenet_std']
    cfg = checkpoint['classifier_config']

    # Rebuild model architecture
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
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        confidence_val = confidence.item()
        predicted_class = CLASSES[predicted_idx.item()]
        probs_dict = {cls: round(float(p), 4) for cls, p in zip(CLASSES, probabilities[0].cpu().tolist())}

        # *** UNCERTAINTY RULE (Constraint #2) ***
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
