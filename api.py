import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "fruit_vegetable_custom_cnn_final.keras"))
CLASS_NAMES_PATH = Path(os.getenv("CLASS_NAMES_PATH", BASE_DIR / "fruit_vegetable_class_names.json"))
IMG_SIZE = (224, 224)

app = FastAPI(title="Fruit and Vegetable Classifier API", version="1.0.0")

model = None
class_names = []

@app.on_event("startup")
def load_artifacts():
    global model, class_names
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}. Copy your trained .keras file beside main.py."
        )
    if not CLASS_NAMES_PATH.exists():
        raise RuntimeError(
            f"Class names file not found: {CLASS_NAMES_PATH}. Copy fruit_vegetable_class_names.json beside main.py."
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    if not isinstance(class_names, list) or not class_names:
        raise RuntimeError("Class names JSON must contain a non-empty list.")
    if model.output_shape[-1] != len(class_names):
        raise RuntimeError(
            f"Model outputs {model.output_shape[-1]} classes, but JSON has {len(class_names)} names."
        )

@app.get("/")
def root():
    return {"message": "Fruit and Vegetable Classifier API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "model_not_loaded"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        image = Image.open(BytesIO(raw)).convert("RGB")
        image = image.resize(IMG_SIZE)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read this image. Try JPG or PNG.")
    except OSError:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Match the notebook: RGB, resize to 224x224, float32 0-255.
    # The model's Rescaling layer converts pixels to 0-1.
    batch = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:min(5, len(class_names))]
    return {
        "filename": file.filename,
        "predicted_class": class_names[int(top_indices[0])],
        "confidence": round(float(probabilities[top_indices[0]]) * 100, 2),
        "top_predictions": [
            {"class": class_names[int(i)], "confidence": round(float(probabilities[i]) * 100, 2)}
            for i in top_indices
        ],
    }
