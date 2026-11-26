
import os
import shutil
import uuid

from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from torchvision import transforms
from torchvision.io import decode_image

from models.environ_model import EnvironPredictModel


app = FastAPI(title="Environment Prediction API")


# Project directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "saved_models" / "environ_model.pth"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}


# Load trained model checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model = EnvironPredictModel(
    **checkpoint["model_config"]
)

model.load_state_dict(checkpoint["model_state_dict"])
classes = checkpoint["class_names"]
model.eval()


@torch.inference_mode()
def predict_image(img_path: str):

    image = decode_image(str(img_path))

    transform_ext = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ConvertImageDtype(torch.float32)
    ])

    image_tensor = transform_ext(image)
    image_tensor = image_tensor.unsqueeze(0)
    logits = model(image_tensor)

    predicted_index = logits.argmax(dim=1).item()

    probabilities = torch.softmax(logits, dim=1)

    confidence = probabilities[0, predicted_index].item()

    return {
        "prediction": classes[predicted_index],
        "confidence": round(confidence * 100, 2)
    }

async def save_uploaded_image(file: UploadFile):

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Use JPEG, PNG or WEBP."
        )

    ext = Path(file.filename or "").suffix.lower()

    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image extension."
        )

    unique_filename = f"{uuid.uuid4()}{ext}"

    file_path = UPLOAD_DIR / unique_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        decode_image(str(file_path))

    except Exception:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image."
        )

    finally:
        await file.close()

    return unique_filename, file_path


@app.get("/")
def index():
    return {
        "message": "Welcome to Environment Prediction API",
        "status": "running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    filename, file_path = await save_uploaded_image(file)

    result = predict_image(str(file_path))

    return {
        "message": "Prediction successful",
        "filename": filename,
        "result": result
    }
