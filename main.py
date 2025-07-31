from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

# Load the model
model = load_model(r"C:\Users\Akhil M\OneDrive\Desktop\brainTumor\brain_tumor_classifier_vgg16.h5")

# Constants
IMAGE_SIZE = (224, 224)

# Initialize FastAPI app
app = FastAPI(title="Brain Tumor Classifier API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files directory
static_dir = os.path.join(os.path.dirname(__file__), "static")

# Mount static files under /static (NOT root "/")
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

# Optional: Serve index.html at root "/"
@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

# Prediction helper
def predict(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img_array = image.img_to_array(img).astype('float32') / 255.0
        if img_array.shape != (224, 224, 3):
            raise ValueError(f"Invalid image shape: {img_array.shape}")
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0][0]
        result = "Tumor" if pred > 0.5 else "No Tumor"
        confidence = float(pred) if result == "Tumor" else 1 - float(pred)
        return result, confidence

    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")

# POST endpoint
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        return JSONResponse(status_code=400, content={"error": "Only PNG, JPG, JPEG allowed."})

    try:
        bytes_data = await file.read()
        if not bytes_data:
            return JSONResponse(status_code=400, content={"error": "Empty file uploaded."})

        try:
            Image.open(io.BytesIO(bytes_data)).verify()
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Invalid image file: {str(e)}"})

        result, confidence = predict(bytes_data)
        return {
            "prediction": result,
            "confidence": f"{confidence:.2f}"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Server error: {str(e)}"})
