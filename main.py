from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model once at startup
model = tf.keras.models.load_model("models/m1-2.keras")
IMG_SIZE = (224, 224)   # confirm this matches training

from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.get("/", response_class=HTMLResponse)
def home():
    with open("template/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    processed = preprocess_image(image)

    pred = model.predict(processed)[0][0]
    label = "NORMAL" if pred >= 0.5 else "PNEUMONIA"
    confidence = float(pred) if pred >= 0.5 else float(1 - pred)

    return {"prediction": label, "confidence": round(confidence, 4)}