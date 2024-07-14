import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#MODEL_PATH = os.path.abspath(""C:/Users/LENOVO/Desktop/leaf disease/models/1"")
MODEL = tf.keras.models.load_model("D:/leaf disease/models/1.keras")
#beta_model = tf.keras.models.load_model("C:/Users/LENOVO/Desktop/leaf disease/models/2")

CLASS_NAMES = ["Bacterial_spot", "healthy", "Early_blight", "Late_blight", "Powdery", "Rust"]

@app.get("/ping")
async def ping():
    return "hello, I am Salil"

def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
        image = np.array(image)
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding image: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Check if the uploaded file is an image
        print("in predict")
        if "image" not in file.content_type:
            raise HTTPException(status_code=400, detail="Uploaded file is not an image")

        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)

        prediction = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))

        return {
            'class': predicted_class,
            'confidence': confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
