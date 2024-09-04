from io import BytesIO
import numpy as np
import tensorflow as tf
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from keras.saving import load_model  # Adjusted the import path for compatibility
from fastapi.middleware.cors import CORSMiddleware

# Correctly initialize the FastAPI app instance
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

# Adding middleware for CORS handling
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL = load_model("C:/hackathon/KrishiSevak-HexaRise/saved_models/1.keras")

# Class names for the model
CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy"
]


# Test endpoint to check if the server is running
@app.get("/ping")
async def ping():
    return {"message": "Hello World!"}


# Function to read and preprocess the image file
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    # Return the prediction result
    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }


# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
