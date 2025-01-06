import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
from fastapi.responses import JSONResponse
import cv2
import matplotlib.pyplot as plt

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model('trained_model.keras')

# Define class names (labels)
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 
    'Potato___healthy', 'Tomato_Bacterial_spot', 
    'Tomato_Early_blight', 'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# Placeholder for chemical to use and description (to be filled)
chemicals = "Chemical details to be added later"
description = "Description of the disease to be added later"

@app.get("/")
def read_root():
    return {"topic": "Zen core "}

@app.post("/predict-disease/")
async def predict_disease(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        img = await file.read()
        img = np.array(bytearray(img), dtype=np.uint8)

        # Decode image and convert to correct color space
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocess image
        image = cv2.resize(img, (128, 128))  # resize to model input shape
        input_arr = img_to_array(image)  # convert to array
        input_arr = np.array([input_arr])  # batch dimension

        # Make predictions
        predictions = model.predict(input_arr)
        result_index = np.argmax(predictions)

        # Get predicted class name
        model_prediction = class_names[result_index]
        
        # Return prediction along with placeholders
        return JSONResponse({
            "predicted_disease": model_prediction,
            "accuracy": float(np.max(predictions) * 100),  # accuracy as percentage
            "chemical_to_use": chemicals,
            "description": description
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# Run the server with:
# uvicorn <filename>:app --reload
