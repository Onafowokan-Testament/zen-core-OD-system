# Plant Disease Detection API 🌱🔍

This FastAPI-based application provides a RESTful API for detecting plant diseases from images. It uses a pre-trained deep learning model to classify images of plant leaves into one of 15 disease categories. The API returns the predicted disease, accuracy, and placeholders for chemical recommendations and disease descriptions.

---

## Features

- **Image Classification**: Classifies plant leaf images into 15 disease categories.
- **FastAPI Backend**: Provides a fast and scalable RESTful API.
- **Pre-trained Model**: Uses a pre-trained convolutional neural network (CNN) for accurate predictions.
- **Placeholder Data**: Includes placeholders for chemical recommendations and disease descriptions (to be implemented).

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/plant-disease-detection.git
   cd plant-disease-detection
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI application**:
   ```bash
   uvicorn app:app --reload
   ```

5. **Access the API**:
   Open your browser and navigate to `http://127.0.0.1:8000/docs` to access the Swagger UI for API documentation.

---

## API Endpoints

### 1. **Root Endpoint**
   - **Endpoint**: `GET /`
   - **Description**: Returns a welcome message.
   - **Response**:
     ```json
     {
       "topic": "Zen core"
     }
     ```

### 2. **Predict Disease**
   - **Endpoint**: `POST /predict-disease/`
   - **Description**: Upload an image of a plant leaf to predict the disease.
   - **Request Body**:
     - `file`: Image file (JPEG, PNG, etc.).
   - **Response**:
     ```json
     {
       "predicted_disease": "Tomato__Tomato_YellowLeaf__Curl_Virus",
       "accuracy": 98.26,
       "chemical_to_use": "Chemical details to be added later",
       "description": "Description of the disease to be added later"
     }
     ```

---

## File Structure

```
plant-disease-detection/
├── app.py                # Main FastAPI application
├── trained_model.keras   # Pre-trained deep learning model
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation
```

---

## Dependencies

- Python 3.8+
- `tensorflow` for deep learning model inference
- `fastapi` for building the RESTful API
- `uvicorn` for running the FastAPI server
- `numpy` for numerical operations
- `opencv-python` for image processing
- `matplotlib` for visualization (not actively used in the API)

---

## How It Works

1. **Image Upload**:
   - The user uploads an image of a plant leaf to the `/predict-disease/` endpoint.

2. **Image Preprocessing**:
   - The image is resized to 128x128 pixels and converted to a NumPy array.

3. **Model Inference**:
   - The pre-trained model predicts the disease category and returns the result.

4. **Response**:
   - The API returns the predicted disease, accuracy, and placeholders for chemical recommendations and disease descriptions.

---

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Detect plant diseases with ease using this API! 🌿🔬
