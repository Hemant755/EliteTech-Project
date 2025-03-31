import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
import traceback

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
host = os.getenv('HOST', 'localhost')  # Changed to 'localhost' for browser accessibility
port = int(os.getenv('PORT', 8000))

# Step 1: Data Collection & Preprocessing
def load_data():
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.feature_names

df, feature_names = load_data()
X = df[feature_names]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logger.info(f'Model Accuracy: {accuracy:.2f}')

# Step 3: Save Model
joblib.dump(model, 'iris_model.pkl')

# Step 4: API Deployment using FastAPI
app = FastAPI()

# Add CORS Middleware for external API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed to restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = joblib.load('iris_model.pkl')
model_version = "1.0"

class Features(BaseModel):
    features: list

@app.get('/')
def home():
    """
    Root endpoint providing API information.
    """
    return {'message': 'Iris Classifier API', 'model_version': model_version}

@app.get('/health/')
def health_check():
    """
    Health endpoint to check API status.
    """
    return {'status': 'OK', 'model_loaded': model is not None}

@app.post('/predict/')
def predict(input_data: Features):
    """
    Predict the class of an Iris flower based on input features.

    Args:
    - features: A list of numeric values representing sepal/petal length and width.

    Returns:
    - prediction: The predicted class (0, 1, or 2).
    """
    try:
        if len(input_data.features) != len(feature_names):
            raise ValueError("Feature list size mismatch!")
        if not all(isinstance(x, (int, float)) for x in input_data.features):
            raise ValueError("All features must be numeric!")
        prediction = model.predict([input_data.features])
        return {'prediction': int(prediction[0])}
    except Exception as e:
        logger.error(f"Prediction error: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))

# Run the API server
if __name__ == '__main__':
    uvicorn.run(app, host=host, port=port)

import requests

url = "http://localhost:8000/predict/"
data = {"features": [5.1, 3.5, 1.4, 0.2]}
response = requests.post(url, json=data)
print(response.json())