
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI(title="NeuralRetail ML API", version="1.0")

# ---------------- LOAD MODEL ----------------
model_path = "churn_model.pkl"
model = joblib.load(model_path)

# ---------------- REQUEST SCHEMA ----------------
class ChurnRequest(BaseModel):
    recency: float
    frequency: float
    monetary: float

# ---------------- HEALTH CHECK ----------------
@app.get("/health")
def health():
    return {"status": "API running"}

@app.get("/")
def home():
    return {
        "message": "NeuralRetail API is running",
        "status": "active",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict/churn"
    }

# ---------------- PREDICT CHURN ----------------
@app.post("/predict/churn")
def predict_churn(data: ChurnRequest):

    input_data = [[
        data.recency,
        data.frequency,
        data.monetary
    ]]

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    return {
    "churn_prediction": int(prediction),
    "probability": float(proba),
    "label": "High Risk" if prediction == 1 else "Low Risk"
}

