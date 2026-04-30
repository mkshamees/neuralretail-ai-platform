import os
import joblib
from fastapi import FastAPI

app = FastAPI()

model_path = "churn_model.pkl"

# Safe loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model = joblib.load(model_path)


@app.get("/")
def home():
    return {"status": "API running"}


@app.get("/health")
def health():
    return {"status": "API running"}


@app.post("/predict/churn")
def predict(data: dict):
    recency = data.get("recency", 0)
    frequency = data.get("frequency", 0)
    monetary = data.get("monetary", 0)

    pred = model.predict([[recency, frequency, monetary]])[0]

    return {"prediction": int(pred)}
