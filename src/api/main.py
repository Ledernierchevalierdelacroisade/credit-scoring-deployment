from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "lightgbm_final.joblib"
FEATURES_PATH = MODEL_DIR / "model_features.joblib"

app = FastAPI(title="Credit Scoring API", version="1.0.0")

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

THRESHOLD = 0.10


@app.get("/")
def root():
    return {"message": "Credit Scoring API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        df = df.reindex(columns=features, fill_value=0)

        probability = float(model.predict_proba(df)[:, 1][0])
        prediction = int(probability >= THRESHOLD)

        return {
            "probability": probability,
            "prediction": prediction,
            "threshold": THRESHOLD,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))