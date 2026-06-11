from fastapi import FastAPI
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Credit Scoring API")

# chemins
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"

# chargement modèle
model = joblib.load(MODEL_DIR / "lightgbm_final.joblib")
features = joblib.load(MODEL_DIR / "model_features.joblib")

THRESHOLD = 0.10

@app.get("/")
def root():
    return {"message": "API Credit Scoring OK"}

@app.post("/predict")
def predict(data: dict):
    
    # transformer en DataFrame
    df = pd.DataFrame([data])
    
    # garantir ordre des features
    df = df.reindex(columns=features)
    
    # prédiction
    proba = model.predict_proba(df)[:, 1][0]
    prediction = int(proba >= THRESHOLD)
    
    return {
        "probability": float(proba),
        "prediction": prediction,
        "threshold": THRESHOLD
    }