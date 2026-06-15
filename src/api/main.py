from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

import os
from fastapi import Header

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "lightgbm_final.joblib"
FEATURES_PATH = MODEL_DIR / "model_features.joblib"

app = FastAPI(
    title="Credit Scoring API",
    version="1.0.0",
    description="API de prédiction du risque de défaut client."
)

# Le modèle est chargé une seule fois au démarrage de l’API.
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

THRESHOLD = 0.10

API_KEY = os.getenv("API_KEY", "dev-secret-key")


@app.get("/")
def root():
    return {"message": "Credit Scoring API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: Dict[str, Any], x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
        
    try:
        if not data:
            raise HTTPException(status_code=400, detail="Aucune donnée fournie.")

        df = pd.DataFrame([data])

        numeric_columns = [
            "AMT_CREDIT",
            "AMT_INCOME_TOTAL",
            "AMT_ANNUITY",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED"
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if df[col].isnull().any():
                    raise HTTPException(
                        status_code=422,
                        detail=f"La variable {col} doit être numérique."
                    )

        if "AMT_INCOME_TOTAL" in df.columns and df["AMT_INCOME_TOTAL"].iloc[0] <= 0:
            raise HTTPException(
                status_code=422,
                detail="Le revenu total doit être supérieur à 0."
            )

        if "AMT_CREDIT" in df.columns and df["AMT_CREDIT"].iloc[0] <= 0:
            raise HTTPException(
            status_code=422,
            detail="Le montant du crédit doit être supérieur à 0."
        )

        if "DAYS_BIRTH" in df.columns and df["DAYS_BIRTH"].iloc[0] >= 0:
            raise HTTPException(
            status_code=422,
            detail="DAYS_BIRTH doit être négatif."
        )

        df = df.reindex(columns=features, fill_value=0)

        probability = float(model.predict_proba(df)[:, 1][0])
        prediction = int(probability >= THRESHOLD)

        log_entry = {
            "timestamp": str(datetime.now()),
            "input": data,
            "probability": probability,
            "prediction": prediction,
            "threshold": THRESHOLD
        }

        with open(LOG_DIR / "predictions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        return {
            "probability": probability,
            "prediction": prediction,
            "threshold": THRESHOLD
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")
        