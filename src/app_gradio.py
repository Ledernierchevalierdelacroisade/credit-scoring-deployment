from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "lightgbm_final.joblib"
FEATURES_PATH = MODEL_DIR / "model_features.joblib"

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

THRESHOLD = 0.10


def predict_default(amt_credit, amt_income_total, amt_annuity, days_birth, days_employed):
    data = {
        "AMT_CREDIT": amt_credit,
        "AMT_INCOME_TOTAL": amt_income_total,
        "AMT_ANNUITY": amt_annuity,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
    }

    df = pd.DataFrame([data])
    df = df.reindex(columns=features, fill_value=0)

    probability = float(model.predict_proba(df)[:, 1][0])
    prediction = int(probability >= THRESHOLD)

    decision = "Client à risque - crédit refusé" if prediction == 1 else "Client peu risqué - crédit accepté"

    return round(probability, 4), prediction, decision


demo = gr.Interface(
    fn=predict_default,
    inputs=[
        gr.Number(label="Montant du crédit"),
        gr.Number(label="Revenu total"),
        gr.Number(label="Annuité"),
        gr.Number(label="Âge en jours négatifs"),
        gr.Number(label="Ancienneté emploi en jours négatifs"),
    ],
    outputs=[
        gr.Number(label="Probabilité de défaut"),
        gr.Number(label="Prédiction"),
        gr.Textbox(label="Décision"),
    ],
    title="Credit Scoring - Démo Gradio",
    description="Interface simple permettant de tester le modèle de scoring crédit."
)

if __name__ == "__main__":
    demo.launch()