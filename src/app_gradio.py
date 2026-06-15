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


def predict_default(amt_credit, amt_income_total, amt_annuity, age_years, employed_years):
    # Validation des entrées côté Gradio
    if amt_credit is None or amt_income_total is None or amt_annuity is None:
        return None, None, "Erreur : les montants doivent être renseignés."

    if age_years is None or employed_years is None:
        return None, None, "Erreur : l'âge et l'ancienneté doivent être renseignés."

    if amt_credit <= 0:
        return None, None, "Erreur : le montant du crédit doit être supérieur à 0."

    if amt_income_total <= 0:
        return None, None, "Erreur : le revenu total doit être supérieur à 0."

    if amt_annuity <= 0:
        return None, None, "Erreur : l'annuité doit être supérieure à 0."

    if age_years <= 0 or age_years > 120:
        return None, None, "Erreur : l'âge doit être compris entre 1 et 120 ans."

    if employed_years < 0 or employed_years > 80:
        return None, None, "Erreur : l'ancienneté professionnelle doit être comprise entre 0 et 80 ans."

    # Conversion vers le format attendu par le modèle Home Credit
    days_birth = -int(age_years * 365)
    days_employed = -int(employed_years * 365)

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

    decision = (
        "Client à risque - crédit refusé"
        if prediction == 1
        else "Client peu risqué - crédit accepté"
    )

    return round(probability, 4), prediction, decision


demo = gr.Interface(
    fn=predict_default,
    inputs=[
        gr.Number(label="Montant du crédit", value=200000),
        gr.Number(label="Revenu total", value=150000),
        gr.Number(label="Annuité", value=25000),
        gr.Number(label="Âge du client en années", value=35),
        gr.Number(label="Ancienneté professionnelle en années", value=5),
    ],
    outputs=[
        gr.Number(label="Probabilité de défaut"),
        gr.Number(label="Prédiction"),
        gr.Textbox(label="Décision / message d'erreur"),
    ],
    title="Credit Scoring - Démo Gradio",
    description=(
        "Interface de démonstration permettant de tester le modèle de scoring crédit. "
        "Les entrées sont validées avant prédiction."
    ),
)

if __name__ == "__main__":
    demo.launch()