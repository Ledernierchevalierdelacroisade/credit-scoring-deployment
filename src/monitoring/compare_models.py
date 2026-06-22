from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "lightgbm_final.joblib"
FEATURES_PATH = MODEL_DIR / "model_features.joblib"
ONNX_PATH = MODEL_DIR / "lightgbm_final.onnx"

THRESHOLD = 0.10

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

payload = {
    "AMT_CREDIT": 200000,
    "AMT_INCOME_TOTAL": 150000,
    "AMT_ANNUITY": 25000,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -2500,
}

df = pd.DataFrame([payload])
df = df.reindex(columns=features, fill_value=0)

# Prediction LightGBM
proba_lgbm = float(model.predict_proba(df)[:, 1][0])
pred_lgbm = int(proba_lgbm >= THRESHOLD)

# Prediction ONNX
X = df.astype(np.float32).values

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: X})

print("Sorties ONNX brutes :")
for i, output in enumerate(outputs):
    print(f"Output {i}: {output}")

# Selon la conversion LightGBM, les probabilités peuvent être dans la 2e sortie.
onnx_proba = outputs[1]

if isinstance(onnx_proba, list):
    proba_onnx = float(onnx_proba[0][1])
elif isinstance(onnx_proba, np.ndarray):
    proba_onnx = float(onnx_proba[0][1])
else:
    raise ValueError("Format de sortie ONNX non reconnu.")

pred_onnx = int(proba_onnx >= THRESHOLD)

print("\n===== Comparaison LightGBM vs ONNX =====")
print(f"LightGBM probability : {proba_lgbm:.6f}")
print(f"ONNX probability     : {proba_onnx:.6f}")
print(f"Différence absolue   : {abs(proba_lgbm - proba_onnx):.8f}")

print(f"LightGBM prediction  : {pred_lgbm}")
print(f"ONNX prediction      : {pred_onnx}")

if pred_lgbm == pred_onnx:
    print("\nValidation OK : aucune régression sur la prédiction.")
else:
    print("\nAttention : prédiction différente entre LightGBM et ONNX.")