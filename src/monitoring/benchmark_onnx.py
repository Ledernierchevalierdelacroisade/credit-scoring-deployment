import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

features = joblib.load(MODEL_DIR / "model_features.joblib")

payload = {
    "AMT_CREDIT": 200000,
    "AMT_INCOME_TOTAL": 150000,
    "AMT_ANNUITY": 25000,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -2500,
}

df = pd.DataFrame([payload])
df = df.reindex(columns=features, fill_value=0)

X = df.astype(np.float32).values

session = ort.InferenceSession(
    str(MODEL_DIR / "lightgbm_final.onnx"),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

times = []

for _ in range(100):
    start = time.perf_counter()

    session.run(None, {input_name: X})

    end = time.perf_counter()

    times.append((end - start) * 1000)

print(f"Moyenne ONNX : {np.mean(times):.4f} ms")
print(f"Min ONNX : {np.min(times):.4f} ms")
print(f"Max ONNX : {np.max(times):.4f} ms")