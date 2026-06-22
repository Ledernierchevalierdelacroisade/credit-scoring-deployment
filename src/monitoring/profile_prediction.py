import cProfile
import pstats
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

model = joblib.load(MODEL_DIR / "lightgbm_final.joblib")
features = joblib.load(MODEL_DIR / "model_features.joblib")

payload = {
    "AMT_CREDIT": 200000,
    "AMT_INCOME_TOTAL": 150000,
    "AMT_ANNUITY": 25000,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -2500,
}


def predict_once():
    df = pd.DataFrame([payload])
    df = df.reindex(columns=features, fill_value=0)
    probability = float(model.predict_proba(df)[:, 1][0])
    prediction = int(probability >= 0.10)
    return probability, prediction


profile_path = REPORT_DIR / "profile_prediction.prof"
txt_path = REPORT_DIR / "profile_prediction.txt"

profiler = cProfile.Profile()
profiler.enable()

for _ in range(100):
    predict_once()

profiler.disable()
profiler.dump_stats(profile_path)

with open(txt_path, "w", encoding="utf-8") as f:
    stats = pstats.Stats(profiler, stream=f)
    stats.sort_stats("cumtime")
    stats.print_stats(20)

print(f"Profiling sauvegardé dans {profile_path}")
print(f"Rapport texte sauvegardé dans {txt_path}")