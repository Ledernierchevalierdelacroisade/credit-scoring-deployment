from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]

LOG_FILE = BASE_DIR / "logs" / "predictions.jsonl"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# Lecture des logs JSONL
logs = []

with open(LOG_FILE, "r", encoding="utf-8") as f:
    for line in f:
        logs.append(json.loads(line))

df = pd.DataFrame(logs)

print("Nombre de prédictions :", len(df))

# Extraire les probabilités
df["probability"] = df["probability"].astype(float)

# Statistiques
print(df["probability"].describe())

# Taux de refus
refusal_rate = df["prediction"].mean()

print(f"Taux de refus : {refusal_rate:.2%}")

# Histogramme des probabilités
plt.figure(figsize=(8, 5))

plt.hist(df["probability"], bins=10)

plt.title("Distribution des probabilités de défaut")
plt.xlabel("Probabilité")
plt.ylabel("Nombre de prédictions")

plt.savefig(REPORT_DIR / "probability_distribution.png")

print("Rapport généré dans reports/")

# Détection simple d'anomalies

high_risk = df[df["probability"] > 0.95]

if len(high_risk) > 0:
    print("\nALERTE : prédictions à très haut risque détectées")
    print(high_risk[["probability", "prediction"]])

if refusal_rate > 0.8:
    print("\nALERTE : taux de refus très élevé")