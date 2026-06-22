from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
LOG_FILE = BASE_DIR / "logs" / "predictions.jsonl"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

logs = []

with open(LOG_FILE, "r", encoding="utf-8") as f:
    for line in f:
        logs.append(json.loads(line))

df_raw = pd.DataFrame(logs)

# Compatible ancien format + nouveau format structuré
if "output" in df_raw.columns:
    df = pd.DataFrame({
        "timestamp": df_raw["timestamp"],
        "source": df_raw.get("source", "api"),
        "model_name": df_raw.get("model_name", "unknown"),
        "model_version": df_raw.get("model_version", "unknown"),
        "status": df_raw.get("status", "unknown"),
        "latency_ms": df_raw.get("latency_ms", None),
        "probability": df_raw["output"].apply(lambda x: x["probability"]),
        "prediction": df_raw["output"].apply(lambda x: x["prediction"]),
        "threshold": df_raw["output"].apply(lambda x: x["threshold"]),
    })
else:
    df = df_raw.copy()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["probability"] = df["probability"].astype(float)
df["prediction"] = df["prediction"].astype(int)

print("Nombre de prédictions :", len(df))
print(df["probability"].describe())

refusal_rate = df["prediction"].mean()
print(f"Taux de refus : {refusal_rate:.2%}")

if "latency_ms" in df.columns and df["latency_ms"].notna().any():
    print("Latence moyenne :", round(df["latency_ms"].mean(), 2), "ms")

# Distribution des probabilités
plt.figure(figsize=(8, 5))
plt.hist(df["probability"], bins=10)
plt.title("Distribution des probabilités de défaut")
plt.xlabel("Probabilité")
plt.ylabel("Nombre de prédictions")
plt.tight_layout()
plt.savefig(REPORT_DIR / "probability_distribution.png")
plt.close()

# Evolution temporelle
plt.figure(figsize=(8, 5))
df_sorted = df.sort_values("timestamp")
plt.plot(df_sorted["timestamp"], df_sorted["probability"], marker="o")
plt.title("Evolution des scores dans le temps")
plt.xlabel("Timestamp")
plt.ylabel("Probabilité")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(REPORT_DIR / "score_over_time.png")
plt.close()

# Détection simple d'anomalies
if refusal_rate > 0.8:
    print("\nALERTE : taux de refus très élevé")

if "latency_ms" in df.columns and df["latency_ms"].notna().any():
    if df["latency_ms"].mean() > 1000:
        print("\nALERTE : latence moyenne élevée")

print("Rapports générés dans reports/")