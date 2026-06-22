import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
LOG_FILE = BASE_DIR / "logs" / "predictions.jsonl"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# Données de référence simulées
reference_data = pd.DataFrame({
    "AMT_CREDIT": [500000, 300000, 700000, 450000, 600000],
    "AMT_INCOME_TOTAL": [200000, 150000, 250000, 180000, 220000],
})

# Données de production depuis les logs API
logs = []
with open(LOG_FILE, "r", encoding="utf-8") as f:
    for line in f:
        logs.append(json.loads(line))

prod_rows = []
for log in logs:
    prod_rows.append(log["input"])

current_data = pd.DataFrame(prod_rows)

common_columns = [col for col in reference_data.columns if col in current_data.columns]

for column in common_columns:
    plt.figure(figsize=(6, 4))
    plt.hist(reference_data[column], alpha=0.5, label="Reference")
    plt.hist(current_data[column], alpha=0.5, label="Production")
    plt.title(f"Data drift - {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"drift_{column}.png")
    plt.close()

print("===== Drift Analysis =====")

for col in common_columns:
    train_mean = reference_data[col].mean()
    prod_mean = current_data[col].mean()
    drift = abs(prod_mean - train_mean)

    print(
        f"{col}: "
        f"reference={train_mean:.2f} "
        f"production={prod_mean:.2f} "
        f"drift={drift:.2f}"
    )

    if drift > reference_data[col].std():
        print(f"ALERTE DRIFT : {col}")

print("Data drift reports generated in reports/")