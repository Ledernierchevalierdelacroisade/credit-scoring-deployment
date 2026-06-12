import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

# Exemple : comparaison entre données de référence et données récentes simulées
reference_data = pd.DataFrame({
    "AMT_CREDIT": [500000, 300000, 700000, 450000, 600000],
    "AMT_INCOME_TOTAL": [200000, 150000, 250000, 180000, 220000],
})

current_data = pd.DataFrame({
    "AMT_CREDIT": [650000, 400000, 800000, 500000, 750000],
    "AMT_INCOME_TOTAL": [180000, 130000, 210000, 160000, 190000],
})

for column in reference_data.columns:
    plt.figure(figsize=(6, 4))
    plt.hist(reference_data[column], alpha=0.5, label="Reference")
    plt.hist(current_data[column], alpha=0.5, label="Current")
    plt.title(f"Data drift - {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"drift_{column}.png")
    plt.close()

print("Data drift reports generated in reports/")