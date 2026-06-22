import time
import requests
import pandas as pd
from pathlib import Path

API_URL = "http://localhost:8000/predict"
HEADERS = {"x-api-key": "dev-secret-key"}

payload = {
    "AMT_CREDIT": 200000,
    "AMT_INCOME_TOTAL": 150000,
    "AMT_ANNUITY": 25000,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -2500
}

results = []

for i in range(30):
    start = time.time()
    response = requests.post(API_URL, json=payload, headers=HEADERS)
    end = time.time()

    results.append({
        "iteration": i + 1,
        "status_code": response.status_code,
        "latency_ms": round((end - start) * 1000, 2)
    })

df = pd.DataFrame(results)

Path("reports").mkdir(exist_ok=True)
df.to_csv("reports/api_benchmark.csv", index=False)

print(df)
print("\nLatence moyenne :", round(df["latency_ms"].mean(), 2), "ms")
print("Latence max :", round(df["latency_ms"].max(), 2), "ms")
print("Résultats sauvegardés dans reports/api_benchmark.csv")