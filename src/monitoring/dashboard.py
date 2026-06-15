from pathlib import Path
import json

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
LOG_FILE = BASE_DIR / "logs" / "predictions.jsonl"

st.set_page_config(
    page_title="Monitoring Credit Scoring",
    layout="wide"
)

st.title("📊 Dashboard de monitoring - Credit Scoring")

if not LOG_FILE.exists():
    st.warning("Aucun fichier de logs trouvé.")
    st.stop()

logs = []

with open(LOG_FILE, "r", encoding="utf-8") as f:
    for line in f:
        logs.append(json.loads(line))

df = pd.DataFrame(logs)

if df.empty:
    st.warning("Aucune prédiction enregistrée.")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["probability"] = df["probability"].astype(float)
df["prediction"] = df["prediction"].astype(int)

# ===== KPIs =====
col1, col2, col3 = st.columns(3)

col1.metric("Nombre de prédictions", len(df))
col2.metric("Probabilité moyenne", round(df["probability"].mean(), 3))
col3.metric("Taux de refus", f"{df['prediction'].mean():.1%}")

st.divider()

# ===== Graphiques =====
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Distribution des probabilités de défaut")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["probability"], bins=10)
    ax.set_xlabel("Probabilité")
    ax.set_ylabel("Nombre de prédictions")
    ax.set_title("Distribution des scores")
    st.pyplot(fig)

with col_right:
    st.subheader("Évolution des probabilités dans le temps")

    df_time = df.sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df_time["timestamp"], df_time["probability"], marker="o")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Probabilité")
    ax.set_title("Score de défaut dans le temps")
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.divider()

# ===== Alertes =====
st.subheader("🚨 Alertes simples")

refusal_rate = df["prediction"].mean()
high_risk_count = (df["probability"] > 0.95).sum()

if refusal_rate > 0.8:
    st.error("Alerte : taux de refus très élevé.")

if high_risk_count > 0:
    st.warning(f"Alerte : {high_risk_count} prédiction(s) à très haut risque détectée(s).")

if refusal_rate <= 0.8 and high_risk_count == 0:
    st.success("Aucune anomalie majeure détectée.")

st.divider()

# ===== Logs bruts =====
st.subheader("Logs de production")

st.dataframe(df)