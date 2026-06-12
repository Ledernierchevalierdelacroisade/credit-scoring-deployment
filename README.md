\# Credit Scoring Deployment



Projet MLOps de déploiement d’un modèle de scoring crédit.



\## Fonctionnalités



\- API FastAPI

\- Interface Gradio

\- Dockerisation

\- Tests automatisés avec Pytest

\- Pipeline CI/CD GitHub Actions



\---



\## Installation



```bash

pip install -r requirements.txt

## Monitoring et Data Drift

Un script de monitoring est disponible dans `src/monitoring/data_drift.py`.

Il compare des données de référence avec des données courantes simulées afin d’identifier un éventuel décalage de distribution sur les variables importantes du modèle.

Les rapports graphiques sont générés dans le dossier `reports/`.