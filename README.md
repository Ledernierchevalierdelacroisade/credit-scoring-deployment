# Credit Scoring Deployment

Projet MLOps de déploiement d’un modèle de scoring crédit.

## Objectif

Déployer un modèle de machine learning via une API FastAPI avec :

- Docker
- GitHub Actions CI/CD
- Tests automatisés
- Interface Gradio
- Monitoring simple du data drift
- Déploiement cloud via Hugging Face Spaces

---

## Stack technique

- Python
- FastAPI
- Gradio
- Docker
- Pytest
- GitHub Actions
- LightGBM
- Hugging Face Spaces

---

## Structure du projet

```bash
src/
├── api/
├── monitoring/
├── app_gradio.py

models/
tests/
reports/