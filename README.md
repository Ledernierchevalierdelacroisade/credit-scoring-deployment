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

## Performance et scalabilité

Le modèle est chargé une seule fois au démarrage de l’API, en dehors de la route `/predict`.

Cela évite de recharger le modèle à chaque requête, ce qui permet de :

- réduire le temps de réponse ;
- éviter une surcharge mémoire ;
- améliorer la scalabilité de l’API.

La route `/health` permet également de vérifier rapidement que l’API est disponible.

---

## Monitoring et analyse des données de production

L’API enregistre chaque prédiction dans un fichier de logs structuré au format JSONL :

### Détection de dérive des données

### Points de vigilance

- Les données de production doivent être conservées selon les règles RGPD.
- Les informations personnelles sensibles doivent être anonymisées.
- Le volume des logs doit être surveillé afin d'éviter une croissance excessive des coûts de stockage.
- Une référence stable est nécessaire pour interpréter correctement la dérive des données.

Le script :

```bash
python src/monitoring/data_drift.py



```bash
logs/predictions.jsonl

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

