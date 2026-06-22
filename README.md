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

# Étape 1 - Déploiement de l'API et pipeline CI/CD

## Objectif

Déployer un modèle de scoring crédit sous forme d'API et automatiser son intégration via un pipeline CI/CD.

---

## API FastAPI

L'API expose les endpoints suivants :

- GET /
- GET /health
- POST /predict

La documentation Swagger est disponible via :

http://localhost:8000/docs

---

## Dockerisation

L'application a été conteneurisée avec Docker afin de garantir la reproductibilité de l'environnement.

Fichiers utilisés :

- Dockerfile
- requirements.txt

---

## Pipeline GitHub Actions

Un pipeline CI/CD a été mis en place avec GitHub Actions.

Étapes automatisées :

1. Installation des dépendances
2. Exécution des tests Pytest
3. Validation du projet avant déploiement

Fichier :

.github/workflows/ci.yml

---

## Tests automatisés

Les tests sont exécutés automatiquement lors des pushs GitHub.

Couverture :

- Endpoint racine
- Endpoint health
- Endpoint predict
- Cas d'erreurs

---

## Déploiement Hugging Face

Une interface Gradio a été déployée sur Hugging Face Spaces.

Technologies utilisées :

- FastAPI
- Docker
- GitHub Actions
- Gradio
- Hugging Face Spaces

---

## Résultat

Une API de scoring crédit déployée et intégrée dans un pipeline CI/CD fonctionnel.

## Étape 2 — Déploiement de l’API et CI/CD

Cette étape met en place une API de scoring crédit, conteneurisée avec Docker, testée automatiquement avec Pytest et intégrée à un pipeline CI/CD GitHub Actions.

### API FastAPI

L’API est disponible dans :

```bash
src/api/main.py
```

Elle expose les routes suivantes :

```bash
GET /
GET /health
POST /predict
```

La documentation Swagger est disponible après lancement local :

```bash
http://localhost:8000/docs
```

### Sécurité de l’API

La route `/predict` est protégée par une clé API transmise dans le header :

```bash
x-api-key: dev-secret-key
```

En production, cette clé ne doit pas être stockée dans le code. Elle doit être injectée via une variable d’environnement ou via GitHub Secrets.

### Validation des entrées

L’API vérifie plusieurs cas critiques :

* données manquantes ;
* types incorrects ;
* revenu inférieur ou égal à 0 ;
* montant du crédit inférieur ou égal à 0 ;
* montant du crédit irréaliste ;
* annuité inférieure ou égale à 0 ;
* âge incohérent ;
* ancienneté professionnelle incohérente.

Ces contrôles permettent de sécuriser l’API et d’éviter des prédictions sur des données invalides.

### Chargement du modèle

Le modèle est chargé une seule fois au démarrage de l’API :

```python
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)
```

Il n’est donc pas rechargé à chaque requête. Cela réduit le temps de réponse, limite la surcharge mémoire et améliore la scalabilité de l’application.

### Tests automatisés

Les tests sont disponibles dans :

```bash
tests/test_api.py
```

Ils vérifient :

* le bon fonctionnement de `/` ;
* le bon fonctionnement de `/health` ;
* une prédiction valide ;
* l’absence de clé API ;
* les données manquantes ;
* les types incorrects ;
* les valeurs hors plages attendues.

Lancement des tests :

```bash
python -m pytest
```

### Docker

L’API est conteneurisée avec Docker.

Construction de l’image :

```bash
docker build -t credit-scoring-api .
```

Lancement du conteneur :

```bash
docker run -p 8000:8000 credit-scoring-api
```

L’API est ensuite disponible à l’adresse :

```bash
http://localhost:8000/docs
```

### CI/CD GitHub Actions

Le pipeline CI/CD est défini dans :

```bash
.github/workflows/ci.yml
```

Il sépare les étapes suivantes :

* installation de l’environnement ;
* installation des dépendances ;
* exécution des tests ;
* construction de l’image Docker.

Le dernier workflow GitHub Actions doit être vert pour valider l’intégration continue.

### Interface Gradio

Une interface Gradio permet de tester le modèle via une interface simple :

```bash
python src/app_gradio.py
```

Elle permet de saisir des informations utilisateur lisibles, comme l’âge ou l’ancienneté en années, puis convertit ces valeurs dans le format attendu par le modèle.

### Déploiement Hugging Face Spaces

L’interface Gradio est également déployée sur Hugging Face Spaces afin de rendre l’application accessible en ligne.

# Étape 3 - Monitoring et analyse des données de production

## Objectif

Mettre en place un système de monitoring permettant de suivre les prédictions du modèle en production et détecter les dérives de données.

---

## Logging des prédictions

Chaque requête est enregistrée dans :

logs/predictions.jsonl

Informations stockées :

- timestamp
- données d'entrée
- probabilité prédite
- décision
- seuil utilisé

Format :

JSON Lines (JSONL)

---

## Analyse des logs

Script :

src/monitoring/analyze_logs.py

Indicateurs calculés :

- nombre de prédictions
- score moyen
- score minimum
- score maximum
- taux de refus

Exemple :

Nombre de prédictions : 7
Probabilité moyenne : 0.856
Taux de refus : 100 %

---

## Détection d'anomalies

Une alerte est déclenchée lorsque :

- le taux de refus dépasse un seuil critique

Exemple :

ALERTE : taux de refus très élevé

---

## Détection de Data Drift

Script :

src/monitoring/data_drift.py

Comparaison entre :

- données de référence
- données de production

Variables analysées :

- AMT_CREDIT
- AMT_INCOME_TOTAL

Exemple :

AMT_CREDIT :
- référence : 510000
- production : 350000
- drift : 160000

Alerte détectée :

ALERTE DRIFT : AMT_CREDIT

---

## Dashboard Streamlit

Visualisation disponible via :

streamlit run src/monitoring/dashboard.py

Indicateurs affichés :

- nombre de prédictions
- probabilité moyenne
- taux de refus
- distribution des scores
- évolution temporelle
- alertes
- logs de production

---

## Technologies utilisées

- Pandas
- Matplotlib
- Streamlit
- JSONL

---

## Résultat

Un système de monitoring complet permettant :

- l'analyse des prédictions
- la visualisation des métriques
- la détection d'anomalies
- la détection de dérive des données

# Étape 4 - Analyse et optimisation des performances

## Objectif

Mesurer les performances de l'API de scoring crédit et mettre en place des optimisations afin de réduire le temps d'inférence.

---

## Benchmark API

Script :

src/monitoring/benchmark_api.py

Résultat :

- Latence moyenne : 43.87 ms
- Latence maximale : 1026.03 ms

La latence maximale correspond au premier appel (warm-up du modèle).

Les appels suivants sont généralement compris entre 7 et 20 ms.

---

## Profiling avec cProfile

Script :

src/monitoring/profile_prediction.py

Résultat :

100 prédictions exécutées en :

0.446 seconde

Temps moyen :

4.46 ms par prédiction

---

## Identification du goulot d'étranglement

Le profiling montre que la majorité du temps est consommée par :

LightGBM.predict_proba()

Le modèle représente la principale source de latence.

---

## Optimisation avec ONNX Runtime

Conversion du modèle :

src/monitoring/convert_to_onnx.py

Modèle généré :

models/lightgbm_final.onnx

---

## Benchmark ONNX

Script :

src/monitoring/benchmark_onnx.py

Résultats :

- Moyenne : 0.0485 ms
- Minimum : 0.0238 ms
- Maximum : 0.7625 ms

Gain observé :

Environ x90 par rapport à la version LightGBM utilisée lors du profiling.

---

## Validation fonctionnelle

Script :

src/monitoring/compare_models.py

Comparaison :

LightGBM : 0.881779

ONNX : 0.881779

Différence :

0.00000003

Résultat :

Validation OK : aucune régression sur la prédiction.

---

## Validation des tests

Pytest :

11 tests validés

11 passed

Aucune régression détectée après optimisation.

---

## Technologies utilisées

- cProfile
- ONNX Runtime
- LightGBM
- Pandas
- Pytest

---

## Conclusion

L'analyse des performances a permis d'identifier le modèle comme principal goulot d'étranglement.

La conversion vers ONNX Runtime a permis de réduire fortement le temps d'inférence tout en conservant exactement les mêmes prédictions.

Le modèle optimisé est prêt pour une utilisation en production.

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

