# Rapport d'optimisation des performances post-déploiement

## 1. Objectif

L'objectif de cette étape est d'évaluer les performances du modèle de scoring crédit après son déploiement, d'identifier les principaux goulots d'étranglement, de tester une stratégie d'optimisation et de vérifier que cette optimisation n'introduit aucune régression fonctionnelle.

---

# 2. Benchmark initial de l'API

Un benchmark a été réalisé sur l'API FastAPI afin de mesurer les temps de réponse.

Script utilisé :

```bash
python src/monitoring/benchmark_api.py
```

Résultats observés :

* Latence moyenne : **43.87 ms**
* Latence maximale : **1026.03 ms**

La latence maximale correspond au premier appel (warm-up du modèle). Les appels suivants présentent des temps de réponse nettement plus faibles, généralement compris entre 7 et 20 ms.

---

# 3. Profiling avec cProfile

Le module **cProfile** a été utilisé afin d'identifier les parties du code consommant le plus de temps CPU.

Script utilisé :

```bash
python src/monitoring/profile_prediction.py
```

Résultats obtenus :

* 100 prédictions exécutées en **0.446 seconde**
* Temps moyen d'inférence : **4.46 ms** par prédiction

Le rapport de profiling montre que la majorité du temps est consommée par la fonction :

* `LightGBM.predict_proba()`

Les opérations de préparation des données (`DataFrame`, `reindex`) représentent une part beaucoup plus faible du temps d'exécution.

Cette analyse permet d'identifier le modèle de prédiction comme principal goulot d'étranglement.

---

# 4. Optimisation testée

Afin de réduire le temps d'inférence, le modèle LightGBM a été converti au format **ONNX**.

Script utilisé :

```bash
python src/monitoring/convert_to_onnx.py
```

Le modèle converti est enregistré dans :

```text
models/lightgbm_final.onnx
```

Cette conversion permet d'utiliser **ONNX Runtime**, moteur d'exécution optimisé pour l'inférence.

---

# 5. Benchmark du modèle ONNX

Les performances du modèle ONNX ont été mesurées à l'aide du script :

```bash
python src/monitoring/benchmark_onnx.py
```

Résultats obtenus :

* Temps moyen : **0.0485 ms**
* Temps minimum : **0.0238 ms**
* Temps maximum : **0.7625 ms**

Comparativement au modèle LightGBM profilé avec cProfile (~4.46 ms par prédiction), ONNX Runtime réduit fortement le temps d'inférence.

---

# 6. Validation fonctionnelle

Afin de vérifier que l'optimisation ne modifie pas le comportement du modèle, une comparaison a été réalisée entre les prédictions LightGBM et ONNX.

Script utilisé :

```bash
python src/monitoring/compare_models.py
```

Résultats :

| Élément     | LightGBM |     ONNX |
| ----------- | -------: | -------: |
| Probabilité | 0.881779 | 0.881779 |
| Prédiction  |        1 |        1 |

Différence absolue observée :

```
0.00000003
```

Aucune différence significative n'a été observée.

L'optimisation est donc considérée comme **fonctionnellement équivalente**.

---

# 7. Vérification de l'absence de régression

Après l'intégration des optimisations, les tests automatisés ont été réexécutés.

Commande :

```bash
python -m pytest
```

Résultat :

```
11 passed
```

Aucune régression n'a été détectée.

---

# 8. Conclusion

L'analyse des performances a permis d'identifier le modèle LightGBM comme principal goulot d'étranglement.

La conversion vers **ONNX Runtime** constitue une optimisation efficace en réduisant fortement le temps d'inférence tout en conservant les mêmes prédictions que le modèle d'origine.

Les tests automatisés confirment l'absence de régression fonctionnelle après cette optimisation.

Cette étape valide ainsi les objectifs d'optimisation post-déploiement tout en garantissant la stabilité du modèle en production.
