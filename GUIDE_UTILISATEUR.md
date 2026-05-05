# Guide Utilisateur — Maintenance Prédictive Industrielle
Projet M2 Data Science — EFREI 2025-26

---

## Table des matières

1. [Prérequis](#1-prérequis)
2. [Installation](#2-installation)
3. [Structure du projet](#3-structure-du-projet)
4. [Description des fichiers](#4-description-des-fichiers)
5. [Lancer le pipeline d'entraînement](#5-lancer-le-pipeline-dentraînement)
6. [Lancer le dashboard](#6-lancer-le-dashboard)
7. [Comprendre les résultats](#7-comprendre-les-résultats)
8. [Description du dashboard](#8-description-du-dashboard)
9. [Dépannage fréquent](#9-dépannage-fréquent)

---

## 1. Prérequis

- Python 3.10 ou supérieur
- pip (gestionnaire de paquets Python)
- Le fichier dataset `predictive_maintenance_v3.csv` placé à la racine du projet

Source du dataset : https://www.kaggle.com/datasets/tatheerabbas/industrial-machine-predictive-maintenance

Vérifier la version Python installée :
```bash
python --version
```

---

## 2. Installation

### 2.1 Cloner ou copier le projet

Placer tous les fichiers dans un dossier, par exemple :
```
C:\Users\VotreNom\Documents\projet data science\
```

### 2.2 Installer les dépendances

Depuis la racine du projet :
```bash
pip install -r requirements.txt
```

Paquets installés :

| Paquet | Rôle |
|---|---|
| `pandas` | Manipulation des données |
| `numpy` | Calculs numériques |
| `scikit-learn` | Modèles ML, pipelines, métriques |
| `xgboost` | Gradient boosting haute performance |
| `shap` | Explicabilité des modèles |
| `matplotlib` / `seaborn` | Visualisations |
| `streamlit` | Dashboard interactif |
| `joblib` | Sauvegarde et chargement des modèles |

### 2.3 Vérifier l'installation

```bash
python -c "import sklearn, xgboost, shap, streamlit; print('OK')"
```

---

## 3. Structure du projet

```
projet data science/
|
|-- predictive_maintenance_v3.csv     (dataset, a placer ici)
|-- main.py                           (pipeline principal d'entrainement)
|-- requirements.txt                  (dependances Python)
|
|-- src/                              (code source modulaire)
|   |-- __init__.py
|   |-- preprocessing.py              (pipeline de preparation des donnees)
|   |-- models.py                     (definition des 4 modeles ML/DL)
|   |-- evaluation.py                 (metriques, figures, SHAP)
|
|-- dashboard/
|   |-- app.py                        (application Streamlit)
|
|-- saved_models/                     (genere apres execution de main.py)
    |-- best_model.pkl
    |-- XGBoost.pkl
    |-- Random_Forest.pkl
    |-- Logistic_Regression.pkl
    |-- MLP_Deep_Learning.pkl
    |-- model_comparison.csv          (tableau des performances)
    |-- metadata.json                 (informations sur le meilleur modele)
    |-- shap_values.npy               (valeurs SHAP precalculees)
    |-- shap_X_sample.npy
    |-- figures/                      (toutes les figures generees)
        |-- roc_curves.png
        |-- metrics_comparison.png
        |-- feature_importance.png
        |-- shap_summary.png
        |-- cm_XGBoost.png
        |-- cm_Random_Forest.png
        |-- cm_Logistic_Regression.png
        |-- cm_MLP_Deep_Learning.png
```

---

## 4. Description des fichiers

### `src/preprocessing.py`

Ce fichier centralise toute la logique de préparation des données.

Il définit le chemin vers le dataset (`DATA_PATH`) et la variable cible `failure_within_24h`. Il identifie également les colonnes à supprimer pour éviter le data leakage :

- `timestamp` et `machine_id` sont de simples identifiants, pas des prédicteurs utiles
- `failure_type` révèle directement qu'une panne est en cours, ce qui rendrait la prédiction triviale
- `rul_hours` encode implicitement la cible (quand la durée de vie restante est inférieure à 24h, la panne est quasi certaine)
- `estimated_repair_cost` est calculé après la panne, donc indisponible en temps réel

Le pipeline sklearn est construit via un `ColumnTransformer` avec deux branches :
- Variables numériques (7) : imputation par la médiane puis StandardScaler
- Variables catégorielles (2, `machine_type` et `operating_mode`) : imputation par le mode puis OneHotEncoder

La médiane est préférée à la moyenne pour l'imputation car les capteurs industriels génèrent régulièrement des pics de vibration ou de température qui fausseraient une moyenne.

---

### `src/models.py`

Définit les quatre modèles, chacun encapsulé dans un `sklearn.Pipeline` :

| Modèle | Paramètres clés | Rôle |
|---|---|---|
| Logistic Regression | `class_weight='balanced'`, `max_iter=1000` | Baseline interprétable |
| Random Forest | 200 arbres, `class_weight='balanced'` | Capture les non-linéarités |
| XGBoost | `scale_pos_weight=5.75` | Boosting itératif, meilleure performance |
| MLP (Deep Learning) | Couches 128, 64, 32, `early_stopping=True` | Réseau de neurones multicouche |

Le dataset contient 85 % de non-pannes et 15 % de pannes. Sans correction, les modèles ignoreraient les pannes pour maximiser l'accuracy. Les solutions appliquées : `class_weight='balanced'` pour la régression logistique et le Random Forest, `scale_pos_weight=5.75` pour XGBoost (ratio 20 482 / 3 560), et `early_stopping` pour le MLP afin d'éviter le surapprentissage sur la classe majoritaire.

---

### `src/evaluation.py`

Contient toutes les fonctions d'évaluation :

- `evaluate_model()` : calcule Accuracy, Precision, Recall, F1, ROC-AUC
- `compare_models()` : tableau comparatif de tous les modèles
- `plot_confusion_matrix()` : matrice de confusion
- `plot_roc_curves()` : courbes ROC superposées
- `plot_metrics_comparison()` : graphique à barres groupées
- `plot_feature_importance()` : importance native ou par permutation
- `compute_shap()` : valeurs SHAP via TreeExplainer ou KernelExplainer
- `plot_shap_summary()` : graphique SHAP beeswarm
- `save_model()` / `load_model()` : sérialisation joblib

---

### `main.py`

Orchestre l'ensemble du pipeline en 6 étapes :

```
[1] Chargement des données
[2] Split stratifié 80/20
[3] Entraînement des 4 modèles
[4] Comparaison et sélection du meilleur
[5] Sauvegarde (modèles + figures)
[6] Cross-validation 5-fold + SHAP
```

---

### `dashboard/app.py`

Application Streamlit avec 4 onglets :

- EDA : analyse exploratoire des données
- Comparaison des modèles : performances et visualisations
- Prédiction en temps réel : saisie des capteurs et score de risque
- Interprétabilité : feature importance et SHAP

---

## 5. Lancer le pipeline d'entraînement

Depuis la racine du projet :

```bash
python main.py
```

Durées approximatives par étape :

| Étape | Durée |
|---|---|
| Logistic Regression | moins de 5 secondes |
| Random Forest | 30 à 60 secondes |
| XGBoost | 20 à 40 secondes |
| MLP (Deep Learning) | 1 à 3 minutes |
| Cross-validation 5-fold | 2 à 5 minutes |
| SHAP | 1 à 3 minutes |
| Total | environ 5 à 10 minutes |

Sortie attendue en console :

```
=================================================================
  MAINTENANCE PRÉDICTIVE — PIPELINE ML/DL
=================================================================

[1/6] Chargement des données...
  Dataset : 24,042 lignes | 9 features
  Cible   : {0: 20482, 1: 3560}  (déséquilibre 14.8% positifs)

[2/6] Split train/test (80/20, stratifié)...
  Train : 19,233 | Test : 4,809

[3/6] Entraînement des modèles...
  > Logistic Regression            F1=0.7468  ROC-AUC=0.9588
  > Random Forest                  F1=0.8865  ROC-AUC=0.9929
  > XGBoost                        F1=0.8977  ROC-AUC=0.9957
  > MLP (Deep Learning)            F1=0.8495  ROC-AUC=0.9843

[4/6] Comparaison des modèles...
[5/6] Sauvegarde...
[6/6] Cross-validation + SHAP...
  CV F1 : 0.9026 ± 0.0099

Pipeline terminé avec succès!
```

---

## 6. Lancer le dashboard

Utiliser `streamlit run` et non `python` directement.

Depuis la racine du projet :
```bash
streamlit run dashboard/app.py
```

Depuis le dossier dashboard :
```bash
cd dashboard
streamlit run app.py
```

Le navigateur s'ouvre automatiquement sur `http://localhost:8501`.

Le pipeline `main.py` doit avoir été exécuté au préalable pour que les modèles et les figures soient disponibles.

---

## 7. Comprendre les résultats

### Pourquoi l'accuracy seule ne suffit pas

Le dataset est déséquilibré (85 % de non-pannes). Un modèle qui prédirait systématiquement "pas de panne" obtiendrait 85 % d'accuracy sans jamais détecter une seule panne réelle. C'est pour cela que le Recall a été choisi comme métrique principale.

| Métrique | Formule | Ce qu'elle mesure |
|---|---|---|
| Recall | TP / (TP + FN) | Proportion des pannes réelles détectées |
| Precision | TP / (TP + FP) | Proportion des alertes réellement justifiées |
| F1-Score | 2 × (P × R) / (P + R) | Compromis entre Precision et Recall |
| ROC-AUC | Aire sous la courbe ROC | Performance globale, indépendante du seuil |

Dans un contexte industriel, un faux négatif (panne non détectée) coûte bien plus cher qu'une fausse alerte. Le Recall est donc prioritaire sur la Precision.

### Résultats obtenus

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.910 | 0.641 | 0.895 | 0.747 | 0.959 |
| Random Forest | 0.965 | 0.859 | 0.916 | 0.887 | 0.993 |
| XGBoost ✔ | 0.968 | 0.847 | 0.955 | 0.898 | 0.996 |
| MLP (Deep Learning) | 0.955 | 0.847 | 0.853 | 0.850 | 0.984 |

### Cross-validation XGBoost

```
CV F1 (5-fold) : 0.9026 ± 0.0099
```

Un écart-type de 0.01 entre les cinq folds confirme que le modèle est stable et généralise bien sur des données qu'il n'a pas vues pendant l'entraînement.

---

## 8. Description du dashboard

### Onglet 1 — Analyse EDA

- Distribution de la variable cible et visualisation du déséquilibre
- Répartition par type de machine
- Distribution de chaque capteur selon le statut (panne / pas de panne), sélectionnable
- Matrice de corrélation entre toutes les variables
- Bilan des valeurs manquantes par colonne

### Onglet 2 — Comparaison des modèles

- Tableau des performances avec mise en évidence du meilleur score par métrique
- Graphique à barres groupées (toutes métriques, tous modèles)
- Courbes ROC superposées
- Quatre matrices de confusion côte à côte
- Justification du modèle retenu

### Onglet 3 — Prédiction en temps réel

1. Sélectionner le type de machine et le mode de fonctionnement
2. Ajuster les sliders des capteurs (vibration, température, RPM...)
3. Choisir le modèle à utiliser
4. Cliquer sur Prédire
5. Obtenir le niveau de risque coloré (vert / orange / rouge), la probabilité de panne et une jauge visuelle

### Onglet 4 — Interprétabilité

- Feature Importance : quelles variables influencent le plus le modèle, en vision globale
- SHAP Summary Plot : impact de chaque variable sur chaque prédiction individuelle
- Tableau d'interprétation métier des variables les plus importantes

---

## 9. Dépannage fréquent

| Problème | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'xgboost'` | `pip install xgboost` |
| `ModuleNotFoundError: No module named 'shap'` | `pip install shap` |
| `FileNotFoundError: predictive_maintenance_v3.csv` | Placer le CSV à la racine du projet |
| Dashboard affiche "Aucun modèle trouvé" | Lancer `python main.py` d'abord |
| `streamlit: command not found` | `pip install streamlit` puis réessayer |
| Lancement avec `python app.py` au lieu de `streamlit run` | Utiliser `streamlit run dashboard/app.py` depuis la racine |
| Caractères incorrects dans le terminal Windows | Comportement cosmétique de PowerShell/CMD, le pipeline s'exécute correctement |

---

*Guide rédigé dans le cadre du Projet M2 Data Science — EFREI 2025-26*
*Validation RNCP36739 — Bloc 4 : Implémenter des méthodes d'IA*
