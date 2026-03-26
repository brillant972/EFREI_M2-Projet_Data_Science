# Guide Utilisateur — Maintenance Prédictive Industrielle
**Projet M2 Data Science — EFREI 2025-26**

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

- **Python 3.10 ou supérieur**
- **pip** (gestionnaire de paquets Python)
- Le fichier dataset : `predictive_maintenance_v3.csv` (placé à la racine du projet)
  - Source : https://www.kaggle.com/datasets/tatheerabbas/industrial-machine-predictive-maintenance

Vérifier la version Python :
```bash
python --version
```

---

## 2. Installation

### 2.1 Cloner / copier le projet

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
| `joblib` | Sauvegarde / chargement des modèles |

### 2.3 Vérifier l'installation

```bash
python -c "import sklearn, xgboost, shap, streamlit; print('OK')"
```

---

## 3. Structure du projet

```
projet data science/
│
├── predictive_maintenance_v3.csv     ← Dataset (à placer ici)
├── main.py                           ← Pipeline principal d'entraînement
├── requirements.txt                  ← Dépendances Python
│
├── src/                              ← Code source modulaire
│   ├── __init__.py
│   ├── preprocessing.py              ← Pipeline de préparation des données
│   ├── models.py                     ← Définition des 4 modèles ML/DL
│   └── evaluation.py                 ← Métriques, figures, SHAP
│
├── dashboard/
│   └── app.py                        ← Application Streamlit
│
└── saved_models/                     ← Généré après main.py
    ├── best_model.pkl                ← Meilleur modèle sauvegardé
    ├── XGBoost.pkl
    ├── Random_Forest.pkl
    ├── Logistic_Regression.pkl
    ├── MLP_Deep_Learning.pkl
    ├── model_comparison.csv          ← Tableau des performances
    ├── metadata.json                 ← Infos sur le meilleur modèle
    ├── shap_values.npy               ← Valeurs SHAP précalculées
    ├── shap_X_sample.npy
    └── figures/                      ← Toutes les figures générées
        ├── roc_curves.png
        ├── metrics_comparison.png
        ├── feature_importance.png
        ├── shap_summary.png
        ├── cm_XGBoost.png
        ├── cm_Random_Forest.png
        ├── cm_Logistic_Regression.png
        └── cm_MLP_Deep_Learning.png
```

---

## 4. Description des fichiers

### `src/preprocessing.py`

Ce fichier centralise toute la logique de préparation des données.

**Ce qu'il fait :**
- Définit le chemin vers le dataset (`DATA_PATH`)
- Définit la variable cible : `failure_within_24h`
- Identifie les colonnes à supprimer pour éviter le **data leakage** :
  - `timestamp`, `machine_id` → identifiants, pas des prédicteurs
  - `failure_type` → révèle directement le type de panne → leakage
  - `rul_hours` → durée de vie restante encode directement la réponse
  - `estimated_repair_cost` → corrélé à l'occurrence de panne
- Construit le pipeline sklearn via `ColumnTransformer` :
  - **Variables numériques** (7) : imputation par la médiane + StandardScaler
  - **Variables catégorielles** (2 : `machine_type`, `operating_mode`) : imputation par le mode + OneHotEncoder

**Pourquoi la médiane et non la moyenne ?**
Les valeurs aberrantes dans les capteurs industriels (pics de vibration, de température) rendent la médiane plus robuste que la moyenne pour l'imputation.

---

### `src/models.py`

Définit les 4 modèles, chacun intégré dans un `sklearn.Pipeline` :

| Modèle | Paramètres clés | Justification |
|---|---|---|
| **Logistic Regression** | `class_weight='balanced'`, `max_iter=1000` | Baseline interprétable, robuste |
| **Random Forest** | 200 arbres, `class_weight='balanced'` | Capture les non-linéarités, feature importance native |
| **XGBoost** | `scale_pos_weight=5.75` (ratio d'imbalance) | Boosting itératif, meilleure performance |
| **MLP (Deep Learning)** | Couches (128→64→32), `early_stopping=True` | Réseau de neurones profond, apprentissage automatique des représentations |

**Gestion du déséquilibre de classes :**
Le dataset contient 85% de non-pannes et 15% de pannes. Sans correction, les modèles auraient tendance à ignorer les pannes. Solutions appliquées :
- `class_weight='balanced'` pour Logistic Regression et Random Forest
- `scale_pos_weight=5.75` pour XGBoost (≈ 20482 / 3560)
- Le MLP avec `early_stopping` évite l'overfitting sur la classe majoritaire

---

### `src/evaluation.py`

Contient toutes les fonctions d'évaluation :

- `evaluate_model()` → calcule Accuracy, Precision, Recall, F1, ROC-AUC
- `compare_models()` → tableau comparatif
- `plot_confusion_matrix()` → matrice de confusion
- `plot_roc_curves()` → courbes ROC multi-modèles
- `plot_metrics_comparison()` → graphique à barres groupées
- `plot_feature_importance()` → importance native ou permutation importance
- `compute_shap()` → valeurs SHAP (TreeExplainer ou KernelExplainer)
- `plot_shap_summary()` → graphique SHAP beeswarm
- `save_model()` / `load_model()` → sérialisation joblib

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
- **EDA** — analyse exploratoire
- **Comparaison des modèles** — performances et visualisations
- **Prédiction en temps réel** — saisie des capteurs → score de risque
- **Interprétabilité** — Feature importance et SHAP

---

## 5. Lancer le pipeline d'entraînement

### Depuis la racine du projet :

```bash
python main.py
```

### Durée estimée :
| Étape | Durée approximative |
|---|---|
| Logistic Regression | < 5 secondes |
| Random Forest | 30–60 secondes |
| XGBoost | 20–40 secondes |
| MLP (Deep Learning) | 1–3 minutes |
| Cross-validation (5-fold) | 2–5 minutes |
| SHAP | 1–3 minutes |
| **Total** | **~5–10 minutes** |

### Sortie attendue :
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

**Important :** utiliser `streamlit run` et non `python`.

### Depuis la racine du projet :
```bash
streamlit run dashboard/app.py
```

### Depuis le dossier dashboard :
```bash
cd dashboard
streamlit run app.py
```

Le navigateur s'ouvre automatiquement sur `http://localhost:8501`.

> **Prérequis :** `main.py` doit avoir été exécuté au préalable pour générer les modèles et figures.

---

## 7. Comprendre les résultats

### Métriques utilisées

Pourquoi ne pas se fier uniquement à l'**Accuracy** ?
Le dataset est déséquilibré (85% non-pannes). Un modèle qui prédit "jamais de panne" aurait 85% d'accuracy mais serait complètement inutile.

| Métrique | Formule | Ce qu'elle mesure |
|---|---|---|
| **Recall** | TP / (TP + FN) | Capacité à détecter les vraies pannes → prioritaire |
| **Precision** | TP / (TP + FP) | Part des alertes réellement justifiées |
| **F1-Score** | 2 × (P × R) / (P + R) | Compromis Precision/Recall |
| **ROC-AUC** | Aire sous la courbe ROC | Performance globale indépendante du seuil |

**Dans un contexte industriel, le Recall est prioritaire :**
Un faux négatif (panne non détectée) coûte beaucoup plus cher qu'une fausse alerte.

### Résultats obtenus

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.910 | 0.641 | **0.895** | 0.747 | 0.959 |
| Random Forest | 0.965 | 0.859 | 0.916 | 0.887 | 0.993 |
| **XGBoost** ✔ | **0.968** | 0.847 | **0.955** | **0.898** | **0.996** |
| MLP (Deep Learning) | 0.955 | 0.847 | 0.853 | 0.850 | 0.984 |

### Cross-validation XGBoost
```
CV F1 (5-fold) : 0.9026 ± 0.0099
```
L'écart-type faible (0.01) confirme que le modèle est stable et généralise bien.

---

## 8. Description du dashboard

### Onglet 1 — Analyse EDA
- Distribution de la variable cible (déséquilibre)
- Répartition par type de machine (pie chart)
- Distribution de chaque capteur selon le statut (panneau/normal) — sélectionnable
- Matrice de corrélation entre toutes les variables
- Bilan des valeurs manquantes

### Onglet 2 — Comparaison des modèles
- Tableau des performances avec mise en évidence du meilleur score par métrique
- Graphique à barres groupées (toutes métriques × tous modèles)
- Courbes ROC superposées
- 4 matrices de confusion côte à côte
- Texte d'analyse et justification du modèle retenu

### Onglet 3 — Prédiction en temps réel
1. Sélectionner le type de machine et le mode de fonctionnement
2. Ajuster les sliders des capteurs (vibration, température, RPM…)
3. Choisir le modèle à utiliser
4. Cliquer sur **Prédire**
5. Obtenir : niveau de risque coloré (vert / orange / rouge), probabilité de panne, jauge visuelle

### Onglet 4 — Interprétabilité
- **Feature Importance** : quelles variables influencent le plus le modèle (vision globale)
- **SHAP Summary Plot** : impact de chaque variable sur chaque prédiction individuelle
- Tableau d'interprétation métier des variables les plus importantes

---

## 9. Dépannage fréquent

| Problème | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'xgboost'` | `pip install xgboost` |
| `ModuleNotFoundError: No module named 'shap'` | `pip install shap` |
| `FileNotFoundError: predictive_maintenance_v3.csv` | Placer le CSV à la racine du projet |
| Dashboard : "Aucun modèle trouvé" | Lancer `python main.py` d'abord |
| `streamlit: command not found` | `pip install streamlit` puis réessayer |
| Lancement avec `python app.py` au lieu de `streamlit run app.py` | Utiliser `streamlit run dashboard/app.py` depuis la racine |
| Erreur d'encodage sur Windows | Le terminal PowerShell/CMD peut afficher des caractères incorrects — c'est cosmétique, le pipeline s'exécute correctement |

---

*Guide rédigé dans le cadre du Projet M2 Data Science — EFREI 2025-26*
*Validation RNCP36739 — Bloc 4 : Implémenter des méthodes d'IA*
