# Rapport de Projet — Maintenance Prédictive Industrielle
**Système Intelligent Multi-Modèles pour la Maintenance Prédictive Industrielle**

*Projet M2 Data Science — EFREI 2025-26*
*Épreuve certifiante RNCP36739 — Bloc 4*

---

## Table des matières

1. [Introduction et contexte](#1-introduction-et-contexte)
2. [Description du dataset](#2-description-du-dataset)
3. [Analyse Exploratoire des Données (EDA)](#3-analyse-exploratoire-des-données-eda)
4. [Méthodologie](#4-méthodologie)
5. [Modèles implémentés](#5-modèles-implémentés)
6. [Résultats et comparaison](#6-résultats-et-comparaison)
7. [Interprétabilité du modèle final](#7-interprétabilité-du-modèle-final)
8. [Dashboard décisionnel](#8-dashboard-décisionnel)
9. [Analyse critique et limites](#9-analyse-critique-et-limites)
10. [Conclusion et recommandations](#10-conclusion-et-recommandations)

---

## 1. Introduction et contexte

### 1.1 Problématique industrielle

Dans les environnements industriels modernes, les équipements (CNC, pompes, compresseurs, bras robotiques) génèrent en permanence des données issues de capteurs physiques : vibrations, température moteur, pression, courant électrique, vitesse de rotation (RPM). Ces signaux contiennent des patterns annonciateurs de défaillances, souvent imperceptibles à l'œil humain.

La maintenance corrective (réparation après panne) engendre des coûts importants :
- Arrêts de production non planifiés
- Coûts de réparation d'urgence élevés
- Risques pour la sécurité des opérateurs

La **maintenance prédictive** vise à anticiper ces pannes avant qu'elles surviennent, en exploitant les données capteurs via des algorithmes d'apprentissage automatique.

### 1.2 Objectifs du projet

Ce projet a pour objectif de développer un **MVP (Minimum Viable Product)** de plateforme de maintenance prédictive comprenant :

1. Un pipeline complet de préparation des données
2. Une comparaison rigoureuse de 4 modèles ML/DL
3. Un dashboard décisionnel interactif
4. Une analyse d'interprétabilité (Feature Importance + SHAP)

### 1.3 Tâche prédictive retenue

**Prédiction binaire de panne dans les 24 heures**
- Variable cible : `failure_within_24h` (0 = pas de panne, 1 = panne imminente)
- Justification : tâche la plus directement exploitable opérationnellement, permettant de planifier des interventions préventives

---

## 2. Description du dataset

**Source :** Kaggle — Industrial Machine Predictive Maintenance Dataset
**Fichier :** `predictive_maintenance_v3.csv`

### 2.1 Caractéristiques générales

| Attribut | Valeur |
|---|---|
| Nombre d'enregistrements | 24 042 |
| Nombre de variables | 15 |
| Période couverte | 2024 (données simulées) |
| Fréquence d'échantillonnage | Toutes les ~3 minutes |
| Types de machines | CNC, Pump, Compressor, Robotic Arm |

### 2.2 Variables du dataset

| Variable | Type | Description | Statut |
|---|---|---|---|
| `timestamp` | datetime | Horodatage de la mesure | Supprimé (identifiant) |
| `machine_id` | int | Identifiant de la machine | Supprimé (identifiant) |
| `machine_type` | string | Type de machine | Feature catégorielle |
| `vibration_rms` | float | Vibration RMS | Feature numérique |
| `temperature_motor` | float | Température moteur (°C) | Feature numérique |
| `current_phase_avg` | float | Courant électrique moyen (A) | Feature numérique |
| `pressure_level` | float | Niveau de pression | Feature numérique |
| `rpm` | float | Vitesse de rotation | Feature numérique |
| `operating_mode` | string | Mode opératoire (idle/normal/peak) | Feature catégorielle |
| `hours_since_maintenance` | float | Heures depuis dernière maintenance | Feature numérique |
| `ambient_temp` | float | Température ambiante (°C) | Feature numérique |
| `rul_hours` | float | Durée de vie restante estimée | **Supprimé (data leakage)** |
| `failure_within_24h` | int | **Variable cible** (0/1) | Cible |
| `failure_type` | string | Type de panne | **Supprimé (data leakage)** |
| `estimated_repair_cost` | int | Coût estimé de réparation | **Supprimé (data leakage)** |

### 2.3 Justification de la suppression des variables leakage

La notion de **data leakage** désigne l'utilisation d'informations qui ne seraient pas disponibles au moment de la prédiction réelle :

- `failure_type` : révèle directement qu'une panne est en cours → leakage évident
- `rul_hours` : encode implicitement la cible (si rul_hours < 24 → failure_within_24h = 1). Corrélation mesurée : -0.25
- `estimated_repair_cost` : calculé à partir de la survenue d'une panne → non disponible avant la panne

Utiliser ces variables aurait produit des performances artificiellement élevées, invalidant les résultats.

---

## 3. Analyse Exploratoire des Données (EDA)

### 3.1 Distribution de la variable cible

| Classe | Nombre | Pourcentage |
|---|---|---|
| 0 — Pas de panne | 20 482 | 85.2% |
| 1 — Panne dans 24h | 3 560 | 14.8% |

Le dataset présente un **déséquilibre significatif** (ratio ~5.75:1). Ce déséquilibre est réaliste dans un contexte industriel : les pannes sont des événements rares. Sans traitement, les modèles tendraient à prédire systématiquement "pas de panne" et afficheraient 85% d'accuracy sans jamais détecter une panne réelle.

### 3.2 Répartition par type de machine

| Machine | Observations |
|---|---|
| CNC | ~25% |
| Pump | ~25% |
| Compressor | ~25% |
| Robotic Arm | ~25% |

La répartition est équilibrée entre les 4 types de machines, ce qui garantit la représentativité du dataset pour chaque type d'équipement.

### 3.3 Valeurs manquantes

| Variable | Valeurs manquantes | % |
|---|---|---|
| `vibration_rms` | 1 000 | 4.2% |
| `temperature_motor` | 834 | 3.5% |
| `current_phase_avg` | 731 | 3.0% |
| `pressure_level` | 924 | 3.8% |
| `rpm` | 533 | 2.2% |

Les valeurs manquantes représentent entre 2% et 4% de chaque capteur. Elles sont probablement dues à des défaillances temporaires de capteurs ou des interruptions de communication. Une **imputation par la médiane** a été choisie pour sa robustesse face aux valeurs extrêmes caractéristiques des données industrielles.

### 3.4 Observations sur les distributions

L'analyse des distributions par statut (panne / normal) révèle des patterns distincts :

- **vibration_rms** : Les observations de classe 1 présentent des valeurs systématiquement plus élevées, confirmant que les vibrations anormales précèdent les pannes
- **temperature_motor** : Une surchauffe moteur est fortement associée aux pannes imminentes (distribution décalée vers les hautes températures)
- **hours_since_maintenance** : Les machines dont la dernière maintenance est ancienne présentent un risque accru, ce qui est cohérent avec l'usure progressive
- **operating_mode** : Le mode `peak` est sur-représenté dans les observations de panne

### 3.5 Corrélations

La matrice de corrélation ne révèle pas de multicolinéarité forte entre les features, ce qui valide la pertinence de conserver toutes les variables dans les modèles. La corrélation la plus notable avec la cible est celle de `vibration_rms` et `temperature_motor`.

---

## 4. Méthodologie

### 4.1 Pipeline de préparation des données

```
Données brutes (24 042 lignes, 15 colonnes)
        ↓
Suppression des colonnes (identifiants + data leakage)
        ↓
Séparation features / cible
        ↓
Split stratifié 80/20 (train: 19 233 | test: 4 809)
        ↓
ColumnTransformer (appliqué uniquement sur train, puis transformé sur test)
   ├── Numériques (7 variables) : SimpleImputer(median) → StandardScaler
   └── Catégorielles (2 variables) : SimpleImputer(mode) → OneHotEncoder
```

### 4.2 Prévention du data leakage

L'ensemble du preprocessing est intégré dans des **sklearn Pipelines**. Cela garantit que :
- L'imputation (médiane) est calculée uniquement sur les données d'entraînement
- Le StandardScaler (moyenne, écart-type) est ajusté uniquement sur le train set
- L'OneHotEncoder apprend les catégories uniquement sur le train set

Ces statistiques sont ensuite appliquées (transformées) sur le test set, sans jamais "voir" les données de test pendant l'entraînement.

### 4.3 Stratégie de gestion du déséquilibre de classes

| Modèle | Stratégie |
|---|---|
| Logistic Regression | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight=5.75` |
| MLP | `early_stopping` pour éviter l'overfitting sur la majorité |

### 4.4 Métriques d'évaluation

Dans un contexte de maintenance industrielle, **le Recall est la métrique prioritaire** :
- Un **faux négatif** (panne non détectée) entraîne un arrêt brutal, des coûts de réparation d'urgence élevés et des risques de sécurité
- Un **faux positif** (fausse alerte) entraîne une intervention préventive inutile, certes coûteuse, mais beaucoup moins grave

Le **F1-Score** est utilisé comme critère de sélection principal car il équilibre Precision et Recall. Le **ROC-AUC** complète l'évaluation en mesurant la qualité discriminante indépendamment du seuil de décision.

### 4.5 Validation croisée

Une **cross-validation stratifiée 5-fold** a été appliquée sur le modèle retenu (XGBoost) pour évaluer sa stabilité et sa capacité de généralisation sur l'ensemble du dataset.

---

## 5. Modèles implémentés

### 5.1 Logistic Regression — Modèle baseline

**Principe :** modélise la probabilité de panne comme une combinaison linéaire des features. C'est le modèle de référence pour évaluer si des approches plus complexes apportent un gain réel.

**Paramètres :** `C=1.0`, `solver='lbfgs'`, `max_iter=1000`, `class_weight='balanced'`

**Forces :** très interprétable (coefficients = impact direct de chaque feature), rapide à entraîner, robuste
**Limites :** ne capture pas les interactions non linéaires entre variables

### 5.2 Random Forest — Modèle ensembliste

**Principe :** ensemble de 200 arbres de décision entraînés sur des sous-échantillons aléatoires du dataset (bagging). La prédiction finale est la moyenne des probabilités de chaque arbre.

**Paramètres :** `n_estimators=200`, `max_depth=15`, `class_weight='balanced'`

**Forces :** capture les non-linéarités, robuste aux outliers, fournit une feature importance native
**Limites :** moins performant que le boosting sur ce type de dataset, modèle volumineux en mémoire

### 5.3 XGBoost — Gradient Boosting

**Principe :** construction séquentielle d'arbres, chaque nouvel arbre corrigeant les erreurs du précédent (boosting). Utilise une descente de gradient sur la fonction de perte.

**Paramètres :** `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `scale_pos_weight=5.75`

**Forces :** meilleure performance prédictive sur données tabulaires, gestion native du déséquilibre via `scale_pos_weight`, régularisation intégrée (L1, L2)
**Limites :** hyperparamétrage plus complexe, moins interprétable que la régression logistique

### 5.4 MLP — Deep Learning (Réseau de neurones multicouche)

**Principe :** réseau de neurones artificiels avec 3 couches cachées (128 → 64 → 32 neurones), fonction d'activation ReLU, optimiseur Adam.

**Architecture :**
```
Input (11 features après OHE) → 128 → 64 → 32 → Output (1 neurone sigmoïde)
```

**Paramètres :** `hidden_layer_sizes=(128, 64, 32)`, `activation='relu'`, `alpha=1e-4`, `early_stopping=True`

**Forces :** peut capturer des interactions complexes et non linéaires sans feature engineering manuel
**Limites :** nécessite plus de données pour atteindre son plein potentiel, boîte noire, sensible à l'initialisation aléatoire

---

## 6. Résultats et comparaison

### 6.1 Tableau des performances (ensemble de test, n=4 809)

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.910 | 0.641 | 0.895 | 0.747 | 0.959 |
| Random Forest | 0.965 | 0.859 | 0.916 | 0.887 | 0.993 |
| **XGBoost** ✔ | **0.968** | 0.847 | **0.955** | **0.898** | **0.996** |
| MLP (Deep Learning) | 0.955 | **0.847** | 0.853 | 0.850 | 0.984 |

### 6.2 Analyse des résultats

**Logistic Regression :**
Bien que le Recall soit satisfaisant (0.895), la Precision faible (0.641) indique de nombreuses fausses alertes. La limitation linéaire du modèle ne permet pas de capturer les interactions complexes entre capteurs. Ce modèle sert de baseline : tous les autres modèles le surpassent significativement.

**Random Forest :**
Excellentes performances générales (F1=0.887, ROC-AUC=0.993). Le gain par rapport à la régression logistique (+0.14 en F1) confirme la présence de relations non linéaires dans les données. Cependant, XGBoost le surpasse sur tous les critères.

**XGBoost (modèle retenu) :**
Meilleur compromis toutes métriques confondues. Le Recall de 0.955 est particulièrement remarquable dans le contexte industriel : 95.5% des pannes réelles sont détectées. Le ROC-AUC de 0.996 indique une quasi-parfaite discrimination entre les classes. La cross-validation confirme la stabilité : **F1 = 0.9026 ± 0.0099**.

**MLP (Deep Learning) :**
Performances honorables (F1=0.850, ROC-AUC=0.984), mais inférieures à XGBoost sur ce dataset. Ce résultat illustre un principe fondamental : le Deep Learning n'est pas toujours supérieur aux méthodes classiques, en particulier sur des données tabulaires avec des features bien ingénieurées. Avec seulement 24 000 observations et 9 features, les arbres de décision boostés ont l'avantage.

### 6.3 Cross-validation XGBoost (5-fold stratifié)

| Fold | F1-Score |
|---|---|
| 1 | 0.8930 |
| 2 | 0.8915 |
| 3 | 0.9105 |
| 4 | 0.9008 |
| 5 | 0.9169 |
| **Moyenne** | **0.9026** |
| **Écart-type** | **0.0099** |

L'écart-type de 0.01 confirme que le modèle est **stable** et généralise bien à différentes partitions du dataset. Il n'est pas surappris sur un découpage particulier.

### 6.4 Justification du choix du modèle final

**XGBoost est retenu** sur la base de :

| Critère | Évaluation |
|---|---|
| Performance (F1, ROC-AUC) | ✅ Meilleur de tous les modèles |
| Recall (détection des pannes) | ✅ 0.955 — crucial en contexte industriel |
| Stabilité (CV) | ✅ Faible variance entre folds |
| Gestion du déséquilibre | ✅ `scale_pos_weight` natif |
| Interprétabilité | ✅ Feature importance + SHAP (TreeExplainer) |
| Coût computationnel | ✅ Entraînement rapide (~30 secondes) |
| Déploiement | ✅ Sérialisation joblib légère (~600 KB) |

---

## 7. Interprétabilité du modèle final

### 7.1 Feature Importance native (XGBoost)

L'importance des variables dans XGBoost est calculée par le **gain moyen** apporté par chaque variable lors des splits dans tous les arbres.

Les variables les plus importantes identifiées sont :
1. **`vibration_rms`** — indicateur principal de dégradation mécanique
2. **`temperature_motor`** — précurseur de défaillances thermiques
3. **`hours_since_maintenance`** — mesure de l'usure accumulée
4. **`rpm`** — stress mécanique lié à la vitesse
5. **`pressure_level`** — stress hydraulique/pneumatique

Ces résultats sont **cohérents avec la physique des machines industrielles** : les vibrations et la température sont les premiers signaux de dégradation avant une panne.

### 7.2 SHAP (SHapley Additive exPlanations)

Les valeurs SHAP permettent d'aller plus loin que la Feature Importance globale en expliquant **chaque prédiction individuelle** :

- Pour chaque observation, SHAP quantifie la contribution de chaque variable à l'écart par rapport à la prédiction moyenne
- Les valeurs positives augmentent la probabilité de panne, les valeurs négatives la diminuent
- Le SHAP Summary Plot (beeswarm) montre simultanément l'importance globale et la direction de l'effet

**Exemples d'interprétation SHAP :**
- Une `vibration_rms` élevée (point rouge, valeur SHAP positive) augmente significativement la probabilité de panne prédite
- Un `hours_since_maintenance` faible (machine récemment entretenue) réduit le risque prédit
- Le mode `operating_mode = peak` contribue positivement à la probabilité de panne

### 7.3 Utilité pour le responsable maintenance

Ces explications permettent à un responsable maintenance de répondre à des questions concrètes :
- *"Pourquoi cette machine est-elle classée à haut risque ?"* → La vibration RMS est anormalement élevée depuis 2 heures
- *"Quelle action prioritaire ?"* → Vérifier les roulements (cause probable des vibrations)
- *"Le modèle est-il fiable ?"* → Les facteurs déclenchants sont physiquement cohérents

---

## 8. Dashboard décisionnel

Le dashboard Streamlit a été conçu comme un **outil décisionnel autonome**, distinct des visualisations d'analyse scientifique (EDA). Il cible un profil **responsable maintenance** non technique.

### 8.1 Architecture du dashboard

| Onglet | Contenu | Valeur métier |
|---|---|---|
| EDA | Distributions, corrélations, manquants | Compréhension des données machine |
| Modèles | Tableau, ROC, matrices de confusion | Confiance dans le système |
| Prédiction | Formulaire capteurs → score de risque | Usage opérationnel quotidien |
| Interprétabilité | Feature importance, SHAP | Explicabilité des alertes |

### 8.2 Onglet Prédiction — Cas d'usage type

Un responsable maintenance reçoit une alerte. Il entre les valeurs des capteurs de la machine suspecte dans le formulaire. Le dashboard affiche :
- **Score de risque coloré** : 🔴 ÉLEVÉ (probabilité > 60%)
- **Probabilité exacte** : ex. 78.3%
- **Recommandation** : Intervention recommandée dans les 24h

Ce workflow remplace un processus manuel de consultation de dashboards de monitoring qui ne donnerait pas de probabilité explicite de panne.

---

## 9. Analyse critique et limites

### 9.1 Limites du dataset

- **Données simulées :** bien que réalistes, ces données ne contiennent pas la complexité et le bruit d'un environnement industriel réel. Les performances pourraient être inférieures sur données réelles.
- **Absence de données temporelles :** chaque enregistrement est traité indépendamment. Une approche séquentielle (LSTM, séries temporelles) pourrait capturer des tendances d'évolution des capteurs avant la panne.
- **Pas de données de contexte :** l'âge de la machine, son historique complet de pannes ou la qualité des pièces utilisées en maintenance ne sont pas inclus.

### 9.2 Limites méthodologiques

- **Seuil de décision fixe à 0.5 :** dans un contexte réel, ce seuil devrait être ajusté selon le coût relatif des faux positifs vs faux négatifs
- **Pas de monitoring de dérive (data drift) :** en production, les distributions des capteurs peuvent évoluer dans le temps (vieillissement des machines, changements de process), ce qui dégraderait progressivement les performances
- **MLP limité :** avec 24 000 observations, un réseau neuronal plus profond (LSTM, Transformer) ne bénéficierait pas d'assez de données pour se distinguer

### 9.3 Comparaison ML vs Deep Learning

Le MLP (Deep Learning) avec 0.850 de F1 est inférieur à XGBoost (0.898). Ce résultat illustre un principe important : **sur des données tabulaires de taille modérée avec des features bien définies, les méthodes d'ensemble basées sur les arbres (XGBoost, Random Forest) surpassent généralement les réseaux de neurones**. Le Deep Learning excelle sur des données non structurées (images, texte, sons) ou des séquences temporelles longues.

---

## 10. Conclusion et recommandations

### 10.1 Bilan

Ce projet a démontré la faisabilité d'un système de maintenance prédictive basé sur le Machine Learning :

- **XGBoost** est le modèle retenu avec un F1-Score de **0.898** et un Recall de **0.955** (95.5% des pannes détectées)
- La cross-validation confirme la stabilité du modèle (F1 = 0.9026 ± 0.0099)
- Le pipeline sklearn avec ColumnTransformer garantit l'absence de data leakage
- Le dashboard Streamlit offre un outil opérationnel exploitable par un profil non technique

### 10.2 Recommandations pour une mise en production

| Recommandation | Priorité |
|---|---|
| Ajuster le seuil de décision selon le ratio coût panne / coût intervention | Haute |
| Intégrer des features temporelles (rolling mean sur 1h, 6h, 24h des capteurs) | Haute |
| Mettre en place un monitoring de dérive des distributions (data drift) | Moyenne |
| Réentraîner périodiquement le modèle avec les nouvelles données | Moyenne |
| Déployer une API REST pour intégration dans le SCADA / ERP industriel | Optionnel |
| Explorer des approches LSTM pour capturer les tendances temporelles | Optionnel |

### 10.3 Impact métier estimé

Un système de maintenance prédictive avec un Recall de 95.5% permettrait, dans un environnement industriel réel :
- De **détecter 19 pannes sur 20** avant leur survenue
- De planifier des interventions préventives à moindre coût
- De réduire les arrêts non planifiés et leurs impacts sur la production
- D'optimiser le stock de pièces détachées (en anticipant le type d'intervention)

---

*Rapport rédigé dans le cadre du Projet M2 Data Science — EFREI 2025-26*
*Validation RNCP36739 — Bloc 4 : Implémenter des méthodes d'intelligence artificielle*
