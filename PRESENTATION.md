# Support de Présentation
## Maintenance Prédictive Industrielle
**Projet M2 Data Science — EFREI 2025-26**
*RNCP36739 — Bloc 4*

> Ce fichier est structuré slide par slide.
> Il peut être importé dans PowerPoint, Google Slides ou Canva.
> Pour une conversion automatique en slides : utiliser **Marp** (extension VS Code) ou **pandoc**.

---

---
## SLIDE 1 — Titre

# Système Intelligent Multi-Modèles
# pour la Maintenance Prédictive Industrielle

**Projet M2 Data Science — EFREI 2025-26**

*[Nom Binôme 1] | [Nom Binôme 2]*

*RNCP36739 — Bloc 4 : Implémenter des méthodes d'IA*

---

---
## SLIDE 2 — Contexte et problématique

### Contexte industriel

Les équipements industriels génèrent en permanence des données capteurs :
- Vibrations, température moteur, pression, RPM, courant électrique

### Problème
Une panne non anticipée entraîne :
- ❌ Arrêt de production non planifié
- ❌ Coûts de réparation d'urgence élevés
- ❌ Risques de sécurité

### Notre solution
**Prédire les pannes 24h à l'avance** grâce au Machine Learning

> *"Passer d'une maintenance réactive à une maintenance prédictive"*

---

---
## SLIDE 3 — Dataset

### Industrial Machine Predictive Maintenance

| Caractéristique | Valeur |
|---|---|
| Enregistrements | 24 042 |
| Variables | 15 (9 retenues) |
| Machines | CNC, Pump, Compressor, Robotic Arm |
| Variable cible | `failure_within_24h` (0/1) |
| Déséquilibre | 85% / 15% |

### Features utilisées
`vibration_rms` · `temperature_motor` · `current_phase_avg`
`pressure_level` · `rpm` · `hours_since_maintenance`
`ambient_temp` · `machine_type` · `operating_mode`

### Variables supprimées (data leakage)
`failure_type` · `rul_hours` · `estimated_repair_cost`

---

---
## SLIDE 4 — Analyse Exploratoire (EDA)

### Déséquilibre de classes
```
Pas de panne  : 20 482 obs. (85.2%)
Panne < 24h   :  3 560 obs. (14.8%)
```
→ Nécessite une gestion explicite du déséquilibre

### Valeurs manquantes (capteurs)
| Capteur | NaN |
|---|---|
| vibration_rms | 1 000 (4.2%) |
| pressure_level | 924 (3.8%) |
| temperature_motor | 834 (3.5%) |

→ Imputation par la **médiane** (robuste aux outliers industriels)

### Observations clés
- Vibrations et température significativement plus élevées avant une panne
- Mode `peak` sur-représenté dans les pannes
- Pas de multicolinéarité forte entre features

---

---
## SLIDE 5 — Architecture du pipeline

```
Dataset brut (24 042 lignes)
        │
        ▼
Suppression des colonnes leakage
        │
        ▼
Split stratifié 80/20
┌────────────────┬──────────────┐
│   TRAIN        │    TEST      │
│  19 233 obs.   │  4 809 obs.  │
└────────────────┴──────────────┘
        │
        ▼
ColumnTransformer (sur TRAIN uniquement)
   ├── Numériques : Median Imputer + StandardScaler
   └── Catégorielles : Mode Imputer + OneHotEncoder
        │
        ▼
   4 Modèles ML/DL
```

### Principe anti-leakage
Toutes les statistiques de preprocessing (médiane, écart-type, catégories)
sont calculées sur le **train set uniquement**, puis appliquées au test set.

---

---
## SLIDE 6 — Les 4 modèles

| # | Modèle | Architecture | Gestion déséquilibre |
|---|---|---|---|
| 1 | **Logistic Regression** | Linéaire, baseline | `class_weight='balanced'` |
| 2 | **Random Forest** | 200 arbres (bagging) | `class_weight='balanced'` |
| 3 | **XGBoost** | 200 arbres (boosting) | `scale_pos_weight=5.75` |
| 4 | **MLP (Deep Learning)** | 128→64→32 neurones | `early_stopping` |

### Choix des métriques
| Métrique | Rôle |
|---|---|
| **Recall** | Prioritaire : détecter toutes les pannes |
| **F1-Score** | Compromis Precision / Recall |
| **ROC-AUC** | Performance globale (seuil-indépendante) |

> ⚠️ L'Accuracy seule est trompeuse sur un dataset déséquilibré

---

---
## SLIDE 7 — Résultats comparatifs

| Modèle | Accuracy | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.910 | 0.895 | 0.747 | 0.959 |
| Random Forest | 0.965 | 0.916 | 0.887 | 0.993 |
| **XGBoost** ✅ | **0.968** | **0.955** | **0.898** | **0.996** |
| MLP (Deep Learning) | 0.955 | 0.853 | 0.850 | 0.984 |

### Progression visible
```
Logistic Regression → Random Forest → XGBoost
F1 : 0.747          → 0.887         → 0.898
```
Chaque niveau de complexité apporte un gain mesurable et justifié.

---

---
## SLIDE 8 — Modèle retenu : XGBoost

### Pourquoi XGBoost ?

| Critère | Score |
|---|---|
| F1-Score (test) | **0.898** — meilleur |
| Recall (test) | **0.955** — 95.5% des pannes détectées |
| ROC-AUC | **0.996** — quasi-parfait |
| Stabilité (CV 5-fold) | **0.9026 ± 0.0099** |

### Cross-validation (5-fold stratifié)
```
Fold 1 : 0.8930
Fold 2 : 0.8915
Fold 3 : 0.9105
Fold 4 : 0.9008
Fold 5 : 0.9169
─────────────────
Moyenne : 0.9026  |  Écart-type : 0.0099
```
✅ Faible variance → modèle **stable et généralisable**

---

---
## SLIDE 9 — ML vs Deep Learning

### Résultat observé
XGBoost (0.898 F1) > MLP Deep Learning (0.850 F1)

### Pourquoi le Deep Learning est moins performant ici ?

| Facteur | Impact |
|---|---|
| Taille du dataset (24k obs.) | Insuffisant pour exploiter pleinement un réseau de neurones |
| Features tabulaires structurées | Les arbres de décision y sont naturellement adaptés |
| Features déjà bien ingénieurées | Peu de valeur ajoutée des représentations apprises |

### Conclusion pédagogique
> **Le Deep Learning n'est pas universellement supérieur.**
> Sur des données tabulaires de taille modérée, les méthodes ensemblistes
> (Random Forest, XGBoost) restent la référence.
> Le Deep Learning excelle sur images, texte, séries temporelles longues.

---

---
## SLIDE 10 — Interprétabilité (SHAP)

### Feature Importance (XGBoost — top 5)
1. `vibration_rms` — indicateur principal de dégradation mécanique
2. `temperature_motor` — précurseur de défaillances thermiques
3. `hours_since_maintenance` — mesure de l'usure accumulée
4. `rpm` — stress mécanique
5. `pressure_level` — anomalie hydraulique

### SHAP Summary Plot
- Chaque point = 1 observation
- Couleur rouge = valeur de feature élevée
- Position droite = augmente le risque de panne

### Interprétation métier
> *"Pourquoi cette machine est-elle à haut risque ?"*
> → vibration_rms anormalement élevée + température moteur en hausse
> → Intervention recommandée : vérification des roulements

✅ Le modèle est **explicable** et **actionnable**

---

---
## SLIDE 11 — Dashboard Streamlit

### 4 onglets décisionnels

| Onglet | Contenu |
|---|---|
| 📊 EDA | Distributions, corrélations, manquants |
| 🤖 Modèles | Performances, courbes ROC, matrices de confusion |
| 🔮 Prédiction | Sliders capteurs → score de risque en temps réel |
| 🔍 Interprétabilité | Feature Importance + SHAP |

### Cas d'usage — Prédiction en temps réel
```
Responsable maintenance
        │
        ▼ (entre les valeurs des capteurs)
Dashboard
        │
        ▼
🔴 RISQUE ÉLEVÉ — Probabilité : 78.3%
→ Intervention recommandée dans les 24h
```

### Lancement
```bash
streamlit run dashboard/app.py
```

---

---
## SLIDE 12 — Architecture technique

```
predictive_maintenance_v3.csv
              │
              ▼
         main.py
    ┌──────────────────┐
    │  src/            │
    │  preprocessing.py│  ← Pipeline sklearn anti-leakage
    │  models.py       │  ← 4 modèles en Pipeline
    │  evaluation.py   │  ← Métriques + SHAP + Figures
    └──────────────────┘
              │
              ▼
       saved_models/
    ├── best_model.pkl     ← XGBoost (~600 KB)
    ├── model_comparison.csv
    ├── shap_values.npy
    └── figures/*.png
              │
              ▼
     dashboard/app.py
     (Streamlit — localhost:8501)
```

---

---
## SLIDE 13 — Limites et perspectives

### Limites actuelles
- Données simulées (performances pourraient être inférieures en production)
- Pas de features temporelles (tendances d'évolution des capteurs)
- Seuil de décision fixe à 0.5 (non optimisé)
- Pas de monitoring de dérive des données (data drift)

### Perspectives d'amélioration
| Amélioration | Impact attendu |
|---|---|
| Features rolling (moyenne sur 1h, 6h) | +3–5% F1 estimé |
| Ajustement du seuil de décision | +5–10% Recall |
| Monitoring data drift | Fiabilité long terme |
| API REST (FastAPI) | Intégration SCADA/ERP |
| LSTM sur séries temporelles | Capture des tendances |

---

---
## SLIDE 14 — Conclusion

### Ce que nous avons réalisé

✅ Pipeline ML complet sans data leakage (sklearn Pipelines)

✅ Comparaison rigoureuse de 4 modèles (ML classique + Deep Learning)

✅ **XGBoost retenu** : F1 = 0.898 | Recall = 0.955 | ROC-AUC = 0.996

✅ Cross-validation confirme la stabilité (0.9026 ± 0.0099)

✅ Interprétabilité complète (Feature Importance + SHAP)

✅ Dashboard Streamlit opérationnel (4 onglets)

### Impact métier estimé
> Avec un Recall de **95.5%**, le système détecte
> **19 pannes sur 20** avant leur survenue,
> permettant de planifier des interventions préventives.

---

---
## SLIDE 15 — Questions ?

# Merci pour votre attention

**Démonstration du dashboard disponible**

```
streamlit run dashboard/app.py
→ http://localhost:8501
```

---
*Projet M2 Data Science — EFREI 2025-26*
*RNCP36739 — Bloc 4 : Implémenter des méthodes d'IA*
*[Nom Binôme 1] | [Nom Binôme 2]*
