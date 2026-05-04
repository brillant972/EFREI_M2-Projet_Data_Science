---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 22px;
    padding: 40px 50px;
  }
  h1 { font-size: 38px; color: #1a3a5c; }
  h2 { font-size: 28px; color: #1a3a5c; border-bottom: 2px solid #e8701a; padding-bottom: 6px; }
  h3 { font-size: 22px; color: #2c6e9e; }
  table { font-size: 18px; width: 100%; }
  th { background: #1a3a5c; color: white; }
  tr:nth-child(even) { background: #f0f4f8; }
  code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; font-size: 17px; }
  pre { background: #1e1e1e; color: #d4d4d4; padding: 16px; border-radius: 8px; font-size: 16px; }
  .highlight { color: #e8701a; font-weight: bold; }
  section.title { background: linear-gradient(135deg, #1a3a5c 0%, #2c6e9e 100%); color: white; }
  section.title h1 { color: white; font-size: 42px; }
  section.title h2 { color: #f0c060; border: none; }
  section.title p { color: #cce0f5; }
---

<!-- _class: title -->

# Système Intelligent Multi-Modèles
# pour la Maintenance Prédictive Industrielle

## Projet M2 Data Science — EFREI 2025-26

*RNCP36739 — Bloc 4 : Implémenter des méthodes d'IA*

---

## Problème → Solution

| ❌ Maintenance **corrective** | ✅ Maintenance **prédictive** |
|---|---|
| Panne → réparation d'urgence | Détection 24h avant → intervention planifiée |
| Arrêt non planifié coûteux | Intervention préventive moins chère |
| Risque de sécurité | Zéro surprise |

### Notre réponse
**Prédire `failure_within_24h` à partir des données capteurs**
→ Classification binaire supervisée : panne dans les 24h ? **OUI / NON**

### Dataset
> 24 042 observations | 4 types de machines (CNC, Pump, Compressor, Robotic Arm)
> 9 features capteurs : vibration, température, RPM, pression, courant, mode…

---

## EDA — Ce que les données nous ont appris

### Déséquilibre des classes

```
Pas de panne  ████████████████████████████  85.2% — 20 482 obs.
Panne < 24h   █████                          14.8% —  3 560 obs.
                                   Ratio = 5.75 : 1
```

> ⚠️ Un modèle qui dit "jamais de panne" aurait **85% d'Accuracy** mais détecterait **0 panne**
> → L'Accuracy est une métrique trompeuse ici

### Métriques choisies

| Métrique | Formule | Pourquoi |
|---|---|---|
| **Recall** | TP / (TP + FN) | Minimiser les pannes ratées (FN = panne manquée = très coûteux) |
| **F1-Score** | 2×(P×R)/(P+R) | Compromis Precision / Recall |
| **ROC-AUC** | Aire sous courbe ROC | Performance globale, indépendante du seuil |

---

## Pipeline & Anti-Leakage

```
Dataset brut (24 042 lignes, 15 colonnes)
    │
    ▼  Suppression colonnes leakage : failure_type · rul_hours · repair_cost
    │  → Ces colonnes "révèlent" la réponse → résultats artificiellement bons
    ▼
Split STRATIFIÉ 80/20
    ┌─────────────────────────┬────────────────────────┐
    │  TRAIN   19 233 obs.    │  TEST   4 809 obs.     │
    └─────────────────────────┴────────────────────────┘
    │
    ▼  ColumnTransformer (ajusté sur TRAIN → appliqué sur TEST)
       ├── 7 variables numériques  : Median Imputer + StandardScaler
       └── 2 variables catégorielles : Mode Imputer + OneHotEncoder
```

**Règle d'or :** toutes les statistiques de preprocessing sont calculées sur le train set uniquement.
Tester sur des données qui ont influencé l'entraînement = tricher.

---

## Les 4 modèles

| # | Modèle | Principe | Gestion déséquilibre |
|---|---|---|---|
| 1 | **Logistic Regression** | Score linéaire → sigmoïde → probabilité | `class_weight='balanced'` |
| 2 | **Random Forest** | 200 arbres indépendants → vote **(bagging)** | `class_weight='balanced'` |
| 3 | **XGBoost** | 200 arbres séquentiels → chaque arbre corrige le précédent **(boosting)** | `scale_pos_weight=5.75` |
| 4 | **MLP Deep Learning** | 128 → 64 → 32 neurones, ReLU, backpropagation | `early_stopping=True` |

### Bagging vs Boosting

```
BAGGING  : arbre1 ─┐
(Random Forest)    arbre2 ─┤  → vote (parallèle, indépendant)
           arbre3 ─┘

BOOSTING : arbre1 → corrige → arbre2 → corrige → arbre3 → ...  (séquentiel)
(XGBoost)
```

---

## Résultats comparatifs

![w:980px](saved_models/figures/metrics_comparison.png)

| Modèle | Recall | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.895 | 0.747 | 0.959 |
| Random Forest | 0.916 | 0.887 | 0.993 |
| **XGBoost ✅** | **0.955** | **0.898** | **0.996** |
| MLP Deep Learning | 0.853 | 0.850 | 0.984 |

---

## Courbes ROC & Validation

![bg right:55% 95%](saved_models/figures/roc_curves.png)

### Courbes ROC — 4 modèles

- Plus la courbe est proche du **coin supérieur gauche** = meilleur
- La diagonale = modèle aléatoire
- XGBoost : **AUC = 0.996**

### Cross-validation 5-fold (XGBoost)

```
Fold 1 : 0.893
Fold 2 : 0.892
Fold 3 : 0.911
Fold 4 : 0.901
Fold 5 : 0.917
───────────────
Moy : 0.9026 ± 0.0099
```
→ Écart-type faible = modèle **stable, pas d'overfitting**

---

## Matrice de confusion — XGBoost

![bg right:50% 90%](saved_models/figures/cm_XGBoost.png)

### Lecture

|  | Prédit : pas de panne | Prédit : panne |
|---|---|---|
| **Réel : pas de panne** | ✅ TN | ❌ FP (fausse alerte) |
| **Réel : panne** | ❌ **FN** (panne ratée !) | ✅ TP |

### Résultat XGBoost

- **~3 399 pannes détectées** (TP)
- **~161 pannes manquées** (FN)
- → **Recall = 95.5%** = 19 pannes sur 20 détectées

> Le FN est le cas le plus coûteux industriellement → on minimise

---

## Interprétabilité — Feature Importance & SHAP

![w:480px](saved_models/figures/feature_importance.png) ![w:480px](saved_models/figures/shap_summary.png)

**Top 5 features :** `vibration_rms` · `temperature_motor` · `hours_since_maintenance` · `rpm` · `pressure_level`

> SHAP : chaque point = 1 observation | 🔴 valeur élevée → augmente le risque | Position droite → panne probable
> *"Pourquoi cette machine est à risque ?"* → vibration anormalement élevée → vérifier les roulements

---

## Dashboard Streamlit & Conclusion

### 4 onglets décisionnels

| Onglet | Contenu |
|---|---|
| 📊 EDA | Distributions, corrélations, valeurs manquantes |
| 🤖 Modèles | Tableau comparatif, ROC, matrices de confusion |
| 🔮 Prédiction | Sliders capteurs → **🔴 RISQUE ÉLEVÉ (78.3%)** |
| 🔍 Interprétabilité | Feature Importance + SHAP |

```bash
streamlit run dashboard/app.py   →  http://localhost:8501
```

### Bilan

✅ 4 modèles comparés · XGBoost retenu · F1 = **0.898** · Recall = **0.955** · AUC = **0.996**
✅ Cross-validation : **0.9026 ± 0.0099** — stable, pas d'overfitting
✅ Pipeline anti-leakage · SHAP · Dashboard opérationnel

> **XGBoost détecte 19 pannes sur 20 avant leur survenue**

---

<!-- _class: title -->

# Merci — Questions ?

## Démonstration disponible

```bash
streamlit run dashboard/app.py
```

*Projet M2 Data Science — EFREI 2025-26*
*RNCP36739 — Bloc 4*
