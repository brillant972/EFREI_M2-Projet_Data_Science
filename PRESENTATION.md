---
marp: true
theme: default
paginate: true
style: |
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

  section {
    font-family: 'Segoe UI', Calibri, Arial, sans-serif;
    font-size: 21px;
    padding: 44px 52px;
    color: #222;
    background: #ffffff;
  }
  h1 {
    font-size: 34px;
    color: #1a3a5c;
    border-bottom: 3px solid #e8701a;
    padding-bottom: 10px;
    margin-bottom: 18px;
    font-weight: 700;
  }
  h2 {
    font-size: 26px;
    color: #1a3a5c;
    border-bottom: 2px solid #d0dce8;
    padding-bottom: 6px;
    margin-bottom: 16px;
    font-weight: 700;
  }
  h3 {
    font-size: 20px;
    color: #2c6e9e;
    margin-top: 16px;
    margin-bottom: 8px;
    font-weight: 600;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 18px;
    margin: 10px 0;
  }
  th {
    background: #1a3a5c;
    color: #ffffff;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
  }
  td {
    padding: 7px 12px;
    border-bottom: 1px solid #dde3ea;
    vertical-align: top;
  }
  tr:nth-child(even) td { background: #f4f7fb; }
  code {
    background: #f0f3f7;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 17px;
    color: #c0392b;
    font-family: Consolas, monospace;
  }
  pre {
    background: #1e2a3a;
    color: #cdd9e5;
    padding: 14px 18px;
    border-radius: 6px;
    font-size: 16px;
    line-height: 1.55;
  }
  pre code { background: none; color: inherit; font-size: inherit; }
  p { margin: 6px 0 10px 0; line-height: 1.6; }
  ul { padding-left: 1.4em; margin: 4px 0; }
  li { margin-bottom: 5px; }
  blockquote {
    border-left: 4px solid #e8701a;
    padding: 8px 16px;
    background: #fff8f0;
    border-radius: 0 4px 4px 0;
    color: #555;
    margin: 10px 0;
  }
  section.cover {
    background: linear-gradient(145deg, #1a3a5c 0%, #2c6e9e 100%);
    color: #ffffff;
    justify-content: center;
  }
  section.cover h1 { color: #ffffff; border-bottom-color: #e8701a; font-size: 36px; }
  section.cover h2 { color: #f0c060; border: none; font-size: 22px; }
  section.cover p  { color: #cce0f5; font-size: 19px; }
  section.cover footer { color: #90b8d8; }
  img { max-width: 100%; height: auto; display: block; margin: 8px auto; }
---

<!-- _class: cover -->

# Système Intelligent Multi-Modèles
# pour la Maintenance Prédictive Industrielle

## Projet M2 Data Science — EFREI 2025-26

*Khalil DJAHEL / Bryan BONTRAIN*

*RNCP36739 — Bloc 4*

---

## Contexte et problème

| Maintenance corrective | Maintenance prédictive |
|---|---|
| Panne survenue → réparation d'urgence | Signal capteur dégradé → intervention planifiée |
| Arrêt non planifié, coûts élevés | Coût maîtrisé, zéro surprise |
| Risque opérateur | Sécurité préservée |

**Notre approche**

Classification binaire supervisée sur la variable `failure_within_24h`.

Les capteurs industriels (vibration, température, pression, RPM) génèrent en continu des signaux porteurs de patterns annonciateurs de panne. L'objectif est de les exploiter pour détecter une défaillance 24 heures à l'avance.

---

## Dataset et pipeline

**24 042 observations — 9 features retenues — 4 types de machines**

Trois colonnes supprimées pour éviter le **data leakage** :

| Colonne supprimée | Raison |
|---|---|
| `failure_type` | Révèle directement qu'une panne est en cours |
| `rul_hours` | Encode implicitement la cible (corrélation -0.25) |
| `estimated_repair_cost` | Calculé après la panne, indisponible en temps réel |

**Pipeline sklearn (anti-leakage)**

```
Split stratifié 80/20   →   ColumnTransformer ajusté sur TRAIN uniquement
  Numériques (7)  :  Median Imputer  +  StandardScaler
  Catégorielles (2):  Mode Imputer   +  OneHotEncoder
```

Toutes les statistiques de preprocessing sont calculées sur le train set, puis appliquées sur le test set sans contamination.

---

## Les 4 modèles

| Modèle | Principe | Gestion du déséquilibre (85/15) |
|---|---|---|
| Logistic Regression | Combinaison linéaire + sigmoïde — baseline | `class_weight='balanced'` |
| Random Forest | 200 arbres indépendants, vote (bagging) | `class_weight='balanced'` |
| XGBoost | 200 arbres séquentiels, chacun corrige le précédent (boosting) | `scale_pos_weight=5.75` |
| MLP Deep Learning | 128 → 64 → 32 neurones, ReLU, backpropagation | `early_stopping=True` |

**Métriques retenues**

Le Recall est prioritaire : un faux négatif (panne non détectée) est le cas le plus coûteux industriellement. Le F1-Score équilibre Recall et Precision. L'Accuracy seule est trompeuse sur un dataset déséquilibré.

---

## Résultats

| Modèle | Recall | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.895 | 0.747 | 0.959 |
| Random Forest | 0.916 | 0.887 | 0.993 |
| **XGBoost** | **0.955** | **0.898** | **0.996** |
| MLP Deep Learning | 0.853 | 0.850 | 0.984 |

![](saved_models/figures/metrics_comparison.png)

Cross-validation 5-fold XGBoost : **F1 = 0.9026 ± 0.0099** — modèle stable, pas d'overfitting.

---

## Interprétabilité

![](saved_models/figures/shap_summary.png)

**Top 5 features :** `vibration_rms` — `temperature_motor` — `hours_since_maintenance` — `rpm` — `pressure_level`

SHAP explique chaque prédiction individuelle. Une vibration élevée (valeur rouge, axe positif) augmente la probabilité de panne. Un entretien récent (valeur bleue, axe négatif) la réduit. Les décisions sont physiquement cohérentes et justifiables auprès d'un responsable maintenance.

---

## Dashboard et conclusion

**Dashboard Streamlit** — 4 onglets : EDA, comparaison des modèles, prédiction en temps réel, interprétabilité.

```
streamlit run dashboard/app.py
```

**Bilan**

XGBoost retenu : F1 = 0.898 — Recall = 0.955 — ROC-AUC = 0.996

Avec un Recall de 95.5 %, le système détecte 19 pannes sur 20 avant leur survenue, permettant de planifier des interventions préventives à moindre coût.

**Perspectives :** features temporelles (rolling mean 1 h / 6 h), ajustement du seuil de décision, monitoring de dérive, API REST FastAPI.
