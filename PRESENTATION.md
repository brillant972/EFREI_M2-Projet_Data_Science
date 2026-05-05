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
    font-size: 15px;
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
  img { max-width: 100%; max-height: 380px; height: auto; display: block; margin: 8px auto; }
---

<!-- _class: cover -->

# Système Intelligent Multi-Modèles
# pour la Maintenance Prédictive Industrielle

## Soutenance — M2 Data Engineering et IA — EFREI 2025-26

*Khalil DJAHEL / Bryan BONTRAIN*

*RNCP36739 — Bloc 4 — Formatrice : Sarah MALAEB*

---

## Contexte et problème métier

| Maintenance corrective | Maintenance prédictive |
|---|---|
| Panne survenue → réparation d'urgence | Signal capteur dégradé → intervention planifiée |
| Arrêt non planifié, coûts élevés | Coût maîtrisé, zéro surprise |
| Risque opérateur | Sécurité préservée |

> **"Un système intelligent multi-modèles permettant de détecter une panne industrielle 24 heures à l'avance, pour aider les responsables maintenance à planifier des interventions préventives à moindre coût."**

**Tâche :** Classification binaire supervisée — variable cible : `failure_within_24h`

---

## Analyse du besoin utilisateur

**Utilisateur cible :** responsable maintenance / ingénieur opérationnel (profil non technique)

| Scénario d'usage | Besoin | Réponse de la solution |
|---|---|---|
| Surveillance quotidienne | Quelles machines surveiller en priorité ? | Score de risque coloré par machine |
| Alerte imminente | Cette machine doit-elle être arrêtée ? | Probabilité + recommandation 24h |
| Bilan de performance | Peut-on faire confiance au système ? | Tableau comparatif des modèles |

**Contraintes :** interface non technique · résultat lisible en 5 secondes · prédiction en temps réel

**Valeur ajoutée :** détecter 19 pannes sur 20 avant leur survenue → réduire les arrêts non planifiés et leurs coûts d'urgence

---

## Méthodologie et organisation du projet

**Approche Agile / Kanban** — sprints hebdomadaires, revue des livrables en binôme

| Sprint | Contenu | Outils |
|---|---|---|
| 1 | Compréhension du sujet, EDA | Jupyter, Pandas, Matplotlib |
| 2 | Preprocessing, pipeline anti-leakage | sklearn, ColumnTransformer |
| 3 | Modélisation (4 algorithmes ML/DL) | XGBoost, sklearn, Keras/MLP |
| 4 | Évaluation, SHAP, interprétabilité | SHAP, Matplotlib, cross-validation |
| 5 | Dashboard Streamlit, soutenance | Streamlit, Marp |

**Répartition :** Khalil (pipeline, modèles ML/DL, SHAP) · Bryan (EDA, dashboard, rapport)

**Risques gérés :** déséquilibre 85/15 · data leakage · overfitting MLP · performance Deep Learning limitée

---

## Référentiel de données

**Source :** Kaggle — Industrial Machine Predictive Maintenance Dataset

| Attribut | Valeur |
|---|---|
| Observations | 24 042 |
| Variables retenues | 9 features + 1 cible |
| Types de machines | CNC, Pump, Compressor, Robotic Arm |
| Variable cible | `failure_within_24h` (0/1) |
| Déséquilibre | 85.2 % pas de panne · 14.8 % panne imminente |

**Features capteurs :** `vibration_rms` · `temperature_motor` · `current_phase_avg` · `pressure_level` · `rpm` · `hours_since_maintenance` · `ambient_temp` · `machine_type` · `operating_mode`

**3 colonnes supprimées (data leakage) :** `failure_type` · `rul_hours` · `estimated_repair_cost`

![h:160px](saved_models/figures/eda_target_distribution.png)

---

## EDA — Exploration et visualisation

![h:300px](saved_models/figures/eda_sensor_distributions.png)

Les machines en **pré-panne** (orange) présentent des valeurs systématiquement plus élevées pour `vibration_rms`, `temperature_motor` et `hours_since_maintenance` — cohérent avec la physique industrielle (usure, surchauffe, absence de maintenance).

---

## Pipeline Data Science anti-leakage

```
Données brutes (24 042 lignes · 15 colonnes)
  ↓  Suppression leakage + identifiants
  ↓  Split stratifié 80/20  →  Train: 19 233 | Test: 4 809
  ↓
  ColumnTransformer  [ajusté sur TRAIN uniquement]
    ├─ Numériques (7)    :  Median Imputer  →  StandardScaler
    └─ Catégorielles (2) :  Mode Imputer   →  OneHotEncoder
  ↓
  Entraînement  →  4 modèles (LR · RF · XGBoost · MLP)
  ↓
  Évaluation comparative (Recall · F1 · ROC-AUC)
  ↓
  Modèle retenu : XGBoost  →  SHAP  →  Dashboard Streamlit
```

Toutes les statistiques de preprocessing calculées **uniquement sur le train set**, jamais sur le test.

---

## Les 4 modèles

| Modèle | Principe | Gestion déséquilibre (85/15) |
|---|---|---|
| Logistic Regression | Combinaison linéaire + sigmoïde — baseline | `class_weight='balanced'` |
| Random Forest | 200 arbres indépendants, vote (bagging) | `class_weight='balanced'` |
| XGBoost | 200 arbres séquentiels, correction progressive (boosting) | `scale_pos_weight=5.75` |
| MLP Deep Learning | 128 → 64 → 32 neurones, ReLU, backpropagation | `early_stopping=True` |

**Métriques prioritaires :** Recall (FN = panne ratée → cas le plus coûteux industriellement) · F1-Score · ROC-AUC

---

## Résultats comparatifs

| Modèle | Recall | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.895 | 0.747 | 0.959 |
| Random Forest | 0.916 | 0.887 | 0.993 |
| **XGBoost** ✔ | **0.955** | **0.898** | **0.996** |
| MLP Deep Learning | 0.853 | 0.850 | 0.984 |

![h:240px](saved_models/figures/metrics_comparison.png)

Cross-validation 5-fold XGBoost : **F1 = 0.9026 ± 0.0099** — stable, pas d'overfitting

---

## Interprétabilité — SHAP

![h:310px](saved_models/figures/shap_summary.png)

**Top 5 :** `vibration_rms` · `temperature_motor` · `hours_since_maintenance` · `rpm` · `pressure_level`

Rouge = valeur élevée → **augmente le risque de panne** · Bleu = valeur faible → **réduit le risque**

---

## Dashboard décisionnel

**Streamlit** — 4 onglets orientés utilisateur métier

| Onglet | Valeur métier |
|---|---|
| EDA | Visualisation des signaux capteurs et corrélations |
| Modèles | Tableau comparatif, courbes ROC, matrices de confusion |
| Prédiction | Score de risque coloré + recommandation en temps réel |
| Interprétabilité | Feature importance + SHAP pour chaque alerte |

**Cas d'usage :** saisir les valeurs capteurs d'une machine → obtenir un score de risque (**ÉLEVÉ / FAIBLE**) avec probabilité et recommandation d'intervention dans les 24h

```
streamlit run dashboard/app.py   →   http://localhost:8501
```

---

## Limites et pistes d'amélioration

**Limites actuelles**

- **Dataset simulé** → performances probablement inférieures sur données industrielles réelles (bruit capteurs, variabilité inter-machines)
- **Absence de features temporelles** → chaque mesure est traitée indépendamment (pas de tendance d'évolution)
- **Seuil de décision fixe à 0.5** → devrait être ajusté selon le ratio coût panne / coût intervention
- **MLP limité** → Deep Learning efficace à partir de 100k+ observations ou séries temporelles longues

**Pistes d'amélioration**

- Rolling features (moyenne glissante 1h / 6h) pour capturer les tendances de dégradation
- Ajustement du seuil de décision selon le coût métier réel
- Monitoring de dérive des capteurs (data drift) en production
- Réentraînement périodique + déploiement cloud
- API REST FastAPI pour intégration SCADA/ERP (optionnel)

---

## Conclusion

**Ce que nous avons réalisé**

Pipeline Data Science complet : EDA → preprocessing anti-leakage → 4 modèles → évaluation comparative → interprétabilité SHAP → dashboard décisionnel

**Modèle retenu — XGBoost**

| Métrique | Valeur | Signification métier |
|---|---|---|
| Recall | **0.955** | 19 pannes sur 20 détectées avant survenue |
| F1-Score | **0.898** | Équilibre précision / détection |
| ROC-AUC | **0.996** | Quasi-parfaite discrimination des classes |

**Compétences acquises :** EDA · Preprocessing · ML supervisé · Deep Learning · Évaluation rigoureuse · Interprétabilité (SHAP) · Dashboard Streamlit · Gestion de projet Agile

> Passage d'un dataset brut à un outil décisionnel opérationnel, explicable et défendable dans un contexte professionnel.
