# Support de Présentation — 10 minutes
## Maintenance Prédictive Industrielle
**Projet M2 Data Science — EFREI 2025-26**
*RNCP36739 — Bloc 4*

> **10 slides = 10 minutes**
> Budget temps indicatif noté sur chaque slide.
> Démo dashboard : avoir la page ouverte à l'avance sur l'onglet Prédiction.
> Commande de lancement : `streamlit run dashboard/app.py`

---

---
## SLIDE 1 — Titre *(~30 secondes)*

# Système Intelligent Multi-Modèles
# pour la Maintenance Prédictive Industrielle

**Projet M2 Data Science — EFREI 2025-26**

*RNCP36739 — Bloc 4 : Implémenter des méthodes d'IA*

> **IMAGE :** Logo EFREI + photo d'une machine industrielle (fond)

---

---
## SLIDE 2 — Problème et solution *(~1 minute)*

### Le problème

| Maintenance **corrective** | Maintenance **prédictive** |
|---|---|
| Panne → réparation d'urgence | Détection 24h avant → intervention planifiée |
| Arrêt non planifié coûteux | Intervention préventive moins chère |
| Risque de sécurité | Zéro surprise |

### Notre réponse
**Prédire `failure_within_24h` grâce aux données capteurs**
→ Classification binaire supervisée : panne dans 24h ? OUI / NON

> **IMAGE :** Schéma simple : capteurs → modèle ML → alerte

**Données :** 24 042 observations | 4 types de machines | 9 features capteurs

---

---
## SLIDE 3 — EDA et déséquilibre *(~1 minute)*

### Ce que les données nous ont appris

```
Distribution de la cible :
Pas de panne  ████████████████████████████████  85.2% (20 482)
Panne < 24h   █████                              14.8%  (3 560)
                                    Ratio = 5.75:1
```

**Conséquence directe :** un modèle qui dit "jamais de panne" aurait 85% d'Accuracy…
mais détecterait 0 panne. → **L'Accuracy ne sert à rien ici.**

### Métriques choisies
| Métrique | Formule | Pourquoi |
|---|---|---|
| **Recall** | TP / (TP + FN) | Prioritaire : minimiser les pannes ratées |
| **F1-Score** | 2×(P×R)/(P+R) | Compromis Precision / Recall |
| **ROC-AUC** | Aire sous courbe ROC | Performance globale seuil-indépendante |

> **IMAGE :** Diagramme circulaire 85/15 + formules métriques

---

---
## SLIDE 4 — Pipeline et anti-leakage *(~1 minute)*

### Architecture du pipeline

```
Dataset brut (24 042 lignes, 15 colonnes)
    │
    ▼  Suppression : timestamp, machine_id, failure_type, rul_hours, repair_cost
    │  → DATA LEAKAGE : ces colonnes révèlent la réponse
    ▼
Split STRATIFIÉ 80/20
    ┌──────────────┬─────────────┐
    │  TRAIN       │  TEST       │
    │  19 233 obs  │  4 809 obs  │
    └──────────────┴─────────────┘
    │
    ▼  ColumnTransformer (ajusté sur TRAIN → appliqué sur TEST)
       ├── 7 variables numériques : Median Imputer + StandardScaler
       └── 2 variables catégorielles : Mode Imputer + OneHotEncoder
```

**Règle d'or anti-leakage :** le test set ne doit JAMAIS influencer le preprocessing.
Toutes les statistiques (médiane, écart-type, catégories) sont calculées sur le train set uniquement.

---

---
## SLIDE 5 — Les 4 modèles *(~1 minute)*

| # | Modèle | Principe | Gestion déséquilibre |
|---|---|---|---|
| 1 | **Logistic Regression** | Score linéaire → sigmoïde → probabilité | `class_weight='balanced'` |
| 2 | **Random Forest** | 200 arbres indépendants → vote (bagging) | `class_weight='balanced'` |
| 3 | **XGBoost** | 200 arbres séquentiels → chaque arbre corrige le précédent (boosting) | `scale_pos_weight=5.75` |
| 4 | **MLP (Deep Learning)** | 128→64→32 neurones, ReLU, backprop | `early_stopping=True` |

**Progression de complexité :**
```
Simple ←──────────────────────────────────────────────────────→ Complexe
Linéaire     Ensembliste     Gradient Boosting     Réseau neuronal
```
Chaque niveau permet de mesurer le gain apporté par la complexité supplémentaire.

> **IMAGE :** Schéma comparatif des 4 architectures (bagging vs boosting vs MLP)

---

---
## SLIDE 6 — Résultats comparatifs *(~1.5 minutes)*

### Performances sur le test set (n = 4 809)

| Modèle | Recall ↑ | F1-Score ↑ | ROC-AUC ↑ |
|---|---|---|---|
| Logistic Regression | 0.895 | 0.747 | 0.959 |
| Random Forest | 0.916 | 0.887 | 0.993 |
| **XGBoost ✅** | **0.955** | **0.898** | **0.996** |
| MLP (Deep Learning) | 0.853 | 0.850 | 0.984 |

**Ce que dit le Recall de 0.955 en langage métier :**
> Sur 20 pannes réelles, XGBoost en détecte **19**. Une seule est manquée.

### Progression visible

```
Logistic Reg → Random Forest → XGBoost
F1 : 0.747   → 0.887         → 0.898
              +0.14              +0.011
```
Chaque complexification est justifiée par un gain mesurable.

> **IMAGES :** `metrics_comparison.png` + `roc_curves.png`

---

---
## SLIDE 7 — Pourquoi XGBoost ? Validation *(~1 minute)*

### Cross-validation 5-fold stratifiée (preuve de généralisation)

```
Fold 1 : F1 = 0.8930     ┐
Fold 2 : F1 = 0.8915     │  Chaque fold = données inédites
Fold 3 : F1 = 0.9105     │  pendant l'entraînement
Fold 4 : F1 = 0.9008     │
Fold 5 : F1 = 0.9169     ┘
─────────────────────────────
Moyenne : 0.9026 | Écart-type : 0.0099
```

**Écart-type de 0.01 → modèle stable, non surappris (pas d'overfitting)**

### Pourquoi pas le MLP ?
Deep Learning < XGBoost sur ce dataset car :
- 24k observations insuffisantes pour un réseau de neurones
- Features tabulaires structurées → domaine naturel des arbres
- XGBoost intègre une régularisation L1/L2 native

> **IMAGE :** `cm_XGBoost.png` (matrice de confusion du modèle retenu)

---

---
## SLIDE 8 — Interprétabilité (SHAP) *(~1 minute)*

### Feature Importance — Top 5
1. `vibration_rms` — dégradation mécanique (roulements, balourd)
2. `temperature_motor` — précurseur thermique
3. `hours_since_maintenance` — usure accumulée
4. `rpm` — stress mécanique
5. `pressure_level` — anomalie hydraulique

### SHAP Summary Plot — lecture rapide
```
Feature          SHAP faible (-)          SHAP élevé (+)
vibration_rms  ──🔵──────────────────── ──🔴────── → PANNE
temperature    ──🔵──────────────────────🔴──────   → PANNE
hours_maint    ──🔵──────────────────────🔴──────   → PANNE
```
🔴 Valeur élevée de la feature → augmente la probabilité de panne
🔵 Valeur faible → réduit le risque

**→ Un responsable peut comprendre POURQUOI la machine est à risque**

> **IMAGES :** `feature_importance.png` + `shap_summary.png`

---

---
## SLIDE 9 — Dashboard Streamlit *(~1 minute — démo live)*

### Architecture

| Onglet | Ce qu'il permet |
|---|---|
| 📊 EDA | Comprendre les données (distributions, corrélations) |
| 🤖 Modèles | Comparer les performances, voir les courbes ROC |
| 🔮 Prédiction | Saisir les capteurs → score de risque en temps réel |
| 🔍 Interprétabilité | Feature Importance + SHAP |

### Démo — Scénario prédiction

```
Responsable maintenance
    ↓ (saisit valeurs capteurs : vibration haute, température haute)
Dashboard
    ↓
🔴 RISQUE ÉLEVÉ — Probabilité : 78.3%
→ Intervention recommandée dans les 24h
```

```bash
streamlit run dashboard/app.py   # → http://localhost:8501
```

> **IMAGES :** 4 captures d'écran du dashboard (une par onglet)

---

---
## SLIDE 10 — Conclusion *(~30 secondes)*

### Ce que nous avons réalisé

| | Résultat |
|---|---|
| Pipeline ML sans data leakage | ✅ sklearn Pipelines + ColumnTransformer |
| 4 modèles comparés | ✅ LR · RF · XGBoost · MLP |
| Modèle retenu | ✅ XGBoost — F1=0.898 · Recall=0.955 · AUC=0.996 |
| Stabilité prouvée | ✅ CV 5-fold — 0.9026 ± 0.0099 |
| Interprétabilité | ✅ Feature Importance + SHAP |
| Dashboard opérationnel | ✅ Streamlit 4 onglets |

### Impact métier
> **XGBoost détecte 19 pannes sur 20 avant leur survenue**
> → Interventions préventives planifiées → moins d'arrêts non planifiés → moins de coûts d'urgence

---

# Merci — Questions ?

**Démonstration disponible :** `streamlit run dashboard/app.py`

---
*Projet M2 Data Science — EFREI 2025-26*
*RNCP36739 — Bloc 4 : Implémenter des méthodes d'IA*
