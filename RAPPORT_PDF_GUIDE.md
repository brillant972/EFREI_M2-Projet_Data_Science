# GUIDE D'INSERTION DES IMAGES — Rapport PDF
## Maintenance Prédictive Industrielle — EFREI M2 2025-26

> **Mode d'emploi :** Ce fichier indique précisément où insérer chaque figure dans le rapport.
> Toutes les images se trouvent dans le dossier `saved_models/figures/`.
> Pour générer un PDF : ouvrir `RAPPORT_PROJET.md` dans Word / Google Docs / Pandoc,
> puis insérer les images aux emplacements indiqués ci-dessous.

---

## ORDRE D'INSERTION DES IMAGES DANS LE RAPPORT

---

### IMAGE 1 — À insérer dans la Section 3.1 (Distribution de la cible)

**Fichier :** Graphique à créer depuis le dashboard (onglet EDA → distribution target)
**Ou :** capture d'écran de l'onglet EDA du dashboard Streamlit

**Légende à utiliser :**
> *Figure 1 — Distribution de la variable cible `failure_within_24h` : 85.2% de non-pannes
> (20 482 obs.) vs 14.8% de pannes (3 560 obs.). Le déséquilibre est significatif (ratio 5.75:1).*

**Pourquoi cette image ?**
Montre visuellement le déséquilibre des classes — argument central pour justifier les métriques
choisies (Recall, F1) plutôt que l'Accuracy.

---

### IMAGE 2 — À insérer dans la Section 3.3 (Valeurs manquantes)

**Fichier :** Capture d'écran du graphique "Valeurs manquantes" dans l'onglet EDA du dashboard

**Légende à utiliser :**
> *Figure 2 — Pourcentage de valeurs manquantes par variable capteur. Entre 2.2% (rpm)
> et 4.2% (vibration_rms). Gérées par imputation médiane.*

---

### IMAGE 3 — À insérer dans la Section 3.4 (Distributions des capteurs)

**Fichier :** Capture d'écran du graphique de distribution capteur (onglet EDA → sélectionner vibration_rms)

**Légende à utiliser :**
> *Figure 3 — Distribution de `vibration_rms` selon le statut (panne / normal). Les machines
> en pré-panne présentent des valeurs systématiquement plus élevées.*

---

### IMAGE 4 — À insérer dans la Section 3.5 (Corrélations)

**Fichier :** Capture d'écran de la matrice de corrélation dans l'onglet EDA du dashboard

**Légende à utiliser :**
> *Figure 4 — Matrice de corrélation entre les features. Absence de multicolinéarité forte :
> toutes les variables apportent une information complémentaire.*

---

### IMAGE 5 — À insérer dans la Section 4.4 (Comment lire une matrice de confusion)

**Fichier :** `saved_models/figures/cm_XGBoost.png`

**Taille recommandée :** 60% de la largeur de la page

**Légende à utiliser :**
> *Figure 5 — Matrice de confusion du modèle XGBoost (test set, n=4 809).
> TN (haut-gauche) : non-pannes correctement identifiées.
> TP (bas-droite) : pannes correctement détectées.
> FN (bas-gauche) : pannes manquées — le cas le plus critique industriellement.
> FP (haut-droite) : fausses alertes.*

**Pourquoi cette image ?**
C'est la preuve visuelle que le modèle détecte effectivement les pannes. Le coin FN doit être
le plus petit possible.

---

### IMAGE 6 — À insérer dans la Section 6.1 (Tableau des performances) — juste après le tableau

**Fichier :** `saved_models/figures/metrics_comparison.png`

**Taille recommandée :** pleine largeur

**Légende à utiliser :**
> *Figure 6 — Comparaison des 5 métriques pour les 4 modèles (test set).
> XGBoost domine sur tous les critères sauf la Precision (ex-aequo avec MLP).
> La progression Logistic Regression → Random Forest → XGBoost illustre le gain
> apporté par chaque niveau de complexité.*

**Pourquoi cette image ?**
Synthèse visuelle de toute la comparaison des modèles. Un jury comprend immédiatement
la hiérarchie des modèles.

---

### IMAGE 7 — À insérer dans la Section 6.2 (Analyse des résultats) — après l'analyse XGBoost

**Fichier :** `saved_models/figures/roc_curves.png`

**Taille recommandée :** 70% de la largeur de page

**Légende à utiliser :**
> *Figure 7 — Courbes ROC des 4 modèles (test set).
> XGBoost atteint un AUC de 0.996, très proche du point idéal (coin supérieur gauche).
> La diagonale représenterait un modèle aléatoire (AUC = 0.5).
> Plus la courbe se rapproche du coin supérieur gauche, meilleur est le modèle.*

---

### IMAGE 8 — À insérer dans la Section 6.2 — après les 4 analyses de modèles

**4 matrices de confusion côte à côte**
**Fichiers :**
- `saved_models/figures/cm_Logistic_Regression.png`
- `saved_models/figures/cm_Random_Forest.png`
- `saved_models/figures/cm_XGBoost.png`
- `saved_models/figures/cm_MLP_Deep_Learning.png`

**Disposition :** 4 images en grille 2×2 (ou 4 en ligne)

**Légende à utiliser :**
> *Figure 8 — Matrices de confusion des 4 modèles. De gauche à droite : Logistic Regression,
> Random Forest, XGBoost, MLP. Le coin bas-gauche (FN = panne non détectée)
> diminue progressivement avec la complexité du modèle.*

**Pourquoi cette image ?**
Permet de comparer visuellement les erreurs de chaque modèle. Le jury voit immédiatement
que XGBoost a le moins de FN.

---

### IMAGE 9 — À insérer dans la Section 8.1 (Feature Importance)

**Fichier :** `saved_models/figures/feature_importance.png`

**Taille recommandée :** 80% de la largeur de page

**Légende à utiliser :**
> *Figure 9 — Importance des features pour le modèle XGBoost (gain moyen dans les arbres).
> Les 5 variables les plus importantes sont cohérentes avec la physique industrielle :
> vibration, température et usure accumulée dominent les prédictions.*

**Pourquoi cette image ?**
Prouve que le modèle utilise des signaux physiquement sensés — pas du bruit. C'est l'argument
clé pour convaincre un responsable technique ou un jury.

---

### IMAGE 10 — À insérer dans la Section 8.2 (SHAP)

**Fichier :** `saved_models/figures/shap_summary.png`

**Taille recommandée :** pleine largeur (image haute résolution, 111 KB)

**Légende à utiliser :**
> *Figure 10 — SHAP Summary Plot (beeswarm, 300 observations du test set).
> Chaque point représente une observation. La couleur indique la valeur de la feature
> (rouge = élevée, bleu = faible). La position sur l'axe X indique l'impact sur la prédiction
> (positif = augmente le risque de panne).
> Une vibration_rms élevée (rouge, axe X positif) augmente fortement la probabilité de panne.
> Un hours_since_maintenance faible (bleu, axe X négatif) réduit le risque.*

**Pourquoi cette image ?**
C'est la pièce maîtresse de l'interprétabilité. Elle montre que le modèle s'explique,
et que les décisions sont cohérentes avec la logique industrielle.

---

### IMAGE 11 — À insérer dans la Section 9 (Dashboard)

**Fichier :** Captures d'écran du dashboard Streamlit (4 onglets)

**Comment prendre ces captures :**
1. Lancer `streamlit run dashboard/app.py`
2. Onglet EDA → capture
3. Onglet Modèles → capture (tableau + ROC visible)
4. Onglet Prédiction → régler les sliders pour simuler une machine à risque élevé → capturer le résultat 🔴 ÉLEVÉ
5. Onglet Interprétabilité → capture du SHAP plot

**Disposition :** 2×2 ou 4 en colonne

**Légende à utiliser :**
> *Figure 11 — Dashboard Streamlit (4 onglets). De gauche à droite :
> (a) Analyse EDA — distributions et corrélations ;
> (b) Comparaison des modèles — tableau et courbes ROC ;
> (c) Prédiction en temps réel — résultat 🔴 RISQUE ÉLEVÉ (78.3%) ;
> (d) Interprétabilité — Feature Importance et SHAP.*

---

## RÉCAPITULATIF DES IMAGES PAR SECTION

| Section | Image | Fichier source |
|---|---|---|
| 3.1 Distribution cible | Fig. 1 | Capture dashboard EDA |
| 3.3 Valeurs manquantes | Fig. 2 | Capture dashboard EDA |
| 3.4 Distributions capteurs | Fig. 3 | Capture dashboard EDA |
| 3.5 Corrélations | Fig. 4 | Capture dashboard EDA |
| 4.4 Matrice de confusion lecture | Fig. 5 | `cm_XGBoost.png` |
| 6.1 Résultats comparatifs | Fig. 6 | `metrics_comparison.png` |
| 6.2 Courbes ROC | Fig. 7 | `roc_curves.png` |
| 6.2 Matrices (x4) | Fig. 8 | `cm_*.png` (×4) |
| 8.1 Feature Importance | Fig. 9 | `feature_importance.png` |
| 8.2 SHAP | Fig. 10 | `shap_summary.png` |
| 9. Dashboard | Fig. 11 | Captures d'écran live |

---

## INSTRUCTIONS POUR GÉNÉRER LE PDF

### Option 1 — Pandoc (recommandé, ligne de commande)
```bash
pandoc RAPPORT_PROJET.md -o RAPPORT_PROJET.pdf \
  --pdf-engine=xelatex \
  --variable mainfont="Arial" \
  --variable geometry:margin=2cm
```

### Option 2 — VS Code + extension Markdown PDF
1. Installer l'extension "Markdown PDF" dans VS Code
2. Ouvrir RAPPORT_PROJET.md
3. Clic droit → "Markdown PDF : Export (pdf)"

### Option 3 — Google Docs
1. Copier le contenu de RAPPORT_PROJET.md dans Google Docs
2. Insérer les images aux emplacements indiqués avec "Insertion > Image"
3. Fichier → Télécharger → PDF

### Option 4 — Word
1. Ouvrir RAPPORT_PROJET.md dans Word (il interprète le Markdown)
2. Insérer les images
3. Fichier → Enregistrer sous → PDF

---

## CONSEILS DE MISE EN PAGE POUR LE PDF

- **Police :** Arial ou Calibri 11pt pour le corps, 14pt pour les titres H2, 12pt pour H3
- **Marges :** 2 cm de chaque côté
- **Images :** centrer avec légende en italique dessous
- **Tableaux :** police 10pt, en-têtes en gras
- **Page de garde :** ajouter logo EFREI + nom du binôme + date
- **Numérotation :** activer la numérotation des pages
- **Table des matières :** générer automatiquement depuis les titres
- **Longueur cible :** 20-25 pages avec images

---

*Document de référence — Projet M2 Data Science EFREI 2025-26*
