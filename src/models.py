"""
Définition des 4 modèles ML/DL - Maintenance Prédictive

Modèles :
  1. Logistic Regression  — baseline interprétable
  2. Random Forest        — capture non-linéarités, feature importance native
  3. XGBoost              — gradient boosting haute performance
  4. MLP (Deep Learning)  — réseau de neurones multicouche (128 → 64 → 32)
"""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from src.preprocessing import build_preprocessor

# Ratio d'imbalance ~20482/3560 ≈ 5.75
SCALE_POS_WEIGHT = 5.75


def build_models(random_state: int = 42) -> dict:
    """
    Retourne un dictionnaire {nom: Pipeline sklearn}.
    Chaque pipeline intègre le preprocessing + le classificateur
    pour éviter tout data leakage.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                random_state=random_state,
            )),
        ]),

        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            )),
        ]),

        "XGBoost": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=SCALE_POS_WEIGHT,
                eval_metric="logloss",
                random_state=random_state,
                verbosity=0,
            )),
        ]),

        # MLP = réseau de neurones profond (3 couches cachées) → Deep Learning
        "MLP (Deep Learning)": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,           # regularisation L2
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=random_state,
            )),
        ]),
    }

    return models
