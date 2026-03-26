"""
Preprocessing pipeline - Maintenance Prédictive
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ── Chemins ──────────────────────────────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "predictive_maintenance_v3.csv")

# ── Cible ─────────────────────────────────────────────────────────────────────
TARGET = "failure_within_24h"

# Colonnes à supprimer : identifiants + data leakage
# rul_hours encode directement la réponse (rul < 24h → failure = 1)
# failure_type révèle le type de panne → leakage direct
# estimated_repair_cost corrélé à l'occurrence de panne
DROP_COLS = [
    "timestamp", "machine_id",
    "failure_type", "rul_hours", "estimated_repair_cost"
]

NUMERICAL_FEATURES = [
    "vibration_rms",
    "temperature_motor",
    "current_phase_avg",
    "pressure_level",
    "rpm",
    "hours_since_maintenance",
    "ambient_temp",
]

CATEGORICAL_FEATURES = ["machine_type", "operating_mode"]

ALL_INPUT_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def get_features_target(df: pd.DataFrame):
    X = df.drop(columns=DROP_COLS + [TARGET])
    y = df[TARGET]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Pipeline sklearn sans data leakage :
    - Numériques : imputation médiane + StandardScaler
    - Catégorielles : imputation mode + OneHotEncoder
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> list:
    """Retourne les noms de features après transformation (pour SHAP)."""
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERICAL_FEATURES + cat_names
