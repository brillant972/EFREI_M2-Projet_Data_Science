"""
Évaluation des modèles - Maintenance Prédictive
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # backend non-interactif pour les scripts
import seaborn as sns
import shap
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    RocCurveDisplay, classification_report,
)
from sklearn.inspection import permutation_importance

from src.preprocessing import get_feature_names_out


# ── Métriques ─────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> tuple:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Modèle":     model_name,
        "Accuracy":   round(accuracy_score(y_test, y_pred), 4),
        "Precision":  round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":     round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score":   round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":    round(roc_auc_score(y_test, y_proba), 4),
    }
    return metrics, y_pred, y_proba


def compare_models(results: list) -> pd.DataFrame:
    df = pd.DataFrame(results).set_index("Modèle")
    return df


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, model_name: str, save_path: str = None):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No Failure", "Failure"],
        yticklabels=["No Failure", "Failure"],
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
    ax.set_ylabel("Réel")
    ax.set_xlabel("Prédit")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100)
    return fig


def plot_roc_curves(models: dict, X_test, y_test, save_path: str = None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("Courbes ROC — Comparaison des modèles")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100)
    return fig


def plot_metrics_comparison(comparison_df: pd.DataFrame, save_path: str = None):
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    df_plot = comparison_df[metrics_to_plot].reset_index().melt(
        id_vars="Modèle", var_name="Métrique", value_name="Score"
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df_plot, x="Métrique", y="Score", hue="Modèle", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparaison des performances par modèle")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100)
    return fig


# ── Feature Importance ────────────────────────────────────────────────────────

def plot_feature_importance(model, X_test, y_test, feature_names: list,
                             model_name: str, save_path: str = None):
    classifier = model.named_steps["classifier"]
    fig, ax = plt.subplots(figsize=(9, 6))

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        title = f"Feature Importance (native) — {model_name}"
    else:
        # Permutation importance pour les modèles sans importance native
        X_trans = model.named_steps["preprocessor"].transform(X_test)
        perm = permutation_importance(
            classifier, X_trans, y_test,
            n_repeats=10, random_state=42, scoring="f1"
        )
        importances = perm.importances_mean
        title = f"Permutation Importance — {model_name}"

    indices = np.argsort(importances)[::-1][:15]
    names   = [feature_names[i] for i in indices]
    vals    = importances[indices]

    sns.barplot(x=vals, y=names, ax=ax, palette="viridis")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100)
    return fig


# ── SHAP ──────────────────────────────────────────────────────────────────────

def compute_shap(model, X_test, feature_names: list, max_samples: int = 300):
    """
    Calcule les valeurs SHAP pour le modèle final.
    TreeExplainer pour RF/XGBoost, KernelExplainer (approximé) pour les autres.
    """
    preprocessor = model.named_steps["preprocessor"]
    classifier   = model.named_steps["classifier"]
    X_trans = preprocessor.transform(X_test)

    # Limiter l'échantillon pour la rapidité
    n = min(max_samples, X_trans.shape[0])
    X_sample = X_trans[:n]

    try:
        explainer   = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_sample)
        # Pour les classifieurs binaires, XGBoost retourne un tableau 2D
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    except Exception:
        background  = shap.sample(X_trans, 50, random_state=42)
        explainer   = shap.KernelExplainer(classifier.predict_proba, background)
        shap_values = explainer.shap_values(X_sample, nsamples=50)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    return shap_values, X_sample, feature_names


def plot_shap_summary(shap_values, X_sample, feature_names: list,
                       model_name: str, save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        show=False, plot_size=(10, 7),
    )
    plt.title(f"SHAP Summary — {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    return plt.gcf()


# ── Sauvegarde / Chargement ───────────────────────────────────────────────────

def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
