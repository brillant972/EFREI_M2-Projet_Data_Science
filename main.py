"""
Pipeline principal - Maintenance Prédictive Industrielle
Projet M2 Data Science - EFREI

Exécution : python main.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from src.preprocessing import load_data, get_features_target, get_feature_names_out
from src.models import build_models
from src.evaluation import (
    evaluate_model, compare_models,
    plot_confusion_matrix, plot_roc_curves, plot_metrics_comparison,
    plot_feature_importance, compute_shap, plot_shap_summary,
    save_model,
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "saved_models")
FIGURES_DIR = os.path.join(BASE_DIR, "saved_models", "figures")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    print("=" * 65)
    print("  MAINTENANCE PRÉDICTIVE — PIPELINE ML/DL")
    print("=" * 65)

    # ── 1. Chargement des données ─────────────────────────────────────────────
    print("\n[1/6] Chargement des données...")
    df = load_data()
    X, y = get_features_target(df)

    print(f"  Dataset : {X.shape[0]:,} lignes | {X.shape[1]} features")
    print(f"  Cible   : {y.value_counts().to_dict()}  "
          f"(déséquilibre {y.mean()*100:.1f}% positifs)")

    # ── 2. Split stratifié 80/20 ──────────────────────────────────────────────
    print("\n[2/6] Split train/test (80/20, stratifié)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train : {len(X_train):,} | Test : {len(X_test):,}")

    # ── 3. Entraînement des 4 modèles ─────────────────────────────────────────
    print("\n[3/6] Entraînement des modèles...")
    models_dict = build_models(random_state=42)
    trained_models = {}
    results = []

    for name, pipeline in models_dict.items():
        print(f"  > {name:<30}", end=" ", flush=True)
        pipeline.fit(X_train, y_train)
        metrics, y_pred, y_proba = evaluate_model(pipeline, X_test, y_test, name)
        results.append(metrics)
        trained_models[name] = {"pipeline": pipeline, "y_pred": y_pred, "y_proba": y_proba}
        print(f"F1={metrics['F1-Score']:.4f}  ROC-AUC={metrics['ROC-AUC']:.4f}  "
              f"Recall={metrics['Recall']:.4f}")

    # ── 4. Comparaison et sélection du meilleur modèle ────────────────────────
    print("\n[4/6] Comparaison des modèles :")
    comparison_df = compare_models(results)
    print(comparison_df.to_string())
    comparison_df.to_csv(os.path.join(MODEL_DIR, "model_comparison.csv"))

    best_name = comparison_df["F1-Score"].idxmax()
    print(f"\n  >> Meilleur modele (F1) : {best_name}")

    # ── 5. Sauvegarde des modèles et des figures ──────────────────────────────
    print("\n[5/6] Sauvegarde...")

    # Sauvegarde de tous les modèles
    for name, data in trained_models.items():
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        save_model(data["pipeline"], os.path.join(MODEL_DIR, f"{safe}.pkl"))

    # Meilleur modèle séparé
    save_model(trained_models[best_name]["pipeline"],
               os.path.join(MODEL_DIR, "best_model.pkl"))

    # Figures
    pipelines_only = {n: d["pipeline"] for n, d in trained_models.items()}

    fig_roc = plot_roc_curves(pipelines_only, X_test, y_test)
    fig_roc.savefig(os.path.join(FIGURES_DIR, "roc_curves.png"), dpi=100)
    plt.close(fig_roc)

    fig_metrics = plot_metrics_comparison(comparison_df)
    fig_metrics.savefig(os.path.join(FIGURES_DIR, "metrics_comparison.png"), dpi=100)
    plt.close(fig_metrics)

    for name, data in trained_models.items():
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        fig_cm = plot_confusion_matrix(y_test, data["y_pred"], name)
        fig_cm.savefig(os.path.join(FIGURES_DIR, f"cm_{safe}.png"), dpi=100)
        plt.close(fig_cm)

    # ── 6. Cross-validation + SHAP sur le meilleur modèle ────────────────────
    print("\n[6/6] Cross-validation (StratifiedKFold 5-fold) + SHAP...")
    best_pipeline = trained_models[best_name]["pipeline"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        best_pipeline, X, y, cv=cv, scoring="f1", n_jobs=-1
    )
    print(f"  CV F1  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Scores : {[round(s,4) for s in cv_scores]}")

    # Feature importance
    feature_names = get_feature_names_out(
        best_pipeline.named_steps["preprocessor"]
    )
    fig_fi = plot_feature_importance(
        best_pipeline, X_test, y_test, feature_names, best_name
    )
    fig_fi.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=100)
    plt.close(fig_fi)

    # SHAP
    print("  Calcul des valeurs SHAP...")
    shap_vals, X_sample, feat_names = compute_shap(
        best_pipeline, X_test, feature_names, max_samples=300
    )
    np.save(os.path.join(MODEL_DIR, "shap_values.npy"), shap_vals)
    np.save(os.path.join(MODEL_DIR, "shap_X_sample.npy"), X_sample)

    fig_shap = plot_shap_summary(shap_vals, X_sample, feat_names, best_name)
    fig_shap.savefig(os.path.join(FIGURES_DIR, "shap_summary.png"),
                     dpi=100, bbox_inches="tight")
    plt.close("all")

    # Sauvegarde des métadonnées pour le dashboard
    import json
    metadata = {
        "best_model_name": best_name,
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std":  float(cv_scores.std()),
        "feature_names": feat_names,
        "model_names": list(trained_models.keys()),
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print("  Pipeline terminé avec succès!")
    print(f"  Modèles et figures sauvegardés dans : {MODEL_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
