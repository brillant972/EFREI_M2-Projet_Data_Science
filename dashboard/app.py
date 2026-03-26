"""
Dashboard - Maintenance Prédictive Industrielle
Projet M2 Data Science - EFREI
Lancement : streamlit run dashboard/app.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import joblib
import streamlit as st

from src.preprocessing import (
    load_data, get_features_target,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, get_feature_names_out,
)
from src.evaluation import evaluate_model

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(BASE_DIR, "saved_models")
FIGURES_DIR = os.path.join(MODEL_DIR, "figures")

# ── Configuration Streamlit ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Maintenance Prédictive",
    page_icon="⚙️",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_dataset():
    df = load_data()
    X, y = get_features_target(df)
    return df, X, y


@st.cache_resource
def load_all_models():
    names = [
        "Logistic_Regression",
        "Random_Forest",
        "XGBoost",
        "MLP_Deep_Learning",
    ]
    models = {}
    for n in names:
        path = os.path.join(MODEL_DIR, f"{n}.pkl")
        if os.path.exists(path):
            models[n.replace("_", " ")] = joblib.load(path)
    return models


@st.cache_resource
def load_metadata():
    path = os.path.join(MODEL_DIR, "metadata.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_comparison():
    path = os.path.join(MODEL_DIR, "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()


def fig_to_streamlit(fig):
    st.pyplot(fig)
    plt.close(fig)


# ── Application principale ────────────────────────────────────────────────────

def main():
    # En-tête
    st.title("⚙️ Plateforme de Maintenance Prédictive Industrielle")
    st.markdown(
        "**Objectif :** Prédire les pannes machines dans les 24 heures "
        "à partir des données capteurs en temps réel."
    )

    # Chargement
    df, X, y = load_dataset()
    models   = load_all_models()
    meta     = load_metadata()
    comp_df  = load_comparison()

    if not models:
        st.error("Aucun modèle trouvé. Lancez d'abord `python main.py`.")
        return

    # KPIs en haut
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total observations", f"{len(df):,}")
    col2.metric("Machines surveillées", df["machine_id"].nunique())
    col3.metric("Taux de panne (24h)", f"{y.mean()*100:.1f}%")
    if meta:
        col4.metric(
            "Meilleur F1 (CV)",
            f"{meta.get('cv_f1_mean', 0):.4f}",
            f"±{meta.get('cv_f1_std', 0):.4f}",
        )

    st.markdown("---")

    # ── Onglets ────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Analyse EDA",
        "🤖 Comparaison des modèles",
        "🔮 Prédiction en temps réel",
        "🔍 Interprétabilité",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 1 : EDA
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.header("Analyse Exploratoire des Données")

        col1, col2 = st.columns(2)

        # Distribution de la cible
        with col1:
            st.subheader("Distribution de la variable cible")
            fig, ax = plt.subplots()
            counts = y.value_counts()
            ax.bar(["Pas de panne (0)", "Panne < 24h (1)"],
                   counts.values,
                   color=["#2196F3", "#F44336"])
            ax.set_ylabel("Nombre d'observations")
            for i, v in enumerate(counts.values):
                ax.text(i, v + 100, f"{v:,} ({v/len(y)*100:.1f}%)",
                        ha="center", fontsize=10)
            ax.set_title("Déséquilibre des classes")
            fig_to_streamlit(fig)

        # Types de machine
        with col2:
            st.subheader("Répartition par type de machine")
            fig, ax = plt.subplots()
            mt = df["machine_type"].value_counts()
            ax.pie(mt.values, labels=mt.index, autopct="%1.1f%%",
                   startangle=90, colors=sns.color_palette("Set2"))
            ax.set_title("Types de machines")
            fig_to_streamlit(fig)

        # Distributions des capteurs
        st.subheader("Distribution des capteurs par statut de panne")
        sensor = st.selectbox(
            "Sélectionner un capteur :", NUMERICAL_FEATURES, key="eda_sensor"
        )
        fig, ax = plt.subplots(figsize=(9, 4))
        for label, color in [(0, "#2196F3"), (1, "#F44336")]:
            subset = df[df["failure_within_24h"] == label][sensor].dropna()
            ax.hist(subset, bins=50, alpha=0.6, color=color,
                    label=f"{'Panne' if label else 'Normal'} (n={len(subset):,})")
        ax.set_xlabel(sensor)
        ax.set_ylabel("Fréquence")
        ax.set_title(f"Distribution de {sensor} par statut")
        ax.legend()
        fig_to_streamlit(fig)

        # Matrice de corrélation
        st.subheader("Matrice de corrélation")
        numeric_cols = NUMERICAL_FEATURES + ["failure_within_24h"]
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, mask=mask, ax=ax, linewidths=0.5)
        ax.set_title("Corrélations entre variables")
        fig_to_streamlit(fig)

        # Valeurs manquantes
        st.subheader("Valeurs manquantes")
        missing = df[NUMERICAL_FEATURES].isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            missing.plot(kind="bar", ax=ax, color="#FF9800")
            ax.set_title("Nombre de valeurs manquantes par capteur")
            ax.set_ylabel("Nombre de NaN")
            ax.set_xlabel("")
            plt.xticks(rotation=30)
            fig_to_streamlit(fig)
        else:
            st.success("Aucune valeur manquante.")

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 2 : Comparaison des modèles
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.header("Comparaison des Modèles")

        if comp_df.empty:
            st.warning("Lancez d'abord `python main.py` pour entraîner les modèles.")
        else:
            st.subheader("Tableau des performances (ensemble de test)")

            # Mise en couleur du meilleur score par métrique
            def highlight_max(s):
                is_max = s == s.max()
                return ["background-color: #C8E6C9" if v else "" for v in is_max]

            styled = comp_df.style.apply(highlight_max, axis=0).format("{:.4f}")
            st.dataframe(styled, use_container_width=True)

            # Graphique comparaison
            st.subheader("Visualisation des métriques")
            metrics_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
            available = [c for c in metrics_cols if c in comp_df.columns]
            df_plot = comp_df[available].reset_index().melt(
                id_vars="Modèle", var_name="Métrique", value_name="Score"
            )
            fig, ax = plt.subplots(figsize=(11, 5))
            sns.barplot(data=df_plot, x="Métrique", y="Score", hue="Modèle",
                        ax=ax, palette="Set2")
            ax.set_ylim(0, 1.08)
            ax.set_title("Performances par modèle et par métrique")
            ax.legend(loc="lower right", fontsize=9)
            for p in ax.patches:
                if p.get_height() > 0:
                    ax.annotate(f"{p.get_height():.3f}",
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha="center", va="bottom", fontsize=7)
            fig_to_streamlit(fig)

            # Courbes ROC
            st.subheader("Courbes ROC")
            roc_path = os.path.join(FIGURES_DIR, "roc_curves.png")
            if os.path.exists(roc_path):
                st.image(roc_path, use_container_width=True)
            else:
                st.info("Figures non générées. Relancez main.py.")

            # Matrices de confusion
            st.subheader("Matrices de confusion")
            col_names = list(models.keys())
            cols = st.columns(min(len(col_names), 2))
            for i, name in enumerate(col_names):
                safe = name.replace(" ", "_").replace("(", "").replace(")", "")
                cm_path = os.path.join(FIGURES_DIR, f"cm_{safe}.png")
                if os.path.exists(cm_path):
                    with cols[i % 2]:
                        st.image(cm_path, caption=name, use_container_width=True)

            # Analyse critique
            if meta:
                best = meta.get("best_model_name", "")
                st.subheader("Analyse et justification du modèle retenu")
                st.info(
                    f"**Modèle retenu : {best}**\n\n"
                    f"- **F1-Score (CV 5-fold)** : {meta.get('cv_f1_mean', 0):.4f} "
                    f"± {meta.get('cv_f1_std', 0):.4f}\n"
                    f"- Choix justifié par le meilleur compromis Recall/Precision "
                    f"dans un contexte où les faux négatifs (pannes non détectées) "
                    f"ont un coût opérationnel élevé.\n"
                    f"- La cross-validation confirme la stabilité du modèle "
                    f"(faible écart-type)."
                )

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 3 : Prédiction en temps réel
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.header("🔮 Simulation de Prédiction en Temps Réel")
        st.markdown(
            "Simulez les données capteurs d'une machine et obtenez "
            "une prédiction de risque de panne dans les 24 heures."
        )

        col_form, col_result = st.columns([2, 1])

        with col_form:
            st.subheader("Paramètres machine")

            machine_type = st.selectbox(
                "Type de machine", ["CNC", "Pump", "Compressor", "Robotic Arm"]
            )
            operating_mode = st.selectbox(
                "Mode de fonctionnement", ["idle", "normal", "peak"]
            )

            col_a, col_b = st.columns(2)
            with col_a:
                vibration_rms = st.slider(
                    "Vibration RMS", 0.0, 10.0,
                    float(df["vibration_rms"].median()), step=0.1
                )
                temperature_motor = st.slider(
                    "Température moteur (°C)", 20.0, 120.0,
                    float(df["temperature_motor"].median()), step=0.5
                )
                current_phase_avg = st.slider(
                    "Courant moyen (A)", 0.0, 30.0,
                    float(df["current_phase_avg"].median()), step=0.1
                )
                pressure_level = st.slider(
                    "Pression (bar)", 0.0, 100.0,
                    float(df["pressure_level"].median()), step=0.5
                )
            with col_b:
                rpm = st.slider(
                    "Vitesse rotation (RPM)", 0.0, 3000.0,
                    float(df["rpm"].median()), step=10.0
                )
                hours_since_maintenance = st.slider(
                    "Heures depuis maintenance", 0.0, 2000.0,
                    float(df["hours_since_maintenance"].median()), step=10.0
                )
                ambient_temp = st.slider(
                    "Température ambiante (°C)", -10.0, 50.0,
                    float(df["ambient_temp"].median()), step=0.5
                )

            model_choice = st.selectbox(
                "Modèle à utiliser", list(models.keys())
            )
            predict_btn = st.button("🔍 Prédire", type="primary")

        with col_result:
            st.subheader("Résultat")
            if predict_btn:
                input_data = pd.DataFrame([{
                    "vibration_rms":          vibration_rms,
                    "temperature_motor":       temperature_motor,
                    "current_phase_avg":       current_phase_avg,
                    "pressure_level":          pressure_level,
                    "rpm":                     rpm,
                    "hours_since_maintenance": hours_since_maintenance,
                    "ambient_temp":            ambient_temp,
                    "machine_type":            machine_type,
                    "operating_mode":          operating_mode,
                }])

                selected_model = models[model_choice]
                proba = selected_model.predict_proba(input_data)[0][1]
                pred  = selected_model.predict(input_data)[0]

                # Jauge de risque
                if proba < 0.3:
                    color = "#4CAF50"
                    risk_label = "🟢 FAIBLE"
                elif proba < 0.6:
                    color = "#FF9800"
                    risk_label = "🟡 MODÉRÉ"
                else:
                    color = "#F44336"
                    risk_label = "🔴 ÉLEVÉ"

                st.markdown(
                    f"""
                    <div style="background:{color};padding:20px;border-radius:10px;
                                text-align:center;color:white;">
                        <h2>Risque de panne</h2>
                        <h1>{risk_label}</h1>
                        <h3>Probabilité : {proba*100:.1f}%</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("---")
                st.write(f"**Décision :** {'⚠️ Intervention recommandée' if pred else '✅ Machine opérationnelle'}")
                st.write(f"**Modèle utilisé :** {model_choice}")
                st.write(f"**Probabilité de panne :** {proba:.4f}")

                # Gauge visuelle
                fig, ax = plt.subplots(figsize=(5, 1))
                ax.barh(0, proba, color=color, height=0.5)
                ax.barh(0, 1 - proba, left=proba,
                        color="#E0E0E0", height=0.5)
                ax.set_xlim(0, 1)
                ax.set_yticks([])
                ax.set_xticks([0, 0.3, 0.6, 1])
                ax.set_xticklabels(["0%", "30%", "60%", "100%"])
                ax.set_title("Score de risque")
                fig_to_streamlit(fig)
            else:
                st.info("Renseignez les paramètres et cliquez sur **Prédire**.")

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 4 : Interprétabilité
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.header("Interprétabilité des Modèles")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Feature Importance")
            fi_path = os.path.join(FIGURES_DIR, "feature_importance.png")
            if os.path.exists(fi_path):
                st.image(fi_path, use_container_width=True)
                if meta:
                    st.caption(
                        f"Feature importance pour le modèle : "
                        f"{meta.get('best_model_name', '')}"
                    )
            else:
                st.info("Lancez main.py pour générer les figures.")

        with col2:
            st.subheader("SHAP — Importance globale des variables")
            shap_path = os.path.join(FIGURES_DIR, "shap_summary.png")
            if os.path.exists(shap_path):
                st.image(shap_path, use_container_width=True)
                st.caption(
                    "**SHAP (SHapley Additive exPlanations)** : chaque point "
                    "représente une observation. La couleur indique la valeur "
                    "de la feature (rouge = élevée, bleu = faible). "
                    "La position sur l'axe X indique l'impact sur la prédiction."
                )
            else:
                st.info("Lancez main.py pour générer les valeurs SHAP.")

        st.subheader("Interprétation métier")
        st.markdown(
            """
            | Variable | Impact attendu |
            |---|---|
            | `vibration_rms` | Vibrations élevées → risque de panne mécanique accru |
            | `temperature_motor` | Surchauffe → risque de défaillance thermique |
            | `hours_since_maintenance` | Plus d'heures sans maintenance → usure accumulée |
            | `rpm` | Régimes excessifs → stress mécanique |
            | `pressure_level` | Pression anormale → risque hydraulique |
            | `operating_mode` | Mode *peak* → sollicitation maximale |
            """
        )

    # Footer
    st.markdown("---")
    st.caption(
        "Projet M2 Data Science — EFREI | Maintenance Prédictive Industrielle "
        "| RNCP36739 Bloc 4"
    )


if __name__ == "__main__":
    main()
