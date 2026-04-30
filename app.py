"""
app.py — Credit Risk Scoring | Société Générale Style
Application Streamlit de scoring de risque crédit bancaire
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os

# ──────────────────────────────────────────────────────────────
# CONFIG PAGE
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Risk Scoring | SG Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# STYLES CSS
# ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #E30613 0%, #8B0000 100%);
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        color: white;
    }
    .main-header h1 {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 4px 0 0;
        opacity: 0.85;
        font-size: 0.9rem;
        font-weight: 300;
    }

    /* Score card */
    .score-card {
        background: #0a0a0a;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        border: 1px solid #222;
    }
    .score-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 3.5rem;
        font-weight: 500;
        line-height: 1;
    }
    .score-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.6;
        margin-top: 8px;
    }

    /* Métriques KPI */
    .kpi-box {
        background: #f8f9fa;
        border-left: 4px solid #E30613;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .kpi-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #666;
        font-weight: 500;
    }
    .kpi-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 500;
        color: #0a0a0a;
        line-height: 1.2;
    }

    /* Badges de risque */
    .badge-low    { background:#e6f4ea; color:#1e7e34; border-radius:20px; padding:6px 16px; font-weight:600; font-size:0.85rem; }
    .badge-medium { background:#fff3cd; color:#856404; border-radius:20px; padding:6px 16px; font-weight:600; font-size:0.85rem; }
    .badge-high   { background:#fde8e8; color:#c0392b; border-radius:20px; padding:6px 16px; font-weight:600; font-size:0.85rem; }
    .badge-critical { background:#c0392b; color:white; border-radius:20px; padding:6px 16px; font-weight:600; font-size:0.85rem; }

    /* Sidebar */
    .css-1d391kg { background-color: #f5f5f5; }
    section[data-testid="stSidebar"] { background: #1a1a2e; }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label { color: #ccc !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: white !important; }

    /* Divider rouge SG */
    .sg-divider {
        height: 3px;
        background: linear-gradient(90deg, #E30613, transparent);
        border: none;
        margin: 20px 0;
    }

    /* Info boxes */
    .info-box {
        background: #fff8f8;
        border: 1px solid #f5c6c6;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    /* Cacher éléments Streamlit par défaut */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Charge le modèle ML depuis le dossier models/"""
    try:
        model   = joblib.load("models/model.pkl")
        scaler  = joblib.load("models/scaler.pkl")
        features = joblib.load("models/features.pkl")
        stats   = joblib.load("models/stats.pkl")
        return model, scaler, features, stats, True
    except FileNotFoundError:
        return None, None, None, None, False


@st.cache_data
def load_reference_data():
    try:
        return pd.read_parquet("data/reference_data.parquet")
    except:
        return None


# ──────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ──────────────────────────────────────────────────────────────

def get_risk_category(proba: float, threshold: float = 0.35):
    if proba < 0.15:
        return "Faible", "badge-low", "#27ae60"
    elif proba < threshold:
        return "Modéré", "badge-medium", "#f39c12"
    elif proba < 0.60:
        return "Élevé", "badge-high", "#e74c3c"
    else:
        return "Critique", "badge-critical", "#c0392b"


def score_to_display(proba: float) -> int:
    """Convertit la probabilité en score crédit (style FICO : 300–850)"""
    return int(850 - proba * 550)


def build_input_features(inputs: dict, feature_cols: list) -> pd.DataFrame:
    """Construit le vecteur de features à partir des inputs utilisateur"""
    d = dict(inputs)
    d["total_late_payments"] = (
        d["nb_late_30_59"] + d["nb_late_60_89"] + d["nb_times_90_days_late"]
    )
    d["debt_to_income"]  = d["debt_ratio"] * d["monthly_income"]
    d["credit_load"]     = d["nb_open_credit_lines"] + d["nb_real_estate_loans"]
    d["is_young"]        = int(d["age"] < 30)
    d["high_utilization"] = int(d["revolving_utilization"] > 0.7)
    return pd.DataFrame([d])[feature_cols]


# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🏦 Credit Risk Scoring Platform</h1>
    <p>Analyse prédictive du risque de défaut client · Modèle Gradient Boosting · Dataset bancaire réel (150K clients)</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# SIDEBAR — SAISIE CLIENT
# ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📋 Profil client")
    st.markdown("---")

    st.markdown("**Informations personnelles**")
    age = st.slider("Âge", 18, 80, 35)
    nb_dependents = st.slider("Nombre de personnes à charge", 0, 10, 1)

    st.markdown("---")
    st.markdown("**Situation financière**")
    monthly_income = st.number_input(
        "Revenu mensuel (€)", min_value=500, max_value=30000,
        value=3500, step=100
    )
    debt_ratio = st.slider(
        "Ratio d'endettement", 0.0, 5.0, 0.35,
        help="Dettes mensuelles / Revenu mensuel"
    )
    revolving_utilization = st.slider(
        "Utilisation du crédit renouvelable", 0.0, 1.5, 0.30,
        help="Montant utilisé / Limite de crédit"
    )

    st.markdown("---")
    st.markdown("**Historique de crédit**")
    nb_open_credit_lines = st.slider("Lignes de crédit ouvertes", 0, 20, 5)
    nb_real_estate_loans = st.slider("Prêts immobiliers", 0, 10, 1)
    nb_late_30_59 = st.slider("Retards 30-59 jours (2 ans)", 0, 15, 0)
    nb_late_60_89 = st.slider("Retards 60-89 jours (2 ans)", 0, 10, 0)
    nb_times_90_days_late = st.slider("Retards 90+ jours", 0, 20, 0)

    st.markdown("---")
    analyze_btn = st.button("🔍 Analyser le profil", use_container_width=True, type="primary")

# Données du client
client_inputs = {
    "revolving_utilization": revolving_utilization,
    "age": age,
    "nb_late_30_59": nb_late_30_59,
    "debt_ratio": debt_ratio,
    "monthly_income": monthly_income,
    "nb_open_credit_lines": nb_open_credit_lines,
    "nb_times_90_days_late": nb_times_90_days_late,
    "nb_real_estate_loans": nb_real_estate_loans,
    "nb_late_60_89": nb_late_60_89,
    "nb_dependents": nb_dependents,
}

# ──────────────────────────────────────────────────────────────
# CHARGEMENT MODÈLE
# ──────────────────────────────────────────────────────────────

model, scaler, feature_cols, model_stats, model_loaded = load_model()
df_ref = load_reference_data()

if not model_loaded:
    st.warning("""
    ⚠️ **Modèle non trouvé**

    Pour démarrer l'application :
    1. Télécharge le dataset depuis [Kaggle](https://www.kaggle.com/competitions/GiveMeSomeCredit/data) → `cs-training.csv`
    2. Place-le dans le dossier `data/`
    3. Lance : `python train_model.py`
    4. Relance l'app : `streamlit run app.py`
    """)
    st.stop()

# ──────────────────────────────────────────────────────────────
# SCORING EN TEMPS RÉEL
# ──────────────────────────────────────────────────────────────

X_input = build_input_features(client_inputs, feature_cols)
X_scaled = scaler.transform(X_input)
proba_default = model.predict_proba(X_scaled)[0][1]
credit_score  = score_to_display(proba_default)
risk_label, badge_class, risk_color = get_risk_category(proba_default, model_stats["threshold"])

# ──────────────────────────────────────────────────────────────
# ONGLETS PRINCIPAUX
# ──────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Score & Décision",
    "📈 Analyse du profil",
    "🔬 Facteurs de risque",
    "📉 Performance modèle",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — SCORE ET DÉCISION
# ══════════════════════════════════════════════════════════════

with tab1:
    col_score, col_decision = st.columns([1, 2])

    with col_score:
        # Jauge score crédit
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Score Crédit", "font": {"size": 16, "family": "IBM Plex Sans"}},
            gauge={
                "axis": {"range": [300, 850], "tickwidth": 1, "tickcolor": "#333"},
                "bar": {"color": risk_color, "thickness": 0.25},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [300, 520], "color": "#fde8e8"},
                    {"range": [520, 630], "color": "#fff3cd"},
                    {"range": [630, 720], "color": "#e6f4ea"},
                    {"range": [720, 850], "color": "#d4edda"},
                ],
                "threshold": {
                    "line": {"color": "#E30613", "width": 3},
                    "thickness": 0.75,
                    "value": credit_score,
                },
            },
            number={"font": {"size": 48, "family": "IBM Plex Mono"}},
        ))
        fig_gauge.update_layout(
            height=280, margin=dict(t=40, b=0, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Probabilité de défaut
        st.markdown(f"""
        <div style="text-align:center; margin-top:-10px">
            <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:2px;color:#888;margin-bottom:6px">
                Probabilité de défaut
            </div>
            <div style="font-family:'IBM Plex Mono';font-size:2rem;font-weight:500;color:{risk_color}">
                {proba_default:.1%}
            </div>
            <div style="margin-top:10px">
                <span class="{badge_class}">Risque {risk_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_decision:
        st.markdown("### Décision de crédit")
        st.markdown('<div class="sg-divider"></div>', unsafe_allow_html=True)

        # KPIs
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        with kpi_col1:
            threshold = model_stats["threshold"]
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">Score</div>
                <div class="kpi-value">{credit_score}</div>
            </div>""", unsafe_allow_html=True)
        with kpi_col2:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">Probabilité défaut</div>
                <div class="kpi-value">{proba_default:.1%}</div>
            </div>""", unsafe_allow_html=True)
        with kpi_col3:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-title">Seuil d'alerte</div>
                <div class="kpi-value">{threshold:.0%}</div>
            </div>""", unsafe_allow_html=True)

        # Recommandation
        st.markdown("#### Recommandation")
        if proba_default < 0.15:
            st.success("✅ **Accord de crédit recommandé** — Profil à faible risque. Conditions standards applicables.")
        elif proba_default < model_stats["threshold"]:
            st.info("ℹ️ **Accord sous conditions** — Risque modéré détecté. Justificatifs supplémentaires conseillés.")
        elif proba_default < 0.60:
            st.warning("⚠️ **Dossier à examiner en comité** — Risque élevé. Analyse approfondie requise.")
        else:
            st.error("🚫 **Refus recommandé** — Risque critique de défaut. Profil non éligible aux conditions actuelles.")

        # Facteurs clés du profil
        st.markdown("#### Facteurs clés identifiés")
        total_lates = nb_late_30_59 + nb_late_60_89 + nb_times_90_days_late
        factors = []
        if revolving_utilization > 0.7:
            factors.append(f"🔴 Utilisation crédit élevée ({revolving_utilization:.0%})")
        if total_lates > 0:
            factors.append(f"🔴 {total_lates} retard(s) de paiement détecté(s)")
        if debt_ratio > 0.5:
            factors.append(f"🟡 Ratio d'endettement élevé ({debt_ratio:.2f})")
        if age < 25:
            factors.append("🟡 Profil jeune — historique de crédit court")
        if revolving_utilization <= 0.4 and total_lates == 0:
            factors.append("🟢 Utilisation crédit maîtrisée")
        if debt_ratio <= 0.35:
            factors.append("🟢 Ratio d'endettement sain")
        if nb_open_credit_lines >= 3:
            factors.append("🟢 Diversification du crédit")

        for f in factors:
            st.markdown(f"- {f}")

# ══════════════════════════════════════════════════════════════
# TAB 2 — ANALYSE DU PROFIL
# ══════════════════════════════════════════════════════════════

with tab2:
    if df_ref is None:
        st.info("Données de référence non disponibles. Lance d'abord train_model.py")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            # Distribution du score par rapport à la population
            fig_dist = px.histogram(
                df_ref, x="revolving_utilization",
                color="default",
                nbins=50,
                color_discrete_map={0: "#27ae60", 1: "#E30613"},
                labels={
                    "revolving_utilization": "Utilisation crédit renouvelable",
                    "default": "Défaut",
                    "count": "Nombre de clients",
                },
                title="Distribution : Utilisation crédit par profil",
                barmode="overlay",
                opacity=0.65,
            )
            fig_dist.add_vline(
                x=revolving_utilization, line_dash="dash",
                line_color="#E30613", line_width=2,
                annotation_text="Client analysé",
                annotation_position="top right",
            )
            fig_dist.update_layout(
                height=300, margin=dict(t=40, b=30, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="",
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_b:
            # Distribution de l'âge
            fig_age = px.histogram(
                df_ref, x="age",
                color="default",
                nbins=40,
                color_discrete_map={0: "#27ae60", 1: "#E30613"},
                labels={"age": "Âge", "default": "Défaut"},
                title="Distribution : Âge par profil de risque",
                barmode="overlay",
                opacity=0.65,
            )
            fig_age.add_vline(
                x=age, line_dash="dash",
                line_color="#E30613", line_width=2,
                annotation_text=f"Client : {age} ans",
            )
            fig_age.update_layout(
                height=300, margin=dict(t=40, b=30, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="",
            )
            st.plotly_chart(fig_age, use_container_width=True)

        # Comparaison du profil vs moyenne de la population
        st.markdown("### Positionnement vs population de référence")
        pop_means = df_ref.drop(columns=["default"]).mean()

        metrics_display = {
            "revolving_utilization": ("Utilisation crédit", revolving_utilization),
            "debt_ratio": ("Ratio endettement", debt_ratio),
            "monthly_income": ("Revenu mensuel (€)", monthly_income),
            "nb_open_credit_lines": ("Lignes de crédit", nb_open_credit_lines),
        }

        cols = st.columns(4)
        for i, (key, (label, client_val)) in enumerate(metrics_display.items()):
            pop_val = pop_means[key]
            delta = client_val - pop_val
            delta_pct = (delta / pop_val * 100) if pop_val != 0 else 0
            with cols[i]:
                st.metric(
                    label=label,
                    value=f"{client_val:,.1f}" if key == "monthly_income" else f"{client_val:.2f}",
                    delta=f"{delta_pct:+.1f}% vs moy.",
                    delta_color="inverse" if key in ["revolving_utilization", "debt_ratio"] else "normal",
                )


# ══════════════════════════════════════════════════════════════
# TAB 3 — FACTEURS DE RISQUE
# ══════════════════════════════════════════════════════════════

with tab3:
    col_imp, col_radar = st.columns([1.2, 1])

    with col_imp:
        importances = model_stats["feature_importances"]
        imp_df = pd.DataFrame(
            list(importances.items()), columns=["Variable", "Importance"]
        ).sort_values("Importance", ascending=True).tail(10)

        # Labels lisibles
        label_map = {
            "revolving_utilization": "Utilisation crédit renouvelable",
            "total_late_payments": "Total retards de paiement",
            "debt_to_income": "Ratio dette/revenu",
            "nb_times_90_days_late": "Retards 90+ jours",
            "age": "Âge",
            "monthly_income": "Revenu mensuel",
            "debt_ratio": "Ratio d'endettement",
            "nb_late_30_59": "Retards 30-59 jours",
            "credit_load": "Charge crédit totale",
            "nb_late_60_89": "Retards 60-89 jours",
            "high_utilization": "Utilisation élevée (flag)",
            "nb_open_credit_lines": "Lignes de crédit ouvertes",
            "nb_real_estate_loans": "Prêts immobiliers",
            "nb_dependents": "Personnes à charge",
            "is_young": "Profil jeune (flag)",
        }
        imp_df["Variable"] = imp_df["Variable"].map(lambda x: label_map.get(x, x))

        fig_imp = go.Figure(go.Bar(
            x=imp_df["Importance"],
            y=imp_df["Variable"],
            orientation="h",
            marker_color=[
                "#E30613" if i >= len(imp_df) - 3 else "#1a1a2e"
                for i in range(len(imp_df))
            ],
        ))
        fig_imp.update_layout(
            title="Top 10 variables prédictives",
            height=380, margin=dict(t=40, b=30, l=0, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Importance (Gain)",
            yaxis_title="",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_radar:
        # Radar chart du profil client
        categories = [
            "Utilisation\ncrédit", "Endettement",
            "Revenus", "Historique\npaiement", "Diversification\ncrédit"
        ]

        # Normalisation 0-1 (inversée pour "revenus" et "diversification")
        radar_client = [
            min(revolving_utilization / 1.5, 1),
            min(debt_ratio / 5, 1),
            1 - min(monthly_income / 15000, 1),   # inversé
            min((nb_late_30_59 + nb_late_60_89 + nb_times_90_days_late) / 10, 1),
            1 - min(nb_open_credit_lines / 15, 1), # inversé
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_client,
            theta=categories,
            fill="toself",
            name="Profil client",
            line_color="#E30613",
            fillcolor="rgba(227,6,19,0.15)",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickfont_size=10),
            ),
            showlegend=False,
            title="Radar de risque (0 = faible risque)",
            height=360,
            margin=dict(t=50, b=30, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Explication textuelle
    st.markdown("### Interprétation du modèle")
    st.markdown("""
    <div class="info-box">
    Le modèle <strong>Gradient Boosting</strong> identifie trois signaux majeurs de risque dans ce dataset bancaire :
    <br><br>
    <strong>1. Utilisation du crédit renouvelable</strong> — Un taux > 70% indique que le client est proche de ses limites.
    C'est le prédicteur le plus fort du défaut futur.<br>
    <strong>2. Historique de retards</strong> — Chaque retard de paiement, même court (30 jours), multiplie significativement
    le risque de défaut.<br>
    <strong>3. Ratio dette/revenu</strong> — Un revenu faible couplé à un endettement élevé crée une vulnérabilité structurelle.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — PERFORMANCE MODÈLE
# ══════════════════════════════════════════════════════════════

with tab4:
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        auc = model_stats["auc"]
        color_auc = "#27ae60" if auc >= 0.80 else "#f39c12"
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">AUC-ROC</div>
            <div class="kpi-value" style="color:{color_auc}">{auc:.4f}</div>
        </div>""", unsafe_allow_html=True)
        st.caption("Capacité discriminante du modèle (> 0.80 = excellent)")

    with col_p2:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">Données d'entraînement</div>
            <div class="kpi-value">{model_stats['n_train']:,}</div>
        </div>""", unsafe_allow_html=True)
        st.caption("Clients utilisés pour l'entraînement")

    with col_p3:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-title">Taux de défaut dataset</div>
            <div class="kpi-value">{model_stats['default_rate']:.2%}</div>
        </div>""", unsafe_allow_html=True)
        st.caption("Prévalence réelle du défaut (classe déséquilibrée)")

    st.markdown("---")
    st.markdown("### À propos du modèle")
    st.markdown(f"""
    <div class="info-box">
    <strong>Algorithme :</strong> {model_stats['model_name']}<br>
    <strong>Rééquilibrage :</strong> SMOTE (Synthetic Minority Oversampling Technique)<br>
    <strong>Seuil de décision :</strong> {model_stats['threshold']:.0%} (optimisé pour minimiser les faux négatifs en contexte bancaire)<br>
    <strong>Dataset :</strong> Give Me Some Credit — Kaggle (150,000 clients, données bancaires réelles)<br><br>
    Le seuil de décision a été abaissé à {model_stats['threshold']:.0%} (vs 50% par défaut) pour prendre en compte
    l'asymétrie des coûts dans le secteur bancaire : un faux négatif (accorder un crédit à un client qui va faire défaut)
    coûte bien plus cher qu'un faux positif (refuser un bon client).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Pipeline complet")
    steps = [
        ("📥 Ingestion", "cs-training.csv · 150K lignes"),
        ("🧹 Nettoyage ETL", "Valeurs manquantes · Outliers · Encoding"),
        ("⚙️ Feature Engineering", "5 nouvelles variables construites"),
        ("⚖️ SMOTE", "Rééquilibrage classe minoritaire (défaut ~6%)"),
        ("🤖 Gradient Boosting", "300 arbres · depth=5 · lr=0.05"),
        ("📊 Évaluation", f"AUC = {model_stats['auc']:.4f} sur test set 20%"),
        ("🚀 Déploiement", "Streamlit Cloud · Scoring temps réel"),
    ]
    for i, (step, detail) in enumerate(steps):
        arrow = " → " if i < len(steps) - 1 else ""
        st.markdown(f"**{step}** — {detail}{arrow}")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#888;font-size:0.8rem'>"
        "Projet Data Analyst · Vincent Loïc MONTI · "
        "<a href='https://github.com/Arvador' target='_blank'>GitHub</a> · "
        "Inspiré des pratiques analytiques bancaires · Société Générale"
        "</div>",
        unsafe_allow_html=True
    )
