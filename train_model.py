"""
train_model.py
──────────────
Script d'entraînement du modèle de scoring crédit.
À exécuter UNE SEULE FOIS après avoir téléchargé le dataset Kaggle.

Dataset : Give Me Some Credit
https://www.kaggle.com/competitions/GiveMeSomeCredit/data
→ Télécharger cs-training.csv et le placer dans le dossier data/
"""

import pandas as pd
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# 1. CHARGEMENT ET NETTOYAGE
# ──────────────────────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)

    # Renommer les colonnes pour plus de lisibilité
    df.columns = [
        "default",                         # Variable cible (1 = défaut)
        "revolving_utilization",           # Taux utilisation crédit renouvelable
        "age",                             # Âge
        "nb_late_30_59",                   # Retards 30-59j (2 ans)
        "debt_ratio",                      # Ratio d'endettement
        "monthly_income",                  # Revenu mensuel
        "nb_open_credit_lines",            # Lignes de crédit ouvertes
        "nb_times_90_days_late",           # Retards 90+ jours
        "nb_real_estate_loans",            # Prêts immobiliers
        "nb_late_60_89",                   # Retards 60-89j (2 ans)
        "nb_dependents",                   # Nombre de dépendants
    ]

    print(f"Dataset chargé : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
    print(f"Taux de défaut : {df['default'].mean():.2%}")

    # ── Valeurs manquantes ──
    # monthly_income : médiane par tranche d'âge
    df["monthly_income"] = df.groupby(
        pd.cut(df["age"], bins=[0, 30, 45, 60, 120])
    )["monthly_income"].transform(lambda x: x.fillna(x.median()))
    df["monthly_income"] = df["monthly_income"].fillna(df["monthly_income"].median())

    # nb_dependents : 0 (personne sans dépendant)
    df["nb_dependents"] = df["nb_dependents"].fillna(0)

    # ── Outliers ──
    df = df[df["age"].between(18, 100)]
    df = df[df["revolving_utilization"] <= 1.5]
    df = df[df["debt_ratio"] <= 20]
    df = df[df["monthly_income"] <= 50_000]

    # ── Feature engineering ──
    df["total_late_payments"] = (
        df["nb_late_30_59"] + df["nb_late_60_89"] + df["nb_times_90_days_late"]
    )
    df["debt_to_income"] = df["debt_ratio"] * df["monthly_income"]
    df["credit_load"] = df["nb_open_credit_lines"] + df["nb_real_estate_loans"]
    df["is_young"] = (df["age"] < 30).astype(int)
    df["high_utilization"] = (df["revolving_utilization"] > 0.7).astype(int)

    print(f"Après nettoyage : {df.shape[0]:,} lignes")
    return df


# ──────────────────────────────────────────────────────────────
# 2. ENTRAÎNEMENT
# ──────────────────────────────────────────────────────────────

def train(df: pd.DataFrame):
    feature_cols = [
        "revolving_utilization", "age", "nb_late_30_59", "debt_ratio",
        "monthly_income", "nb_open_credit_lines", "nb_times_90_days_late",
        "nb_real_estate_loans", "nb_late_60_89", "nb_dependents",
        "total_late_payments", "debt_to_income", "credit_load",
        "is_young", "high_utilization",
    ]

    X = df[feature_cols]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE pour rééquilibrer les classes (le défaut est rare ~6%)
    print("Application de SMOTE pour rééquilibrage des classes...")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # Modèle : Gradient Boosting (meilleure performance sur ce dataset)
    print("Entraînement du modèle Gradient Boosting...")
    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        n_iter_no_change=20,
        validation_fraction=0.1,
    )

    scaler = StandardScaler()
    X_res_sc = scaler.fit_transform(X_res)
    X_test_sc = scaler.transform(X_test)

    model.fit(X_res_sc, y_res)

    # ── Évaluation ──
    y_pred = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    print(f"\n{'─'*50}")
    print(f"AUC-ROC : {auc:.4f}")
    print(f"\nRapport de classification :")
    print(classification_report(y_test, y_pred))

    # Importance des variables
    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("\nTop 10 variables importantes :")
    print(importances.head(10).to_string())

    # ── Sauvegarde ──
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_cols, "models/features.pkl")

    # Stats pour le dashboard
    stats = {
        "auc": round(auc, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "default_rate": round(y.mean(), 4),
        "feature_importances": importances.to_dict(),
        "model_name": "Gradient Boosting Classifier",
        "threshold": 0.35,   # seuil ajusté pour le secteur bancaire
    }
    joblib.dump(stats, "models/stats.pkl")

    # Données de référence pour comparaison dans l'app
    df_ref = df[feature_cols + ["default"]].copy()
    df_ref.to_parquet("data/reference_data.parquet", index=False)
    print("\nModèle sauvegardé dans models/")
    print("Données de référence sauvegardées dans data/")
    return stats


# ──────────────────────────────────────────────────────────────
# 3. MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "data/cs-training.csv"

    if not os.path.exists(DATA_PATH):
        print("❌ Fichier introuvable !")
        print(f"   Télécharge cs-training.csv depuis Kaggle et place-le dans : {DATA_PATH}")
        print("   URL : https://www.kaggle.com/competitions/GiveMeSomeCredit/data")
    else:
        df = load_and_clean(DATA_PATH)
        stats = train(df)
        print(f"\n✅ Entraînement terminé ! AUC = {stats['auc']}")
        print("   Lance l'app avec : streamlit run app.py")
