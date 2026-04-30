# 🏦 Credit Risk Scoring Platform
### Analyse prédictive du risque de défaut client bancaire

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Contexte

Projet réalisé dans le cadre de ma recherche de stage **Data Analyst / Business Analyst** dans le secteur bancaire.

Ce projet simule un système de scoring crédit tel qu'on pourrait le trouver dans un département **Risques** d'une banque retail comme Société Générale, BNP Paribas ou Crédit Agricole.

---

## 🎯 Objectif

Prédire la probabilité qu'un client **fasse défaut sur un crédit** dans les 2 prochaines années, à partir de ses données financières et comportementales.

---

## 📊 Dataset

**Give Me Some Credit** — Kaggle Competition  
- 150 000 clients bancaires réels
- 11 variables financières (revenus, endettement, historique de crédit...)
- Variable cible : défaut de paiement (6% de défauts — classe déséquilibrée)

---

## 🏗️ Architecture du projet

```
credit_scoring_sg/
│
├── app.py                  # Application Streamlit principale
├── train_model.py          # Script d'entraînement ML
├── requirements.txt        # Dépendances Python
│
├── data/
│   ├── cs-training.csv     # Dataset Kaggle (à télécharger)
│   └── reference_data.parquet
│
└── models/
    ├── model.pkl           # Modèle Gradient Boosting entraîné
    ├── scaler.pkl          # StandardScaler
    ├── features.pkl        # Liste des features
    └── stats.pkl           # Métriques et métadonnées
```

---

## 🔬 Méthodologie

### 1. Préparation des données (ETL)
- Traitement des **valeurs manquantes** (monthly_income : médiane par tranche d'âge)
- Suppression des **outliers** (âge, utilisation crédit, revenus)
- **Feature engineering** : 5 nouvelles variables construites
  - `total_late_payments` = somme de tous les retards
  - `debt_to_income` = dette ratio × revenu
  - `credit_load` = lignes crédit + prêts immobiliers
  - `is_young` = flag client < 30 ans
  - `high_utilization` = flag utilisation > 70%

### 2. Modélisation
- **Algorithme** : Gradient Boosting Classifier (GBC)
- **Rééquilibrage** : SMOTE (classe minoritaire ~6%)
- **Validation** : Train/Test split 80/20, stratifié

### 3. Résultats
| Métrique | Valeur |
|----------|--------|
| **AUC-ROC** | **~0.8395** |
| Seuil de décision | 35% |
| Données d'entraînement | ~120 000 clients |

### 4. Déploiement
- Application **Streamlit** avec scoring en temps réel
- Formulaire de saisie client
- Visualisations interactives **Plotly**
- Comparaison profil vs population de référence

---

## 🚀 Installation et lancement

```bash
# 1. Cloner le repo
git clone https://github.com/Arvador/credit-risk-scoring
cd credit-risk-scoring

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Télécharger le dataset
# → https://www.kaggle.com/competitions/GiveMeSomeCredit/data
# → Placer cs-training.csv dans data/

# 4. Entraîner le modèle
python train_model.py

# 5. Lancer l'application
streamlit run app.py
```

---

## 📱 Fonctionnalités de l'app

| Onglet | Contenu |
|--------|---------|
| **Score & Décision** | Jauge crédit (300–850), probabilité de défaut, recommandation |
| **Analyse du profil** | Comparaison vs population, positionnement sur distributions |
| **Facteurs de risque** | Importance des variables, radar chart de risque |
| **Performance modèle** | AUC-ROC, pipeline complet, méthodologie |

---

## 🛠️ Stack technique

`Python` · `Scikit-learn` · `Streamlit` · `Plotly` · `Pandas` · `NumPy` · `SMOTE` · `Joblib`

---

## 👤 Auteur

**Vincent Loïc MONTI** — Data Analyst · Master IA & Big Data (ESTIAM Paris)

- 💼 [LinkedIn](https://linkedin.com/in/vincent-loic-monti)
- 💻 [GitHub](https://github.com/Arvador)
- 📧 loicmonti318@gmail.com

> *Ouvert aux opportunités de stage Data Analyst / Business Analyst — secteur bancaire & fintech*

---

## 📄 Licence

Projet à ne pas recopier ni à des fins éducatives, ni professionnelles, ni pour toutes autres fins.
