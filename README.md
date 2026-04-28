# Credit Scoring MLOps Project

Projet de bout en bout : développement, déploiement et monitoring d'un modèle de scoring de risque de crédit.

## Contexte

Prédire la probabilité qu'un client fasse défaut sur son prêt à partir des données du Home Credit Default Risk dataset (Kaggle).

## Stack technique

- **ML** : scikit-learn, XGBoost, LightGBM
- **Tracking** : MLFlow
- **API** : FastAPI
- **UI** : Streamlit
- **Monitoring** : Evidently (data drift)
- **CI/CD** : GitHub Actions
- **Déploiement** : Render

## Structure

```
.
├── data/              # Données brutes et traitées (non versionnées)
├── notebooks/         # EDA et expérimentations
├── src/
│   ├── data/          # Preprocessing
│   ├── models/        # Entraînement, évaluation
│   ├── api/           # FastAPI
│   └── monitoring/    # Data drift
├── streamlit_app/     # Interface utilisateur
├── tests/             # Tests unitaires
└── .github/workflows/ # CI/CD
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

## Auteur

David — M2 Data & AI