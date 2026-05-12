"""
Module de prédiction de l'API.

Ce module :
1. Charge le modèle XGBoost et le preprocessor (au démarrage de l'API)
2. Transforme les inputs utilisateur en DataFrame complet
3. Applique le preprocessing
4. Lance la prédiction
5. Retourne la réponse structurée

Auteur : David
"""

import pandas as pd
import json
import joblib
from src.api.schemas import ClientInput, PredictionOutput


# Chemins
MODEL_PATH = 'models/xgboost_champion.pkl'
PREPROCESSOR_PATH = 'models/preprocessor.pkl'
DEFAULTS_PATH = 'src/api/default_values.json'

# Seuil de décision (0.5 = par défaut, ajustable selon le score métier)
DECISION_THRESHOLD = 0.5


# === Chargement au démarrage ===
print("Chargement du modèle XGBoost...")
model = joblib.load(MODEL_PATH)

print("Chargement du preprocessor...")
preprocessor = joblib.load(PREPROCESSOR_PATH)

print("Chargement des valeurs par défaut...")
with open(DEFAULTS_PATH, 'r', encoding='utf-8') as f:
    DEFAULT_VALUES = json.load(f)

print(f"Modèle prêt ({len(DEFAULT_VALUES)} features attendues)")


def build_features_dataframe(client: ClientInput) -> pd.DataFrame:
    """
    Construit un DataFrame complet à partir des inputs utilisateur.

    Les features non fournies sont imputées avec les valeurs par défaut
    (médiane pour les numériques, mode pour les catégorielles).

    Parameters
    ----------
    client : ClientInput
        Les données fournies par l'utilisateur

    Returns
    -------
    pd.DataFrame
        DataFrame avec toutes les features attendues par le modèle
    """
    # Partir des valeurs par défaut
    features = DEFAULT_VALUES.copy()
    
    # Écraser avec les valeurs fournies par l'utilisateur
    user_data = client.model_dump()
    for key, value in user_data.items():
        if value is not None:  # Ne pas écraser avec un None
            features[key] = value
    
    # Recalculer les features dérivées (créées en EDA)
    features = _add_derived_features(features)
    
    # Convertir en DataFrame (1 ligne)
    df = pd.DataFrame([features])
    
    return df


def _add_derived_features(features: dict) -> dict:
    """
    Recalcule les features dérivées (créées en EDA) à partir des valeurs.
    """
    # Âge
    features['AGE_YEARS'] = -features['DAYS_BIRTH'] / 365
    
    # Ancienneté professionnelle + flag retraité
    features['DAYS_EMPLOYED_ANOM'] = int(features['DAYS_EMPLOYED'] == 365243)
    if features['DAYS_EMPLOYED'] == 365243:
        features['EMPLOYED_YEARS'] = features.get('EMPLOYED_YEARS', 0)
    else:
        features['EMPLOYED_YEARS'] = -features['DAYS_EMPLOYED'] / 365
    
    # Ratios financiers
    features['CREDIT_INCOME_RATIO'] = features['AMT_CREDIT'] / features['AMT_INCOME_TOTAL']
    features['ANNUITY_INCOME_RATIO'] = features['AMT_ANNUITY'] / features['AMT_INCOME_TOTAL']
    features['CREDIT_TERM'] = features['AMT_ANNUITY'] / features['AMT_CREDIT']
    
    # Moyenne / Min / Max / Std des EXT_SOURCES
    ext_values = [
        features.get('EXT_SOURCE_1'),
        features.get('EXT_SOURCE_2'),
        features.get('EXT_SOURCE_3')
    ]
    ext_valid = [v for v in ext_values if v is not None]
    
    if ext_valid:
        import numpy as np
        features['EXT_SOURCES_MEAN'] = float(np.mean(ext_valid))
        features['EXT_SOURCES_MIN'] = float(np.min(ext_valid))
        features['EXT_SOURCES_MAX'] = float(np.max(ext_valid))
        features['EXT_SOURCES_STD'] = float(np.std(ext_valid)) if len(ext_valid) > 1 else 0.0
    
    return features


def get_risk_level(probability: float) -> str:
    """Convertit la probabilité en niveau de risque qualitatif."""
    if probability < 0.3:
        return "FAIBLE"
    elif probability < 0.6:
        return "MODÉRÉ"
    else:
        return "ÉLEVÉ"


def predict(client: ClientInput) -> PredictionOutput:
    """
    Pipeline complet de prédiction.

    Parameters
    ----------
    client : ClientInput
        Les données client validées par Pydantic

    Returns
    -------
    PredictionOutput
        La prédiction structurée
    """
    # 1. Construire le DataFrame complet
    df = build_features_dataframe(client)
    
    # 2. Supprimer les colonnes qu'on a supprimées au preprocessing
    cols_to_drop = ['SK_ID_CURR', 'DAYS_BIRTH', 'DAYS_EMPLOYED']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # 3. Appliquer le preprocessor
    X = preprocessor.transform(df)
    
    # 4. Prédire la probabilité
    proba = float(model.predict_proba(X)[0, 1])
    
    # 5. Décision selon le seuil
    decision = "REFUSER" if proba >= DECISION_THRESHOLD else "ACCORDER"
    risk_level = get_risk_level(proba)
    
    return PredictionOutput(
        probability_default=proba,
        decision=decision,
        threshold=DECISION_THRESHOLD,
        risk_level=risk_level
    )