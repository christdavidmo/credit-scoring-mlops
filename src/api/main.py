"""
API FastAPI pour le projet de credit scoring.

Cette API expose le modèle XGBoost champion via une interface REST.

Auteur : David
Projet : MLOps Credit Scoring
"""

from fastapi import FastAPI, HTTPException
from src.api.schemas import ClientInput, PredictionOutput
from src.api.predictor import predict as run_prediction


# Création de l'instance FastAPI
app = FastAPI(
    title="Credit Scoring API",
    description="API de prédiction du risque de défaut sur prêt",
    version="1.0.0"
)


@app.get("/")
def read_root():
    """Endpoint d'accueil."""
    return {
        "message": "Bienvenue sur l'API Credit Scoring",
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Endpoint de healthcheck (utile pour Render plus tard)."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(client: ClientInput):
    """
    Prédit le risque de défaut pour un client.
    
    Le modèle XGBoost est entraîné sur le dataset Home Credit Default Risk.
    Pour les features non fournies, on impute des valeurs par défaut
    (médianes du train set).
    """
    try:
        result = run_prediction(client)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction : {str(e)}"
        )