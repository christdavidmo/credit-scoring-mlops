"""
Entraînement et comparaison de modèles de credit scoring.

Ce script :
1. Charge les données préprocessées via le module preprocessing
2. Entraîne 3 modèles (LogReg, Random Forest, XGBoost)
3. Logge chaque entraînement dans MLFlow
4. Calcule les métriques classiques + le score métier custom
5. Identifie le meilleur modèle selon le coût métier

Auteur : David
Projet : MLOps Credit Scoring
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report
)

# Imports de nos modules custom
from src.data.preprocessing import prepare_data
from src.models.business_score import business_cost, business_score


# Configuration MLFlow
MLFLOW_EXPERIMENT_NAME = "credit_scoring"
DATA_PATH = 'data/processed/application_train_clean.csv'


def evaluate_model(model, X_test, y_test, model_name):
    """
    Évalue un modèle et retourne toutes les métriques utiles.
    
    Parameters
    ----------
    model : sklearn-compatible model
        Modèle entraîné
    X_test, y_test : array-like
        Données de test
    model_name : str
        Nom du modèle (pour affichage)
    
    Returns
    -------
    dict
        Dictionnaire des métriques
    """
    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probabilité de la classe 1
    
    # Matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Métriques
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'f1': f1_score(y_test, y_pred),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'business_cost': business_cost(y_test, y_pred),
        'business_score': business_score(y_test, y_pred),
    }
    
    # Affichage
    print(f"\n--- Résultats {model_name} ---")
    print(f"  Accuracy        : {metrics['accuracy']:.4f}")
    print(f"  AUC             : {metrics['auc']:.4f}")
    print(f"  F1              : {metrics['f1']:.4f}")
    print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    print(f"  Business cost : {metrics['business_cost']:,}")
    print(f"  Business score: {metrics['business_score']:.4f}")
    
    return metrics


def train_and_log_model(model, model_name, params, X_train, X_test, y_train, y_test):
    """
    Entraîne un modèle et logge tout dans MLFlow.
    
    Parameters
    ----------
    model : sklearn-compatible model
        Modèle à entraîner
    model_name : str
        Nom du run MLFlow
    params : dict
        Paramètres à logger
    X_train, X_test, y_train, y_test : array-like
        Données train/test
    
    Returns
    -------
    metrics : dict
    """
    with mlflow.start_run(run_name=model_name):
        
        print(f"\n{'='*60}")
        print(f"Entraînement : {model_name}")
        print(f"{'='*60}")
        
        # Logger les paramètres
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_param("model_type", model_name)
        
        # Entraînement avec mesure du temps
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        print(f"  Temps d'entraînement : {train_time:.1f}s")
        mlflow.log_metric("train_time_sec", train_time)
        
        # Évaluation
        metrics = evaluate_model(model, X_test, y_test, model_name)
        
        # Logger toutes les métriques
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        # Sauvegarder le modèle
        mlflow.sklearn.log_model(model, "model")
        
        return metrics


def main():
    """Fonction principale : pipeline complet d'entraînement."""
    
    print("=" * 60)
    print("ENTRAÎNEMENT DES MODÈLES DE CREDIT SCORING")
    print("=" * 60)
    
    # 1. Charger et préparer les données
    print("\n[1/3] Chargement des données...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Dataset chargé : {df.shape}")
    
    print("\n[2/3] Preprocessing...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
    
    # 2. Configurer MLFlow
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # 3. Définir les modèles à tester
    print("\n[3/3] Entraînement des modèles...")
    
    models_config = [
        {
            'name': 'LogisticRegression',
            'model': LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',  # gestion du déséquilibre
                random_state=42,
                n_jobs=-1
            ),
            'params': {
                'C': 1.0,
                'max_iter': 1000,
                'class_weight': 'balanced'
            }
        },
        {
            'name': 'RandomForest',
            'model': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'class_weight': 'balanced'
            }
        },
        {
            'name': 'XGBoost',
            'model': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # ratio déséquilibre
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'scale_pos_weight': float((y_train == 0).sum() / (y_train == 1).sum())
            }
        },
    ]
    
    # 4. Entraîner et tracker
    all_results = []
    for config in models_config:
        metrics = train_and_log_model(
            model=config['model'],
            model_name=config['name'],
            params=config['params'],
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test
        )
        metrics['model_name'] = config['name']
        all_results.append(metrics)
    
    # 5. Comparaison finale
    print("\n" + "=" * 60)
    print("COMPARAISON FINALE")
    print("=" * 60)
    
    results_df = pd.DataFrame(all_results)[
        ['model_name', 'accuracy', 'auc', 'f1', 'business_cost', 'business_score']
    ].sort_values('business_score', ascending=False)
    
    print("\n", results_df.to_string(index=False))
    
    best = results_df.iloc[0]
    print(f"\n Meilleur modèle (selon score métier) : {best['model_name']}")
    print(f"   Business score : {best['business_score']:.4f}")
    print(f"   Business cost  : {int(best['business_cost']):,}")
    print(f"\n Voir les résultats sur http://127.0.0.1:5000")


if __name__ == "__main__":
    main()