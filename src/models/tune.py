"""
Optimisation des hyperparamètres de XGBoost via GridSearchCV.

Stratégie :
- Tuning sur un sous-échantillon (30% du train) pour gagner du temps
- GridSearchCV avec validation croisée (3 plis)
- Scorer custom basé sur notre business_score (optimisation alignée avec l'objectif métier)
- Réentraînement final du modèle optimal sur 100% des données
- Logging dans MLFlow

Auteur : David
Projet : MLOps Credit Scoring
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import time

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, make_scorer
)
from xgboost import XGBClassifier

from src.data.preprocessing import prepare_data
from src.models.business_score import business_cost, business_score


MLFLOW_EXPERIMENT_NAME = "credit_scoring"
DATA_PATH = 'data/processed/application_train_clean.csv'


def main():
    print("=" * 60)
    print("HYPERPARAMETER TUNING — XGBoost (optimisé sur business_score)")
    print("=" * 60)
    
    # 1. Charger et préparer les données
    print("\n[1/4] Chargement et preprocessing...")
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
    
    # 2. Sous-échantillonnage pour le tuning
    print("\n[2/4] Création d'un sous-échantillon pour le tuning (30%)...")
    np.random.seed(42)
    sample_size = int(len(X_train) * 0.3)
    sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
    X_train_sample = X_train[sample_idx]
    y_train_sample = y_train.iloc[sample_idx]
    
    print(f"  Taille du sous-échantillon : {sample_size:,} clients")
    print(f"  Taux de défauts conservé    : {y_train_sample.mean()*100:.2f}%")
    
    # 3. Définir la grille d'hyperparamètres
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
    }
    
    print(f"\n[3/4] GridSearchCV en cours...")
    print(f"  Combinaisons à tester : 8")
    print(f"  Avec validation croisée à 3 plis : 24 entraînements")
    print(f"  Optimisation sur : business_score (métier custom)")
    print(f"  Patience, ça peut prendre quelques minutes...")
    
    # 4. Lancer le GridSearchCV
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="XGBoost_GridSearchCV_BusinessScore"):
        
        # Modèle de base
        scale_pos_weight = (y_train_sample == 0).sum() / (y_train_sample == 1).sum()
        base_model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        # Scorer custom : on optimise directement sur notre business_score
        # greater_is_better=True car un score haut = bon modèle
        business_scorer = make_scorer(business_score, greater_is_better=True)
        
        # GridSearchCV avec scorer custom
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=business_scorer,
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        
        start = time.time()
        grid_search.fit(X_train_sample, y_train_sample)
        tuning_time = time.time() - start
        
        print(f"\n  ✅ Tuning terminé en {tuning_time:.1f}s ({tuning_time/60:.1f} min)")
        print(f"  Meilleurs hyperparamètres : {grid_search.best_params_}")
        print(f"  Meilleur business_score (sous-échantillon, CV) : {grid_search.best_score_:.4f}")
        
        # Logger les meilleurs paramètres dans MLFlow
        for key, value in grid_search.best_params_.items():
            mlflow.log_param(f"best_{key}", value)
        mlflow.log_metric("best_cv_business_score", grid_search.best_score_)
        mlflow.log_metric("tuning_time_sec", tuning_time)
        mlflow.log_param("scoring_metric", "business_score (custom)")
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("sample_ratio", 0.3)
        
        # 5. Réentraîner sur 100% du train avec les meilleurs paramètres
        print(f"\n[4/4] Réentraînement sur 100% des données avec les meilleurs hyperparamètres...")
        
        best_model = XGBClassifier(
            **grid_search.best_params_,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        start = time.time()
        best_model.fit(X_train, y_train)
        final_train_time = time.time() - start
        print(f"  Temps d'entraînement final : {final_train_time:.1f}s")
        mlflow.log_metric("final_train_time_sec", final_train_time)
        
        # 6. Évaluation finale sur le test set
        print(f"\n  Évaluation sur le test set...")
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
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
        
        print(f"\n--- Résultats XGBoost optimisé sur business_score ---")
        print(f"  Accuracy        : {metrics['accuracy']:.4f}")
        print(f"  AUC             : {metrics['auc']:.4f}")
        print(f"  F1              : {metrics['f1']:.4f}")
        print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
        print(f"  ⭐ Business cost : {metrics['business_cost']:,}")
        print(f"  ⭐ Business score: {metrics['business_score']:.4f}")
        
        # Logger toutes les métriques finales
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        # Sauvegarder le modèle optimisé
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"\n👉 Voir les résultats sur http://127.0.0.1:5000")
        print(f"   Run name : XGBoost_GridSearchCV_BusinessScore")
        
        # Petit comparatif rapide
        print(f"\n--- Comparaison avec XGBoost de base ---")
        print(f"  XGBoost base    : business_cost = 31,833 | business_score = 0.7002")
        print(f"  XGBoost optimisé: business_cost = {metrics['business_cost']:,} | business_score = {metrics['business_score']:.4f}")
        
        if metrics['business_cost'] < 31833:
            print(f"  ✅ Amélioration : -{31833 - metrics['business_cost']:,} en coût métier")
        else:
            print(f"  ⚠️ Pas d'amélioration cette fois")


if __name__ == "__main__":
    main()