"""
Analyse et sauvegarde du modèle XGBoost champion.

Ce script :
1. Réentraîne le XGBoost de base (modèle champion identifié)
2. Calcule la feature importance globale
3. Sauvegarde le modèle pour la suite (API)
4. Génère un graphique des top features

Auteur : David
Projet : MLOps Credit Scoring
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

from src.data.preprocessing import prepare_data, prepocessing
from src.models.business_score import business_cost, business_score


DATA_PATH = 'data/processed/application_train_clean.csv'
MODELS_DIR = 'models'


def main():
    print("=" * 60)
    print("ANALYSE DU MODÈLE CHAMPION (XGBoost de base)")
    print("=" * 60)
    
    # 1. Charger et préparer
    print("\n[1/5] Chargement des données...")
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
    
    # 2. Récupérer les noms des features après preprocessing
    # (utile pour la feature importance)
    feature_names = (
        list(preprocessor.named_transformers_['num'].get_feature_names_out()) +
        list(preprocessor.named_transformers_['cat'].get_feature_names_out())
    )
    print(f"\n  Nombre de features après preprocessing : {len(feature_names)}")
    
    # 3. Réentraîner XGBoost de base
    print("\n[2/5] Entraînement du modèle champion...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # 4. Évaluation rapide
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print(f"\n[3/5] Performances du modèle :")
    print(f"  Accuracy        : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  AUC             : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"  F1              : {f1_score(y_test, y_pred):.4f}")
    print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    print(f"  Business cost   : {business_cost(y_test, y_pred):,}")
    print(f"  Business score  : {business_score(y_test, y_pred):.4f}")
    
    # 5. Feature importance
    print(f"\n[4/5] Calcul de la feature importance...")
    
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\n  Top 20 features les plus importantes :")
    print(importance_df.head(20).to_string(index=False))
    
    # 6. Graphique
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    top_20 = importance_df.head(20)
    ax.barh(top_20['feature'][::-1], top_20['importance'][::-1], color='#3498db')
    ax.set_xlabel('Importance')
    ax.set_title('Top 20 features - XGBoost')
    plt.tight_layout()
    
    plot_path = os.path.join(MODELS_DIR, 'feature_importance.png')
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"\n  Graphique sauvegardé : {plot_path}")
    
    # 7. Sauvegarder le modèle et le preprocessor
    print(f"\n[5/5] Sauvegarde du modèle...")
    
    model_path = os.path.join(MODELS_DIR, 'xgboost_champion.pkl')
    joblib.dump(model, model_path)
    print(f"  Modèle sauvegardé : {model_path}")
    
    preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"  Preprocessor sauvegardé : {preprocessor_path}")
    
    # Sauvegarder aussi le top 20 en CSV
    importance_path = os.path.join(MODELS_DIR, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"  Feature importance sauvegardée : {importance_path}")
    
    print("\n" + "=" * 60)
    print("Phase 4 TERMINÉE")
    print("=" * 60)
    print("\nProchaine étape : Phase 5 — Création de l'API FastAPI")


if __name__ == "__main__":
    main()