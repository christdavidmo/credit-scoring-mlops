"""
Calcule les valeurs par défaut (médianes/modes) à partir du train set.

Ces valeurs sont sauvegardées dans un JSON et utilisées par l'API pour
imputer les features non fournies par l'utilisateur.

Lance ce script UNE SEULE FOIS pour générer le JSON.

Auteur : David
"""

import pandas as pd
import json
import os

DATA_PATH = 'data/processed/application_train_clean.csv'
OUTPUT_PATH = 'src/api/default_values.json'


def main():
    print("Chargement du dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # On enlève la cible et les colonnes qu'on supprime au preprocessing
    cols_to_exclude = ['TARGET', 'SK_ID_CURR']
    df = df.drop(columns=[c for c in cols_to_exclude if c in df.columns])
    
    print(f"Calcul des valeurs par défaut sur {len(df)} clients...")
    
    defaults = {}
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # Numérique : médiane
            defaults[col] = float(df[col].median())
        else:
            # Catégoriel : mode (valeur la plus fréquente)
            defaults[col] = str(df[col].mode().iloc[0])
    
    # Sauvegarder
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(defaults, f, indent=2, ensure_ascii=False)
    
    print(f" {len(defaults)} valeurs par défaut sauvegardées dans : {OUTPUT_PATH}")
    print(f"   Aperçu :")
    for i, (k, v) in enumerate(list(defaults.items())[:10]):
        print(f"     {k}: {v}")
    print(f"     ... et {len(defaults)-10} autres")


if __name__ == "__main__":
    main()