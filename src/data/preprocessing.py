



import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler



# SK_ID_CURR = cette colonne represente l'ID
# DAYS_BIRTH
# DAYS_EMPLOYED
colonnes_a_supprimer = ['SK_ID_CURR','DAYS_BIRTH','DAYS_EMPLOYED',]



def prepocessing(X):

    # ==== sépare d'abord les différents types de colonne ====
    cols_num = X.select_dtypes(include=['number']).columns.tolist()
    cols_cat = X.select_dtypes(include=['object']).columns.tolist()

    #===== Que faifre sur chaque type de colonne
    pipeline_num = Pipeline(steps=[ 
                            ('imputer',SimpleImputer(strategy='median') ),
                            ('scaler',StandardScaler())  ])
    
    pipeline_cat = Pipeline( steps= [ 
        ('imputer',SimpleImputer(strategy='most_frequent')) ,
        ('One_Hot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))    ])
    
    prepocessor = ColumnTransformer(transformers=[ ('num',pipeline_num,cols_num) ,
                                                  ('cat',pipeline_cat,cols_cat)])
    
    return prepocessor



def prepare_data( df , target_col = 'TARGET' , test_size = 0.2 , random_state = 42):

    # separer le df
    X = df.drop(columns=[target_col])
    Y = df[target_col]

    delet_col = [ c for c in colonnes_a_supprimer if c in X.columns ]

    # Supprimer les colonnes non prédictives
    if delet_col :
        X = X.drop(columns=delet_col)
        print(f"Les colonnes supprimées sont : {delet_col}")

    print(f"Shape avant preprocessing : {X.shape}")
    print(f"Variables numériques    : {len(X.select_dtypes(include=['number']).columns)}")
    print(f"Variables catégorielles : {len(X.select_dtypes(include=['object']).columns)}")



    X_train, X_test, y_train, y_test = train_test_split( X , Y , random_state=random_state , test_size = test_size , stratify=Y)

    preprocessor = prepocessing(X_train)
    
    # fit sur train uniquement (pour éviter la fuite d'info du test set)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)


    print(f"\nShape après preprocessing :")
    print(f"  X_train : {X_train_processed.shape}")
    print(f"  X_test  : {X_test_processed.shape}")

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor






if __name__ == "__main__":

    # === Test du module ===
    print("=== Test du module preprocessing ===\n")


    # === charge d'abord le dataset ===
    df = pd.read_csv('data/processed/application_train_clean.csv')
    L,C = df.shape
    print(f"Le DataFrame contient {L} ligne(s) et {C} colonne(s)")


    X_train , X_test , y_train ,y_test , preprocessor = prepare_data(df)

    print("\n Preprocessing terminé avec succès")
    print(f"Nombre de features après One-Hot Encoding : {X_train.shape[1]}")
