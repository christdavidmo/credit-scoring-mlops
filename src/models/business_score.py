"""
Score métier custom pour le projet de credit scoring.

Ce module définit une fonction de coût qui reflète les enjeux business :
un Faux Négatif (mauvais client accepté) coûte 10× plus qu'un Faux Positif
(bon client refusé).

Auteur : David
Projet : MLOps Credit Scoring
"""

import numpy as np
from sklearn.metrics import confusion_matrix


# Coefficients métier par défaut
DEFAULT_COST_FN = 10  # Coût d'un Faux Négatif
DEFAULT_COST_FP = 1   # Coût d'un Faux Positif


def business_cost(y_true, y_pred, cost_fn=DEFAULT_COST_FN, cost_fp=DEFAULT_COST_FP):
    """
    Calcule le coût métier d'un modèle de credit scoring.

    Un FN (mauvais client accepté) coûte cost_fn (par défaut 10).
    Un FP (bon client refusé) coûte cost_fp (par défaut 1).

    Parameters
    ----------
    y_true : array-like
        Les vraies valeurs (0 ou 1)
    y_pred : array-like
        Les prédictions du modèle (0 ou 1)
    cost_fn : int, default=10
        Le coût d'un Faux Négatif
    cost_fp : int, default=1
        Le coût d'un Faux Positif

    Returns
    -------
    int
        Le coût métier total (plus c'est bas, mieux c'est)

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 1, 0])
    >>> business_cost(y_true, y_pred)
    11
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cout = (cost_fn * fn) + (cost_fp * fp)
    return cout








def business_score(y_true, y_pred, cost_fn=DEFAULT_COST_FN, cost_fp=DEFAULT_COST_FP):
    """
    Calcule un score métier normalisé entre 0 et 1.

    Le score est normalisé par rapport au coût maximum possible (cas où
    le modèle se tromperait à 100% sur les défauts).

    score = 1 - (coût_modèle / coût_maximum)

    - score proche de 1 : excellent modèle
    - score proche de 0 : très mauvais modèle

    Parameters
    ----------
    y_true : array-like
        Les vraies valeurs (0 ou 1)
    y_pred : array-like
        Les prédictions du modèle (0 ou 1)
    cost_fn : int, default=10
        Le coût d'un Faux Négatif
    cost_fp : int, default=1
        Le coût d'un Faux Positif

    Returns
    -------
    float
        Score métier normalisé entre 0 et 1 (1 = meilleur)
    """
    # Coût du modèle actuel
    cout = business_cost(y_true, y_pred, cost_fn, cost_fp)

    # Coût maximum théorique : tous les défauts ratés (= 100% FN)
    # + tous les bons clients refusés (= 100% FP)
    n_defauts = (np.array(y_true) == 1).sum()
    n_bons = (np.array(y_true) == 0).sum()
    cout_max = (cost_fn * n_defauts) + (cost_fp * n_bons)

    # Normalisation (on évite la division par zéro)
    if cout_max == 0:
        return 1.0

    score = 1 - (cout / cout_max)
    return score


if __name__ == "__main__":
    # Test rapide quand on exécute le fichier directement
    print("=== Test du module business_score ===\n")

    # Exemple 1 : mini-cas
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0])

    print("Exemple 1 (5 clients) :")
    print(f"  Coût métier : {business_cost(y_true, y_pred)}")
    print(f"  Score métier normalisé : {business_score(y_true, y_pred):.3f}")
    print()

    # Exemple 2 : comparer 2 modèles sur 1000 clients
    y_true_test = np.array([1]*100 + [0]*900)

    y_pred_A = np.array([1]*70 + [0]*30 + [1]*50 + [0]*850)
    y_pred_B = np.array([1]*95 + [0]*5 + [1]*100 + [0]*800)

    print("Exemple 2 (comparaison Modèle A vs B sur 1000 clients) :")
    print(f"  Modèle A : coût={business_cost(y_true_test, y_pred_A)}, score={business_score(y_true_test, y_pred_A):.3f}")
    print(f"  Modèle B : coût={business_cost(y_true_test, y_pred_B)}, score={business_score(y_true_test, y_pred_B):.3f}")