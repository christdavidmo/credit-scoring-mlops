"""
 ======== les variables selon leurs impotrtances ========

EXT_SOURCES_MEAN,0.12702414
NAME_EDUCATION_TYPE_Higher education,0.020355223
CODE_GENDER_M,0.017711174
NAME_INCOME_TYPE_Pensioner,0.014709617
NAME_EDUCATION_TYPE_Secondary / secondary special,0.013267862
FLAG_DOCUMENT_3,0.012557027
CODE_GENDER_F,0.012383984
FLAG_OWN_CAR_N,0.011860168
CREDIT_TERM,0.01131464
NB_DOCS_FOURNIS,0.01087464
NAME_CONTRACT_TYPE_Cash loans,0.010853756
EXT_SOURCES_MAX,0.010692771
AMT_GOODS_PRICE,0.009898936
REG_CITY_NOT_LIVE_CITY,0.009345575
REGION_RATING_CLIENT_W_CITY,0.009283631
FLOORSMAX_MEDI,0.009158261
NAME_FAMILY_STATUS_Married,0.008939366
DEF_60_CNT_SOCIAL_CIRCLE,0.0088413395
OCCUPATION_TYPE_Core staff,0.008830314
EMPLOYED_YEARS,0.00872639


Schémas Pydantic pour la validation des données d'entrée/sortie de l'API.

On définit ici la forme exacte de :
- ClientInput : les données qu'un utilisateur envoie pour avoir une prédiction
- PredictionOutput : la réponse renvoyée par l'API

Pydantic valide automatiquement les types et les contraintes.

Auteur : David

"""


from pydantic import BaseModel , Field
from typing import Optional

class ClientInput(BaseModel):
    """
    Données d'entrée pour une prédiction.

    On demande uniquement les variables business essentielles
    (les autres sont imputées avec des valeurs par défaut côté API).
    """

    # === Informations financières ===
    AMT_INCOME_TOTAL: float = Field(
        ..., gt=0,
        description="Revenu total annuel (en €)",
        examples=[150000]
    )
    AMT_CREDIT: float = Field(
        ..., gt=0,
        description="Montant du crédit demandé (en €)",
        examples=[500000]
    )
    AMT_ANNUITY: float = Field(
        ..., gt=0,
        description="Montant de l'annuité (mensualité × 12)",
        examples=[25000]
    )
    AMT_GOODS_PRICE: float = Field(
        ..., gt=0,
        description="Prix du bien financé",
        examples=[450000]
    )

    # === Informations personnelles ===
    CODE_GENDER: str = Field(
        ..., pattern="^[MF]$",
        description="Genre du client (M ou F)",
        examples=["M"]
    )
    DAYS_BIRTH: int = Field(
        ..., lt=0,
        description="Âge en jours négatifs (ex: -12000 = ~33 ans)",
        examples=[-12000]
    )
    CNT_CHILDREN: int = Field(
        ..., ge=0,
        description="Nombre d'enfants",
        examples=[0]
    )
    CNT_FAM_MEMBERS: float = Field(
        ..., ge=1,
        description="Nombre de membres de la famille (incluant le client)",
        examples=[2.0]
    )

    # === Situation professionnelle ===
    NAME_INCOME_TYPE: str = Field(
        ...,
        description="Type de revenu",
        examples=["Working"]
    )
    NAME_EDUCATION_TYPE: str = Field(
        ...,
        description="Niveau d'études",
        examples=["Higher education"]
    )
    NAME_FAMILY_STATUS: str = Field(
        ...,
        description="Situation familiale",
        examples=["Married"]
    )
    DAYS_EMPLOYED: int = Field(
        ...,
        description="Ancienneté en jours négatifs (365243 = retraité)",
        examples=[-3000]
    )
    OCCUPATION_TYPE: Optional[str] = Field(
        None,
        description="Métier (optionnel)",
        examples=["Laborers"]
    )

    # === Type de prêt ===
    NAME_CONTRACT_TYPE: str = Field(
        ...,
        description="Type de contrat",
        examples=["Cash loans"]
    )

    # === Scores externes (optionnels mais TRÈS prédictifs) ===
    EXT_SOURCE_1: Optional[float] = Field(
        None, ge=0, le=1,
        description="Score externe 1 (entre 0 et 1)",
        examples=[0.5]
    )
    EXT_SOURCE_2: Optional[float] = Field(
        None, ge=0, le=1,
        description="Score externe 2 (entre 0 et 1)",
        examples=[0.6]
    )
    EXT_SOURCE_3: Optional[float] = Field(
        None, ge=0, le=1,
        description="Score externe 3 (entre 0 et 1)",
        examples=[0.55]
    )

    # === Patrimoine ===
    FLAG_OWN_CAR: str = Field(
        ..., pattern="^[YN]$",
        description="Possède une voiture (Y ou N)",
        examples=["Y"]
    )
    FLAG_OWN_REALTY: str = Field(
        ..., pattern="^[YN]$",
        description="Possède un bien immobilier (Y ou N)",
        examples=["Y"]
    )


class PredictionOutput(BaseModel):
    """
    Réponse renvoyée par l'API.
    """
    probability_default: float = Field(
        ...,
        description="Probabilité de défaut entre 0 et 1"
    )
    decision: str = Field(
        ...,
        description="Décision finale : ACCORDER ou REFUSER"
    )
    threshold: float = Field(
        ...,
        description="Seuil utilisé pour la décision"
    )
    risk_level: str = Field(
        ...,
        description="Niveau de risque qualitatif (FAIBLE, MODÉRÉ, ÉLEVÉ)"
    )

