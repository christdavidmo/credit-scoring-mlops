"""
Application Streamlit pour le projet de credit scoring.

Interface web qui permet de :
- Saisir les caractéristiques d'un client via un formulaire
- Appeler l'API FastAPI pour obtenir une prédiction
- Visualiser le résultat (probabilité, décision, jauge de risque)

Auteur : David
Projet : MLOps Credit Scoring
"""

import streamlit as st
import requests
import plotly.graph_objects as go


# === Configuration de la page (DOIT ÊTRE EN PREMIER) ===
st.set_page_config(
    page_title="Credit Scoring App",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# === Configuration de l'API ===
API_LOCAL = "http://127.0.0.1:8000/predict"
API_PUBLIC = "https://credit-scoring-mlops-srt6.onrender.com/predict"


# === Style custom CSS ===
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-top: 0;
    }
    .stMetric {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)


# === En-tête ===
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown('<p class="main-header">💳 Credit Scoring</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Évaluation automatisée du risque de défaut sur prêt — Modèle XGBoost</p>', unsafe_allow_html=True)
with col_badge:
    st.markdown("")
    st.markdown("")
    st.info("🤖 **Powered by ML**\nXGBoost · MLFlow · FastAPI")


# === Sidebar complète ===
with st.sidebar:
    # Sélecteur API
    st.header("⚙️ Configuration API")
    api_choice = st.radio(
        "Source",
        options=["🌍 API Publique (Render)", "💻 API Locale"],
        index=0,
        help="Choisis quelle API appeler. L'API publique peut mettre 30-60s à se réveiller."
    )
    API_URL = API_PUBLIC if "Publique" in api_choice else API_LOCAL
    
    st.divider()
    
    st.header("ℹ️ À propos")
    st.markdown("""
    Cette application utilise un modèle de **Machine Learning (XGBoost)** 
    pour évaluer le risque de défaut sur un prêt.

    **Source des données :**  
    [Home Credit Default Risk - Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)

    **Modèle entraîné sur :**  
    • 307 511 clients  
    • 88 features (203 après preprocessing)  
    • Évaluation : score métier custom (10× FN)
    """)
    
    st.divider()
    
    st.header("📊 Métriques du modèle")
    st.metric("AUC", "0.7644")
    st.metric("Business Score", "0.7002")
    st.metric("Accuracy", "72.3%")
    
    st.divider()
    
    st.header("🎯 Top features")
    st.markdown("""
    1. **EXT_SOURCES_MEAN** (12.7%)
    2. NAME_EDUCATION_TYPE
    3. CODE_GENDER
    4. NAME_INCOME_TYPE_Pensioner
    5. FLAG_DOCUMENT_3
    """)


st.divider()


# === Formulaire de saisie ===
st.header("📋 Informations du client")

tab1, tab2, tab3 = st.tabs(["💰 Financier", "👤 Personnel & Pro", "🌐 Scores externes"])

with tab1:
    st.markdown("##### Informations financières du dossier")
    col1, col2 = st.columns(2)
    
    with col1:
        AMT_INCOME_TOTAL = st.number_input(
            "Revenu annuel total (€)",
            min_value=10000, max_value=1000000, value=150000, step=5000
        )
        
        AMT_CREDIT = st.number_input(
            "Montant du crédit demandé (€)",
            min_value=10000, max_value=5000000, value=500000, step=10000
        )
        
        NAME_CONTRACT_TYPE = st.selectbox(
            "Type de contrat",
            options=["Cash loans", "Revolving loans"]
        )
    
    with col2:
        AMT_ANNUITY = st.number_input(
            "Montant de l'annuité (€/an)",
            min_value=1000, max_value=300000, value=25000, step=1000
        )
        
        AMT_GOODS_PRICE = st.number_input(
            "Prix du bien financé (€)",
            min_value=10000, max_value=5000000, value=450000, step=10000
        )
        
        st.markdown("##### 🏠 Patrimoine")
        cola, colb = st.columns(2)
        with cola:
            FLAG_OWN_CAR = st.selectbox(
                "Voiture ?", options=["Y", "N"],
                format_func=lambda x: "Oui" if x == "Y" else "Non"
            )
        with colb:
            FLAG_OWN_REALTY = st.selectbox(
                "Immobilier ?", options=["Y", "N"],
                format_func=lambda x: "Oui" if x == "Y" else "Non"
            )


with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 👤 Identité")
        
        CODE_GENDER = st.selectbox(
            "Genre",
            options=["M", "F"],
            format_func=lambda x: "Homme" if x == "M" else "Femme"
        )
        
        age_years = st.slider("Âge", min_value=18, max_value=80, value=35)
        DAYS_BIRTH = -int(age_years * 365)
        
        CNT_CHILDREN = st.number_input(
            "Nombre d'enfants",
            min_value=0, max_value=15, value=0, step=1
        )
        
        CNT_FAM_MEMBERS = st.number_input(
            "Nb membres de la famille",
            min_value=1, max_value=20, value=2, step=1
        )
        
        NAME_FAMILY_STATUS = st.selectbox(
            "Situation familiale",
            options=[
                "Married", "Single / not married",
                "Civil marriage", "Separated", "Widow"
            ]
        )
    
    with col2:
        st.markdown("##### 💼 Situation professionnelle")
        
        NAME_INCOME_TYPE = st.selectbox(
            "Type de revenu",
            options=[
                "Working", "Commercial associate", "Pensioner",
                "State servant", "Unemployed", "Student"
            ]
        )
        
        NAME_EDUCATION_TYPE = st.selectbox(
            "Niveau d'études",
            options=[
                "Higher education", "Secondary / secondary special",
                "Incomplete higher", "Lower secondary", "Academic degree"
            ]
        )
        
        is_pensioner = st.checkbox("☂️ Le client est retraité")
        
        if is_pensioner:
            DAYS_EMPLOYED = 365243
            st.caption("ℹ️ Code spécial retraité activé")
        else:
            years_employed = st.slider(
                "Ancienneté professionnelle (années)",
                min_value=0, max_value=50, value=8
            )
            DAYS_EMPLOYED = -int(years_employed * 365)
        
        OCCUPATION_TYPE = st.selectbox(
            "Métier",
            options=[
                "Laborers", "Sales staff", "Core staff", "Managers",
                "Drivers", "High skill tech staff", "Accountants",
                "Medicine staff", "Security staff", "Cooking staff",
                "Cleaning staff", "Private service staff",
                "Low-skill Laborers", "Waiters/barmen staff",
                "Secretaries", "Realty agents", "HR staff", "IT staff"
            ]
        )


with tab3:
    st.markdown("##### Scores externes (de bureaux de crédit tiers)")
    st.caption("💡 Ces scores sont les variables les plus prédictives du modèle. Plus ils sont élevés, plus le client est fiable.")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        EXT_SOURCE_1 = st.slider("Score externe 1", 0.0, 1.0, 0.5, 0.01, key="ext1")
        st.caption(f"📊 {EXT_SOURCE_1:.2f}")
    
    with col4:
        EXT_SOURCE_2 = st.slider("Score externe 2", 0.0, 1.0, 0.6, 0.01, key="ext2")
        st.caption(f"📊 {EXT_SOURCE_2:.2f}")
    
    with col5:
        EXT_SOURCE_3 = st.slider("Score externe 3", 0.0, 1.0, 0.55, 0.01, key="ext3")
        st.caption(f"📊 {EXT_SOURCE_3:.2f}")


# === Bouton de prédiction ===
st.divider()
predict_button = st.button(
    "🚀 Lancer l'évaluation du dossier",
    type="primary",
    use_container_width=True
)


# === Exécution de la prédiction ===
if predict_button:
    client_data = {
        "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
        "AMT_CREDIT": AMT_CREDIT,
        "AMT_ANNUITY": AMT_ANNUITY,
        "AMT_GOODS_PRICE": AMT_GOODS_PRICE,
        "CODE_GENDER": CODE_GENDER,
        "DAYS_BIRTH": DAYS_BIRTH,
        "CNT_CHILDREN": CNT_CHILDREN,
        "CNT_FAM_MEMBERS": float(CNT_FAM_MEMBERS),
        "NAME_INCOME_TYPE": NAME_INCOME_TYPE,
        "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE,
        "NAME_FAMILY_STATUS": NAME_FAMILY_STATUS,
        "DAYS_EMPLOYED": DAYS_EMPLOYED,
        "OCCUPATION_TYPE": OCCUPATION_TYPE,
        "NAME_CONTRACT_TYPE": NAME_CONTRACT_TYPE,
        "EXT_SOURCE_1": EXT_SOURCE_1,
        "EXT_SOURCE_2": EXT_SOURCE_2,
        "EXT_SOURCE_3": EXT_SOURCE_3,
        "FLAG_OWN_CAR": FLAG_OWN_CAR,
        "FLAG_OWN_REALTY": FLAG_OWN_REALTY
    }
    
    try:
        with st.spinner("🔄 Analyse du dossier en cours..."):
            response = requests.post(API_URL, json=client_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            proba = result['probability_default']
            decision = result['decision']
            risk_level = result['risk_level']
            threshold = result['threshold']
            
            st.divider()
            st.header("📊 Résultat de l'évaluation")
            
            col_gauge, col_metrics = st.columns([2, 1])
            
            with col_gauge:
                gauge_color = "#10b981" if proba < 0.3 else ("#f59e0b" if proba < 0.6 else "#ef4444")
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probabilité de défaut (%)", 'font': {'size': 18}},
                    number={'suffix': "%", 'font': {'size': 40}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': gauge_color, 'thickness': 0.3},
                        'steps': [
                            {'range': [0, 30], 'color': "#d1fae5"},
                            {'range': [30, 60], 'color': "#fef3c7"},
                            {'range': [60, 100], 'color': "#fee2e2"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold * 100
                        }
                    }
                ))
                fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            with col_metrics:
                st.markdown("###### Décision")
                if decision == "ACCORDER":
                    st.success(f"## ✅ ACCORDER")
                else:
                    st.error(f"## ❌ REFUSER")
                
                st.markdown("###### Niveau de risque")
                if risk_level == "FAIBLE":
                    st.markdown("### 🟢 FAIBLE")
                elif risk_level == "MODÉRÉ":
                    st.markdown("### 🟡 MODÉRÉ")
                else:
                    st.markdown("### 🔴 ÉLEVÉ")
                
                st.metric(
                    "Seuil de décision",
                    f"{threshold*100:.0f}%",
                    help="Au-dessus du seuil, le crédit est refusé"
                )
            
            if decision == "ACCORDER":
                st.success(f"""
                ✅ **Crédit accordable.**  
                La probabilité de défaut estimée ({proba*100:.1f}%) est en dessous du seuil de décision ({threshold*100:.0f}%).
                """)
            else:
                st.error(f"""
                ❌ **Crédit à refuser.**  
                La probabilité de défaut estimée ({proba*100:.1f}%) dépasse le seuil de décision ({threshold*100:.0f}%).
                """)
            
            with st.expander("🔍 Détails techniques"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("##### Réponse de l'API")
                    st.json(result)
                with col_b:
                    st.markdown("##### Données envoyées")
                    st.json(client_data)
        
        else:
            st.error(f"❌ Erreur de l'API (code {response.status_code})")
            st.json(response.json())
    
    except requests.exceptions.ConnectionError:
        st.error("""
        ❌ **Impossible de contacter l'API**
        
        Vérifie que l'API FastAPI tourne sur `http://127.0.0.1:8000`.
        Pour la lancer : `uvicorn src.api.main:app --reload`
        """)
    except requests.exceptions.Timeout:
        st.error("⏱️ L'API a mis trop de temps à répondre.")
    except Exception as e:
        st.error(f"❌ Erreur inattendue : {str(e)}")


# === Footer ===
st.divider()
st.caption("🛠️ Projet MLOps — David | Master 2 Data & AI | Tech Lead: AIDARA CHAMSEDINE")