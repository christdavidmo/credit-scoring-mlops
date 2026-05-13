


# === Étape 1 : image de base ===
# Python 3.12 slim = version légère sans les outils inutiles
FROM python:3.12-slim

# === Étape 2 : dossier de travail dans le conteneur ===
WORKDIR /app

# === Étape 3 : installer les dépendances système (légères) ===
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# === Étape 4 : copier et installer les dépendances Python ===
# (on copie d'abord requirements pour bénéficier du cache Docker)
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# === Étape 5 : copier le code source ===
COPY src/ ./src/
COPY models/xgboost_champion.pkl ./models/xgboost_champion.pkl
COPY models/preprocessor.pkl ./models/preprocessor.pkl

# === Étape 6 : exposer le port ===
EXPOSE 8000

# === Étape 7 : commande de démarrage ===
# Render fournit la variable PORT dynamiquement
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}