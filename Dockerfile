# Image de base légère
FROM python:3.11-slim

# Éviter les prompts interactifs et optimiser pip
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

WORKDIR /app

# Déps système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Code
COPY . .

# Changer le répertoire de travail vers src
WORKDIR /app/src

# Port Streamlit
EXPOSE 8501

# Chemin par défaut (peut être surchargé à l'exécution)
ENV APP_PATH=b.py

# Commande de démarrage
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]
