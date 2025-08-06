# Base image avec Python 3.10
FROM python:3.10-slim

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libsqlite3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Créer un utilisateur non-root (important pour éviter les erreurs de permission nltk/llama)
RUN useradd -m appuser
USER appuser

# Créer le dossier de travail
WORKDIR /home/appuser/app

# Copier les fichiers nécessaires
COPY --chown=appuser:appuser requirements.txt .

# Installer les paquets Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Télécharger les ressources NLTK nécessaires dans un dossier accessible
RUN python -m nltk.downloader -d /home/appuser/nltk_data punkt stopwords

# Copier le code (en tant que appuser)
COPY --chown=appuser:appuser . .

# Définir la variable d'environnement pour nltk
ENV NLTK_DATA=/home/appuser/nltk_data

# Exposer le port Streamlit
EXPOSE 7860

# Commande de démarrage
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
