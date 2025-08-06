FROM python:3.10-slim

# Dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libsqlite3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Créer utilisateur non-root
RUN useradd -m appuser

# Dossier de travail
WORKDIR /code

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🔧 Créer le dossier NLTK + télécharger tous les packages nécessaires
RUN mkdir -p /home/appuser/nltk_data && \
    python -m nltk.downloader -d /home/appuser/nltk_data punkt stopwords

# Copier le code
COPY . .

# Donner les droits
RUN chown -R appuser /code /home/appuser/nltk_data

# Utiliser utilisateur non-root
USER appuser

# Définir variable d'environnement
ENV NLTK_DATA=/home/appuser/nltk_data

# Port pour Streamlit
EXPOSE 7860

# Commande de lancement
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
