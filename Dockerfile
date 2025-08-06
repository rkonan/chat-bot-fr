# Image de base Python avec support CPU
FROM python:3.10-slim

# Dépendances système nécessaires à llama-cpp-python et nltk
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libsqlite3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Définir le dossier de travail
WORKDIR /code

# Copier les fichiers requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# 🔧 Créer dossier pour NLTK
RUN mkdir -p /tmp/nltk_data

# ✅ Télécharger punkt AVANT le lancement de l'app
RUN python -m nltk.downloader -d /tmp/nltk_data punkt

# Copier le reste du code
COPY . .

# Exposer le port pour Streamlit
EXPOSE 7860

# CMD adapté pour Hugging Face Spaces
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
