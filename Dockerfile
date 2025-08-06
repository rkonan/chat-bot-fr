# Image de base Python avec support CPU
FROM python:3.10-slim

# Dépendances système nécessaires à llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libsqlite3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Définir le dossier de travail
WORKDIR /code

# Copier le fichier de dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code dans l’image
COPY . .

# Définir le port (pour FastAPI ou Streamlit)
EXPOSE 7860

# Commande de démarrage (adapte selon ton app : Streamlit, FastAPI...)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
