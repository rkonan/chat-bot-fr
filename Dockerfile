# ✅ Base image Python 3.12
FROM python:3.12-slim

# ✅ Installer dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libsqlite3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ✅ Dossier de travail
WORKDIR /code

# ✅ Copier le fichier de requirements minimal (runtime only)
COPY requirements-base.txt .

# ✅ Forcer installation binaire pour éviter les compilations longues
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --only-binary=:all: --prefer-binary -r requirements-base.txt

# ✅ Télécharger les ressources NLTK dans un dossier accessible
RUN mkdir -p /root/nltk_data && \
    python -m nltk.downloader -d /root/nltk_data punkt stopwords

# ✅ Définir la variable d'environnement pour NLTK
ENV NLTK_DATA=/root/nltk_data

# ✅ Copier le code
COPY . .

# ✅ Exposer le port (Streamlit)
EXPOSE 7860

# ✅ Commande de démarrage (adaptée Hugging Face)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
