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

# 👤 Créer un utilisateur non-root
RUN useradd -m appuser

# 📁 Dossier de travail
WORKDIR /code

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Télécharger punkt dans un dossier propre
RUN mkdir -p /home/appuser/nltk_data && \
    python -m nltk.downloader -d /home/appuser/nltk_data punkt

# Copier le reste du code
COPY . .

# Donner les droits à appuser sur le code
RUN chown -R appuser /code

# Utiliser l'utilisateur non-root
USER appuser

# ✅ Définir la variable d'environnement pour nltk
ENV NLTK_DATA=/home/appuser/nltk_data

# Exposer le port Streamlit
EXPOSE 7860

# Démarrer l'application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
