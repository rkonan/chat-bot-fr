# 🐍 Image de base
FROM python:3.10-slim

# 🧱 Dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libsqlite3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 📁 Dossier de travail
WORKDIR /code

# 📝 Copier les requirements et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 📦 Préparer le cache NLTK
RUN mkdir -p /tmp/nltk_data && python -m nltk.downloader -d /tmp/nltk_data punkt

# 📁 Copier tout le code
COPY . .

# 📤 Exposer le port Streamlit
EXPOSE 7860

# 🚀 Lancer l'application (le patch doit être dans app.py AVANT l'import llama_index)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
