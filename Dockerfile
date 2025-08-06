# ⚙️ Étape 1 : Image de base stable avec GLIBC + wheels compatibles
FROM python:3.10-slim

# 📦 Étape 2 : dépendances système minimales
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    curl \
    libopenblas-dev \
    libsqlite3-dev \
    libgl1 \
    libglib2.0-0 \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# 🏠 Fix Streamlit PermissionError (écriture dans /.streamlit interdite)
ENV HOME="/code"

# 📁 Étape 3 : répertoire de travail
WORKDIR /code

# 🔄 Étape 4 : copier les requirements
COPY requirements.txt .

# ⚡ Étape 5 : mise à jour pip + installation rapide
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --prefer-binary -r requirements.txt

# 📁 Étape 6 : copier tout le code de l'app
COPY . .

# 🌐 Étape 7 : exposer le port Streamlit
EXPOSE 7860

# 🚀 Étape 8 : lancer l'application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
