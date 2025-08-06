# ⚙️ Étape 1 : base Python 3.12 slim
FROM python:3.12-slim

# 📦 Étape 2 : installer les dépendances système minimales
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    curl \
    libopenblas-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# 📁 Étape 3 : définir le répertoire de travail
WORKDIR /code

# 📥 Étape 4 : copier les wheels précompilés (AVANT le pip install)
COPY wheels/ ./wheels/

# ⚡ Étape 5 : installer les wheels manuellement AVANT tout (évite les builds !)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --find-links=./wheels \
    llama-cpp-python==0.3.14 \
    numpy==1.26.4 \
    typing-extensions==4.14.1 \
    diskcache==5.6.3 \
    jinja2==3.1.6 \
    MarkupSafe==3.0.2

# 📄 Étape 6 : copier le requirements.txt (hors wheels)
COPY requirements.txt .

# 🔧 Étape 7 : installer les dépendances restantes via wheels locaux (si dispo)
RUN pip install --no-cache-dir --prefer-binary \
    --find-links=./wheels \
    -r requirements.txt

# 📦 Étape 8 : copier le code de l'app
COPY . .

# 🌐 Étape 9 : exposer le port utilisé par Streamlit
EXPOSE 7860

# 🚀 Étape 10 : lancer l'application Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
