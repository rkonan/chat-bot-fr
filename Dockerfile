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

# 📥 Étape 4 : copier la wheel précompilée AVANT requirements
COPY wheels/ ./wheels/

# ⚡ Étape 5 : install pip + llama-cpp-python via wheel locale
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir ./wheels/llama_cpp_python-0.3.14-*.whl

# 📄 Étape 6 : copier le requirements sans llama-cpp-python
COPY requirements.txt .

# 📦 Étape 7 : installer le reste
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# 📁 Étape 8 : copier le reste du code de l'app
COPY . .

# 🌐 Étape 9 : exposer le port Streamlit
EXPOSE 7860

# 🚀 Étape 10 : lancer l'application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
