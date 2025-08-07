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

# 👤 Étape 3 : Créer un utilisateur non-root "appuser"
RUN useradd -ms /bin/bash appuser

# 📁 Étape 4 : Créer le dossier de travail avec droits pour appuser
WORKDIR /code
RUN mkdir -p /code/.streamlit && chown -R appuser:appuser /code

# ⚠️ Étape 5 : définir le HOME pour Streamlit
ENV HOME="/code"

# 📥 Étape 6 : copier les wheels (llama-cpp-python précompilée)
COPY wheels/ ./wheels/

# ⚡ Étape 7 : installer pip et la wheel locale
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir ./wheels/llama_cpp_python-0.3.14-*.whl

# 📄 Étape 8 : copier requirements.txt (sans llama-cpp-python dedans)
COPY requirements.txt .

# 📦 Étape 9 : installer le reste des dépendances
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# 📁 Étape 10 : copier tout le code de l'application
COPY . .

# 👤 Étape 11 : basculer en utilisateur non-root pour exécution
USER appuser

# 🌐 Étape 12 : exposer le port Streamlit
EXPOSE 7860

# 🚀 Étape 13 : lancer Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
