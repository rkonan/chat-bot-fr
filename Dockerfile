# Utilise Python 3.12 slim
FROM python:3.12-slim

# Dépendances système minimales
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    libopenblas-dev \
    libsqlite3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dossier de travail
WORKDIR /code

# Copie des requirements hors llama-cpp
COPY requirements.txt .

# Install rapide de tout sauf llama-cpp-python (optionnel si inclus dans wheels)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copier les wheels (que tu as déjà créées en local)
COPY wheels/ ./wheels/

# Installer les .whl précompilés pour éviter le build (crucial sur HF Spaces)
RUN pip install --no-cache-dir \
    ./wheels/llama_cpp_python-0.3.14-*.whl \
    ./wheels/numpy-2.3.2-*.whl \
    ./wheels/typing_extensions-4.14.1-*.whl \
    ./wheels/diskcache-5.6.3-*.whl \
    ./wheels/jinja2-3.1.6-*.whl \
    ./wheels/MarkupSafe-3.0.2-*.whl

# Copier le reste du code
COPY . .

# Exposer le port Streamlit
EXPOSE 7860

# Lancement Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
