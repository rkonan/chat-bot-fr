import os
import logging
import streamlit as st
from huggingface_hub import hf_hub_download

# ✅ Nouveau moteur RAG (Ollama)
from rag_model_ollama_v1 import RAGEngine

# --- Config & logs ---
os.environ.setdefault("NLTK_DATA", "/home/appuser/nltk_data")
os.environ["OLLAMA_KEEP_ALIVE"] = "15m"  # garde le modèle chaud

logger = logging.getLogger("Streamlit")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

st.set_page_config(page_title="Chatbot RAG (Ollama)", page_icon="🤖")

# --- ENV ---
ENV = os.getenv("ENV", "local")  # "local" ou "space"
logger.info(f"ENV: {ENV}")

# --- Chemins FAISS & chunks ---
if ENV == "local":
    faiss_index_path = "chatbot-models/vectordb_docling/index.faiss"
    vectors_path = "chatbot-models/vectordb_docling/chunks.pkl"
else:
    faiss_index_path = hf_hub_download(
        repo_id="rkonan/chatbot-models",
        filename="chatbot-models/vectordb_docling/index.faiss",
        repo_type="dataset"
    )
    vectors_path = hf_hub_download(
        repo_id="rkonan/chatbot-models",
        filename="chatbot-models/vectordb_docling/chunks.pkl",
        repo_type="dataset"
    )

# --- UI Sidebar ---
st.sidebar.header("⚙️ Paramètres")
default_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_host = st.sidebar.text_input("Ollama host", value=default_host)

suggested_models = [
    "qwen2.5:3b-instruct-q4_K_M",
    "noushermes_rag",
    "mistral",
    "gemma3",
    "deepseek-r1",
    "granite3.3",
    "llama3.1:8b-instruct-q4_K_M",
    "nous-hermes2:Q4_K_M",
]
model_name = st.sidebar.selectbox("Modèle Ollama", options=suggested_models, index=0)
num_threads = st.sidebar.slider("Threads", min_value=2, max_value=16, value=6, step=1)
temperature = st.sidebar.slider("Température", min_value=0.0, max_value=1.5, value=0.1, step=0.1)

st.title("🤖 Chatbot RAG Local (Ollama)")

# --- Cache du moteur ---
@st.cache_resource(show_spinner=True)
def load_rag_engine(_model_name: str, _host: str, _threads: int, _temp: float):
    ollama_opts = {
        "num_thread": int(_threads),
        "temperature": float(_temp),
        "num_ctx": 512,   # identique au CLI
        "num_batch": 16,
    }

    rag = RAGEngine(
        model_name=_model_name,
        vector_path=vectors_path,
        index_path=faiss_index_path,
        model_threads=_threads,
        ollama_host=_host,
        ollama_opts=ollama_opts
    )

    # Warmup proche du CLI
    try:
        list(rag._complete_stream("Bonjour", max_tokens=8))
    except Exception as e:
        logger.warning(f"Warmup Ollama échoué: {e}")

    return rag

rag = load_rag_engine(model_name, ollama_host, num_threads, temperature)

# --- Chat ---
user_input = st.text_area("Posez votre question :", height=120,
                          placeholder="Ex: Quels sont les traitements appliqués aux images ?")
col1, col2 = st.columns([1, 1])

if col1.button("Envoyer"):
    if user_input.strip():
        with st.spinner("Génération en cours..."):
            try:
                response = rag.ask(user_input)
                st.markdown("**Réponse :**")
                st.success(response)
            except Exception as e:
                st.error(f"Erreur pendant la génération: {e}")
    else:
        st.info("Saisissez une question.")

if col2.button("Envoyer (stream)"):
    if user_input.strip():
        with st.spinner("Génération en cours (stream)..."):
            try:
                ph = st.empty()
                acc = ""
                for token in rag.ask_stream(user_input):
                    acc += token
                    ph.markdown(acc)
                st.balloons()
            except Exception as e:
                st.error(f"Erreur pendant la génération (stream): {e}")
    else:
        st.info("Saisissez une question.")
