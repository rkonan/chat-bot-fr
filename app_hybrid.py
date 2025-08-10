import os
import logging
import streamlit as st
from huggingface_hub import hf_hub_download
from rag_engine_hybrid import RAGEngine

# Logs
os.environ.setdefault("NLTK_DATA", "/home/appuser/nltk_data")
logger = logging.getLogger("Streamlit")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)

st.set_page_config(page_title="Chat (Hybride RAG + Chat)", page_icon="🤖")

# Données
ENV = os.getenv("ENV", "local")
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

# Sidebar
st.sidebar.header("⚙️ Paramètres")
default_host = os.getenv("OLLAMA_HOST", "http://localhost:11435")
ollama_host = st.sidebar.text_input("Ollama host", value=default_host)
model_name = st.sidebar.text_input("Modèle", value="qwen2.5:3b-instruct-q4_K_M")

# (Optionnel) Mode RAG forcé — désactivé ici, on laisse l’heuristique faire son travail.
st.title("🤖 Chat Hybride")

@st.cache_resource(show_spinner=True)
def load_engine(_model: str, _host: str):
    os.environ["OLLAMA_KEEP_ALIVE"] = "15m"
    return RAGEngine(
        model_name=_model,
        vector_path=vectors_path,
        index_path=faiss_index_path,
        model_threads=4,
        ollama_host=_host
    )

rag = load_engine(model_name, ollama_host)

# UI
user_input = st.text_area("Votre message :", height=120, placeholder="Ex: 'Quelle est la capitale de la France ?'")
col1, col2 = st.columns([1, 1])

if col1.button("Envoyer"):
    if user_input.strip():
        with st.spinner("Génération en cours..."):
            try:
                ans = rag.ask(user_input)
                st.markdown("**Réponse :**")
                st.success(ans)
            except Exception as e:
                st.error(f"Erreur: {e}")
    else:
        st.info("Saisissez un message.")

if col2.button("Envoyer (stream)"):
    if user_input.strip():
        with st.spinner("Génération en cours (stream)..."):
            try:
                ph = st.empty()
                acc = ""
                for tok in rag.ask_stream(user_input):
                    acc += tok
                    ph.markdown(acc)
            except Exception as e:
                st.error(f"Erreur (stream): {e}")
    else:
        st.info("Saisissez un message.")
