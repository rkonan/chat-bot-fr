import os
import logging
import streamlit as st
from huggingface_hub import hf_hub_download

from rag_model_ollama import RAGEngine

# --- Config & logs ---
os.environ.setdefault("NLTK_DATA", "/home/appuser/nltk_data")
logger = logging.getLogger("Streamlit")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

st.set_page_config(page_title="Chatbot RAG (Ollama)", page_icon="🤖")

# --- ENV ---
ENV = os.getenv("ENV", "local")
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

# --- Sidebar ---
st.sidebar.header("⚙️ Paramètres")
default_host = os.getenv("OLLAMA_HOST", "http://localhost:11435")  # via proxy mitm si besoin
ollama_host = st.sidebar.text_input("Ollama host", value=default_host)
suggested_models = [
    "qwen2.5:3b-instruct-q4_K_M",
    "mistral",
    "gemma3",
    "deepseek-r1",
    "granite3.3",
    "llama3.1:8b-instruct-q4_K_M",
]
model_name = st.sidebar.selectbox("Modèle Ollama", options=suggested_models, index=0)
rag_mode = st.sidebar.selectbox("Mode RAG", options=["Auto", "RAG", "LLM pur"], index=0)

st.title("🤖 Chatbot RAG Local (Ollama) — lazy + meta + toggle")

# --- Cache du moteur ---
@st.cache_resource(show_spinner=True)
def load_rag_engine(_model_name: str, _host: str):
    os.environ["OLLAMA_KEEP_ALIVE"] = "15m"
    return RAGEngine(
        model_name=_model_name,
        vector_path=vectors_path,
        index_path=faiss_index_path,
        model_threads=4,
        ollama_host=_host
    )

rag = load_rag_engine(model_name, ollama_host)
# appliquer le mode sélectionné
rag.set_mode({"Auto": "auto", "RAG": "rag", "LLM pur": "llm"}[rag_mode])

# --- Chat ---
user_input = st.text_area("Posez votre question :", height=120,
                          placeholder="Ex: Quels sont les traitements appliqués aux images ?")
col1, col2 = st.columns([1, 1])

if col1.button("Envoyer (non-stream)"):
    if user_input.strip():
        with st.spinner("Génération en cours..."):
            try:
                response = rag.ask(user_input)
                st.markdown("**Réponse :**")
                st.success(response)

                # Affiche les sources si RAG a été utilisé
                sources = rag.get_last_sources()
                if sources:
                    with st.expander("📚 Sources (RAG)"):
                        for i, s in enumerate(sources, 1):
                            st.markdown(
                                f"**{i}.** {s.get('doc','?')} "
                                f"(page {s.get('page','?')}) — *{s.get('title') or 'Sans titre'}*"
                            )
                            st.code(s.get("preview", ""), language="markdown")
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

                # Affiche les sources si RAG a été utilisé
                sources = rag.get_last_sources()
                if sources:
                    with st.expander("📚 Sources (RAG)"):
                        for i, s in enumerate(sources, 1):
                            st.markdown(
                                f"**{i}.** {s.get('doc','?')} "
                                f"(page {s.get('page','?')}) — *{s.get('title') or 'Sans titre'}*"
                            )
                            st.code(s.get("preview", ""), language="markdown")

            except Exception as e:
                st.error(f"Erreur pendant la génération (stream): {e}")
    else:
        st.info("Saisissez une question.")
