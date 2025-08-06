import streamlit as st 
from llama_cpp import Llama
import os 
from rag_model import RAGEngine
import logging
from huggingface_hub import hf_hub_download
import time


import os
os.environ["NLTK_DATA"] = "/tmp/nltk_data"

# Appliquer le patch avant tout import de llama_index
from patches.llama_patch import patch_llamaindex_nltk
patch_llamaindex_nltk()

logger = logging.getLogger("Streamlit")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)



logger.info(f"ENV :{ENV}")

#time.sleep(5)

if ENV == "local":
    model_path = "chatbot-models/Nous-Hermes-2-Mistral-7B-DPO.Q4_K_M.gguf"
    faiss_index_path="chatbot-models/vectordb_docling/index.faiss"
    vectors_path="chatbot-models/vectordb_docling/chunks.pkl"
    
else:
  # Télécharger le modèle GGUF
    model_path = hf_hub_download(
        repo_id="rkonan/chatbot-models",
        filename="chatbot-models/Nous-Hermes-2-Mistral-7B-DPO.Q4_K_M.gguf",
        repo_type="dataset"
    )

    # Télécharger les fichiers FAISS
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




st.set_page_config(page_title="Chatbot RAG local",page_icon="🤖")





@st.cache_resource
def load_rag_engine():
    rag = RAGEngine(model_path,vectors_path,faiss_index_path)
    return rag

rag=load_rag_engine()

st.title("🤖 Chatbot LLM Local (CPU)")

user_input=st.text_area("Posez votre question :", height=100)

if st.button("Envoyer") and user_input.strip():
       with st.spinner("Génération en cours..."):
            response = rag.ask(user_input)
            st.markdown("**Réponse :**")
            st.success(response)