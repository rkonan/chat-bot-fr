import streamlit as st 
from llama_cpp import Llama
import os 
from rag_model import RAGEngine

st.set_page_config(page_title="Chatbot RAG local",page_icon="🤖")

@st.cache_resource
def load_rag_engine():
    rag = RAGEngine(model_path="models/Nous-Hermes-2-Mistral-7B-DPO.Q4_K_M.gguf")
    return rag

rag=load_rag_engine()

st.title("🤖 Chatbot LLM Local (CPU)")

user_input=st.text_area("Posez votre question :", height=100)

if st.button("Envoyer") and user_input.strip():
       with st.spinner("Génération en cours..."):
            response = rag.ask(user_input,mode="docling")
            st.markdown("**Réponse :**")
            st.success(response)