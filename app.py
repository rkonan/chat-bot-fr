import streamlit as st 
from llama_cpp import Llama
import os 

st.set_page_config(page_title="Chatbot RAG local",page_icon="🤖")

@st.cache_resource
def load_model():
    model_path="models/phi-2.Q4_K_M.gguf"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    return Llama(model_path=model_path,n_ctx=2048,n_threads=4)

llm=load_model()

st.title("🤖 Chatbot LLM Local (CPU)")

user_input=st.text_area("Posez votre question :", height=100)

if st.button("Envoyer") and user_input.strip():
       with st.spinner("Génération en cours..."):
            full_prompt = f"### Instruction: {user_input.strip()}\n### Réponse:"
            #full_prompt = f"Question: {user_input.strip()}\nAnswer:"
            #output = llm(full_prompt, max_tokens=100, stop=["Question:", "Answer:", "\n\n"])
            output = llm(full_prompt, max_tokens=150, stop=["### Instruction:"])
            #output = llm(full_prompt, max_tokens=80)
            #response = output["choices"][0]["text"]
            response = output["choices"][0]["text"].strip()
            response = response.split("### Instruction:")[0].strip()

            st.markdown("**Réponse :**")
            st.success(response)