import os
import logging
import streamlit as st
import requests
import json

# --- Config & logs ---
os.environ.setdefault("NLTK_DATA", "/home/appuser/nltk_data")
logger = logging.getLogger("Streamlit")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

st.set_page_config(page_title="Chat Ollama", page_icon="🤖")

# --- UI Sidebar ---
st.sidebar.header("⚙️ Paramètres")
default_host = os.getenv("OLLAMA_HOST", "http://localhost:11435")
ollama_host = st.sidebar.text_input("Ollama host", value=default_host)
model_name = st.sidebar.text_input("Modèle Ollama", value="qwen2.5:3b-instruct-q4_K_M")

st.title("💬 Chat Ollama (simple)")

# --- Historique ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.text_area("Votre message :", height=100, placeholder="Ex: Bonjour ?")
col1, col2 = st.columns([1, 1])

# --- Fonction d'appel API /api/chat ---
def ollama_chat(messages, stream=False):
    url = ollama_host.rstrip("/") + "/api/chat"
    payload = {"model": model_name, "messages": messages, "stream": stream}

    if stream:
        # renvoie un générateur de tokens
        def token_gen():
            with requests.post(url, json=payload, stream=True, timeout=300) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    data = json.loads(line)
                    if "message" in data and not data.get("done"):
                        yield data["message"]["content"]
                    if data.get("done"):
                        break
        return token_gen()
    else:
        # renvoie directement la réponse complète (dict)
        r = requests.post(url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()

# --- Bouton : envoi normal ---
if col1.button("Envoyer"):
    if user_input.strip():
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.spinner("Génération en cours..."):
            try:
                result = ollama_chat(st.session_state["messages"], stream=False)
                content = result.get("message", {}).get("content", "")
                st.session_state["messages"].append({"role": "assistant", "content": content})
                st.markdown("**Réponse :**")
                st.success(content)
                st.write(f"⏱ Temps total : {result['total_duration']/1e9:.2f}s")
                st.write(f"📝 Tokens prompt : {result['prompt_eval_count']}, génération : {result['eval_count']}")

            except Exception as e:
                st.error(f"Erreur: {e}")
    else:
        st.info("Saisissez un message.")

# --- Bouton : envoi streaming ---
if col2.button("Envoyer (stream)"):
    if user_input.strip():
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.spinner("Génération en cours (stream)..."):
            try:
                ph = st.empty()
                acc = ""
                for token in ollama_chat(st.session_state["messages"], stream=True):
                    acc += token
                    ph.markdown(acc)
                st.session_state["messages"].append({"role": "assistant", "content": acc})
            except Exception as e:
                st.error(f"Erreur (stream): {e}")
    else:
        st.info("Saisissez un message.")

# --- Affichage historique ---
st.subheader("Historique de la conversation")
for msg in st.session_state["messages"]:
    role = "🧑‍💻" if msg["role"] == "user" else "🤖"
    st.markdown(f"{role} **{msg['role']}**: {msg['content']}")
