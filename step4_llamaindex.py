import os
import signal
import sys
import pickle
import faiss
import numpy as np
import textwrap

from llama_cpp import Llama
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers.util import cos_sim



MAX_TOKENS=512
mode_type=[ "docling"]
#mode_type=["sentence", "docling"]
def reformulate_question(llm, question: str) -> str:
    prompt = f"""Tu es un assistant expert chargé de clarifier des questions floues.

Transforme la question suivante en une question claire, explicite et complète, sans ajouter d'informations extérieures.

Question floue : {question}
Question reformulée :"""
    
    output = llm(prompt, max_tokens=128, stop=["\n"], stream=False)
    return output["choices"][0]["text"].strip()


def reformulate_with_context_v0(llm, question: str, context_sample: str) -> str:
    prompt = f"""Tu es un assistant expert qui reformule les questions utilisateur en tenant compte du contexte d'un document.

Ton objectif est de transformer une question floue, vague ou incomplète en une question claire, explicite et pertinente par rapport au contexte ci-dessous.

Contexte :
{context_sample}

Question initiale : {question}
Question reformulée :"""

    output = llm(prompt, max_tokens=128, stop=["\n"], stream=False)
    return output["choices"][0]["text"].strip()



def reformulate_with_context(llm, question: str, context_sample: str) -> str:
    prompt = f"""Tu es un assistant expert en machine learning. Ton rôle est de reformuler les questions utilisateur en tenant compte du contexte ci-dessous, extrait d’un rapport technique sur un projet de reconnaissance de maladies de plantes.

Ta mission est de transformer une question vague ou floue en une question précise et adaptée au contenu du rapport. Ne donne pas une interprétation hors sujet. Ne reformule pas en termes de produits commerciaux.

Contexte :
{context_sample}

Question initiale : {question}
Question reformulée :"""
    
    output = llm(prompt, max_tokens=128, stop=["\n"], stream=False)
    return output["choices"][0]["text"].strip()



# 🔁 top_k adaptatif
def get_adaptive_top_k(question: str) -> int:
    q = question.lower()
    if len(q.split()) <= 7:
        return 8
    elif any(w in q for w in ["liste", "résume", "quels sont", "explique", "comment"]):
        return 10
    else:
        return 8

def rerank_nodes(question, retrieved_nodes, embed_model, top_k=3):
    print(f"\n🔍 Re-ranking des {len(retrieved_nodes)} chunks pour la question : « {question} »\n")
    q_emb = embed_model.get_query_embedding(question)
    scored_nodes = []

    for node in retrieved_nodes:
        chunk_text = node.get_content()
        chunk_emb = embed_model.get_text_embedding(chunk_text)
        score = cos_sim(q_emb, chunk_emb).item()
        scored_nodes.append((score, node))

    ranked_nodes = sorted(scored_nodes, key=lambda x: x[0], reverse=True)
    print(f"📊 Top {top_k} chunks les plus pertinents :\n")
    for rank, (score, node) in enumerate(ranked_nodes[:top_k], start=1):
        chunk_preview = textwrap.shorten(node.get_content().replace("\n", " "), width=150)
        print(f"#{rank:>2} | Score: {score:.4f} | {chunk_preview}")

    return [n for _, n in ranked_nodes[:top_k]]


def ask_llm(question: str, retriever, embed_model, top_k=3) -> str:
    retrieved_nodes = retriever.retrieve(question)
    top_nodes = rerank_nodes(question, retrieved_nodes, embed_model, top_k)
    context = "\n\n".join(n.get_content()[:500] for n in top_nodes)

    prompt = f"""### Instruction: En te basant uniquement sur le contexte ci-dessous, réponds à la question de manière précise et en français.

Si la réponse ne peut pas être déduite du contexte, indique : "Information non présente dans le contexte."

Contexte :
{context}

Question : {question}
### Réponse:"""

    output = llm(prompt, max_tokens=MAX_TOKENS, stop=["### Instruction:"], stream=False)
    return output["choices"][0]["text"].strip().split("###")[0]



def ask_llm_stream(question: str, retriever, embed_model, top_k=3) :
    retrieved_nodes = retriever.retrieve(question)
    top_nodes = rerank_nodes(question, retrieved_nodes, embed_model, top_k)
    context = "\n\n".join(n.get_content()[:500] for n in top_nodes)

    prompt = f"""### Instruction: En te basant uniquement sur le contexte ci-dessous, réponds à la question de manière précise et en français.

Si la réponse ne peut pas être déduite du contexte, indique : "Information non présente dans le contexte."

Contexte :
{context}

Question : {question}
### Réponse:"""

    stream = llm(prompt, max_tokens=MAX_TOKENS, stop=["### Instruction:"], stream=True)

    for chunk in stream:
        print(chunk["choices"][0]["text"], end="", flush=True)

    return 

# 📦 Chargement global une seule fois

MODEL_PATH = "models/Nous-Hermes-2-Mistral-7B-DPO.Q4_K_M.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4)

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

INDEXES = {}

for mode in mode_type:
    vectordir = f"vectordb_{mode}" if mode != "sentence" else "vectordb"
    index_file = os.path.join(vectordir, "index.faiss")
    chunks_file = os.path.join(vectordir, "chunks.pkl")

    print(f"📂 Chargement {mode} depuis {vectordir}...")
    with open(chunks_file, "rb") as f:
        chunk_texts = pickle.load(f)
    nodes = [TextNode(text=chunk) for chunk in chunk_texts]

    faiss_index = faiss.read_index(index_file)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    index = VectorStoreIndex(nodes=nodes, embed_model=embed_model, vector_store=vector_store)

    INDEXES[mode] = {
        "nodes": nodes,
        "index": index,
    }


# 🧠 Interface CLI
def exit_gracefully(sig, frame):
    print("\n👋 Au revoir !")
    sys.exit(0)

signal.signal(signal.SIGINT, exit_gracefully)

print("🧠 RAG CLI interactif avec LlamaIndex (CTRL+C pour quitter)")

while True:
    question_raw = input("\n❓> ").strip()
    if not question_raw:
        continue
    


   

    docling_mode = input("\n⚙️  Utiliser Docling ? (o/n) : ").strip().lower()
    mode = "docling" if docling_mode in ["o", "oui", "y", "yes"] else "sentence"

    print(f"\n📂 Mode sélectionné : {mode}")

    index_obj = INDEXES[mode]["index"]
    

    if len(question_raw.split()) <= 3:
        retriever= index_obj.as_retriever(similarity_top_k=3)
        retrieved_nodes = retriever.retrieve(question_raw)
        context_sample = "\n\n".join(n.get_content()[:500] for n in retrieved_nodes[:3])
        reformulated = reformulate_with_context(llm, question_raw, context_sample)
    else:
        reformulated = reformulate_question(llm, question_raw)

    print(f"📝 Question reformulée : {reformulated}")
    question = reformulated

    top_k = get_adaptive_top_k(question)
    print(f"🔍 top_k = {top_k} adapté à la question")

    
    retriever = index_obj.as_retriever(similarity_top_k=top_k)

    # response = ask_llm(question, retriever, embed_model, top_k=top_k)
    # print(f"\n💬 Réponse : {response}")

    ask_llm_stream(question, retriever, embed_model, top_k=top_k)
