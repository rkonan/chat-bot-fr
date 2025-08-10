import os
import pickle
import textwrap
import logging
from typing import List

import faiss
import numpy as np
from llama_cpp import Llama
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers.util import cos_sim

# === Logger configuration ===
logger = logging.getLogger("RAGEngine")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

MAX_TOKENS = 512

class RAGEngine:
    def __init__(self, model_path: str, vector_path: str, index_path: str, model_threads: int = 4):
        logger.info("📦 Initialisation du moteur RAG...")
        self.llm = Llama(model_path=model_path, n_ctx=2048, n_threads=model_threads)
        self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


        logger.info(f"📂 Chargement des données vectorielles depuis {vector_path}")
        with open(vector_path, "rb") as f:
            chunk_texts = pickle.load(f)
        nodes = [TextNode(text=chunk) for chunk in chunk_texts]

        faiss_index = faiss.read_index(index_path)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        self.index = VectorStoreIndex(nodes=nodes, embed_model=self.embed_model, vector_store=vector_store)

        logger.info("✅ Moteur RAG initialisé avec succès.")

    def reformulate_question(self, question: str) -> str:
        logger.info("🔁 Reformulation de la question (sans contexte)...")
        prompt = f"""Tu es un assistant expert chargé de clarifier des questions floues.

Transforme la question suivante en une question claire, explicite et complète, sans ajouter d'informations extérieures.

Question floue : {question}
Question reformulée :"""
        output = self.llm(prompt, max_tokens=128, stop=["\n"], stream=False)
        reformulated = output["choices"][0]["text"].strip()
        logger.info(f"📝 Reformulée : {reformulated}")
        return reformulated

    def reformulate_with_context(self, question: str, context_sample: str) -> str:
        logger.info("🔁 Reformulation de la question avec contexte...")
        prompt = f"""Tu es un assistant expert en machine learning. Ton rôle est de reformuler les questions utilisateur en tenant compte du contexte ci-dessous, extrait d’un rapport technique sur un projet de reconnaissance de maladies de plantes.

Ta mission est de transformer une question vague ou floue en une question précise et adaptée au contenu du rapport. Ne donne pas une interprétation hors sujet. Ne reformule pas en termes de produits commerciaux.

Contexte :
{context_sample}

Question initiale : {question}
Question reformulée :"""
        output = self.llm(prompt, max_tokens=128, stop=["\n"], stream=False)
        reformulated = output["choices"][0]["text"].strip()
        logger.info(f"📝 Reformulée avec contexte : {reformulated}")
        return reformulated

    def get_adaptive_top_k(self, question: str) -> int:
        q = question.lower()
        if len(q.split()) <= 7:
            top_k = 8
        elif any(w in q for w in ["liste", "résume", "quels sont", "explique", "comment"]):
            top_k = 10
        else:
            top_k = 8
        logger.info(f"🔢 top_k déterminé automatiquement : {top_k}")
        return top_k

    def rerank_nodes(self, question: str, retrieved_nodes, top_k: int = 3):
        logger.info(f"🔍 Re-ranking des {len(retrieved_nodes)} chunks pour la question : « {question} »")
        q_emb = self.embed_model.get_query_embedding(question)
        scored_nodes = []

        for node in retrieved_nodes:
            chunk_text = node.get_content()
            chunk_emb = self.embed_model.get_text_embedding(chunk_text)
            score = cos_sim(q_emb, chunk_emb).item()
            scored_nodes.append((score, node))

        ranked_nodes = sorted(scored_nodes, key=lambda x: x[0], reverse=True)

        logger.info("📊 Chunks les plus pertinents :")
        for i, (score, node) in enumerate(ranked_nodes[:top_k]):
            chunk_preview = textwrap.shorten(node.get_content().replace("\n", " "), width=100)
            logger.info(f"#{i+1} | Score: {score:.4f} | {chunk_preview}")

        return [n for _, n in ranked_nodes[:top_k]]

    def retrieve_context(self, question: str, top_k: int = 3):
        logger.info(f"📥 Récupération du contexte...")
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        retrieved_nodes = retriever.retrieve(question)
        reranked_nodes = self.rerank_nodes(question, retrieved_nodes, top_k)
        context = "\n\n".join(n.get_content()[:500] for n in reranked_nodes)
        return context, reranked_nodes

    def ask(self, question_raw: str) -> str:
        logger.info(f"💬 Question reçue : {question_raw}")
        if len(question_raw.split()) <= 100:
            context_sample, _ = self.retrieve_context(question_raw, top_k=3)
            reformulated = self.reformulate_with_context(question_raw, context_sample)
        else:
            reformulated = self.reformulate_question(question_raw)

        logger.info(f"📝 Question reformulée : {reformulated}")
        top_k = self.get_adaptive_top_k(reformulated)
        context, _ = self.retrieve_context(reformulated, top_k)

        prompt = f"""### Instruction: En te basant uniquement sur le contexte ci-dessous, réponds à la question de manière précise et en français.

Si la réponse ne peut pas être déduite du contexte, indique : "Information non présente dans le contexte."

Contexte :
{context}

Question : {reformulated}
### Réponse:"""

        output = self.llm(prompt, max_tokens=MAX_TOKENS, stop=["### Instruction:"], stream=False)
        response = output["choices"][0]["text"].strip().split("###")[0]
        logger.info(f"🧠 Réponse générée : {response[:120]}{'...' if len(response) > 120 else ''}")
        return response

    def ask_stream(self, question: str):
        logger.info(f"💬 [Stream] Question reçue : {question}")
        top_k = self.get_adaptive_top_k(question)
        context, _ = self.retrieve_context(question, top_k)

        prompt = f"""### Instruction: En te basant uniquement sur le contexte ci-dessous, réponds à la question de manière précise et en français.

Si la réponse ne peut pas être déduite du contexte, indique : "Information non présente dans le contexte."

Contexte :
{context}

Question : {question}
### Réponse:"""

        logger.info("📡 Début du streaming de la réponse...")
        stream = self.llm(prompt, max_tokens=MAX_TOKENS, stop=["### Instruction:"], stream=True)
        for chunk in stream:
            print(chunk["choices"][0]["text"], end="", flush=True)
