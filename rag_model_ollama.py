
import os
import pickle
import textwrap
import logging
from typing import List, Optional, Dict, Any, Iterable

import requests
import faiss
import numpy as np
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
if not logger.handlers:
    logger.addHandler(handler)

MAX_TOKENS = 512


class OllamaClient:
    """
    Minimal Ollama client for /api/generate (text completion) with streaming support.
    Docs: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
    """
    def __init__(self, model: str, host: Optional[str] = None, timeout: int = 120):
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout
        self._gen_url = self.host.rstrip("/") + "/api/generate"

    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> str | Iterable[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
        }
        if stop:
            payload["stop"] = stop
        if max_tokens is not None:
            # Ollama uses "num_predict" for max new tokens
            payload["num_predict"] = int(max_tokens)
        if options:
            payload["options"] = options

        logger.debug(f"POST {self._gen_url} (stream={stream})")

        if stream:
            with requests.post(self._gen_url, json=payload, stream=True, timeout=self.timeout) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        # In case a broken line appears
                        continue
                    if "response" in data and data.get("done") is not True:
                        yield data["response"]
                    if data.get("done"):
                        break
            return

        # Non-streaming
        r = requests.post(self._gen_url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")


# Lazy import json to keep top clean
import json


class RAGEngine:
    def __init__(
        self,
        model_name: str,
        vector_path: str,
        index_path: str,
        model_threads: int = 4,
        ollama_host: Optional[str] = None,
        ollama_opts: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            model_name: e.g. "nous-hermes2:Q4_K_M" or "llama3.1:8b-instruct-q4_K_M"
            vector_path: pickle file with chunk texts list[str]
            index_path: FAISS index path
            model_threads: forwarded to Ollama via options.n_threads (if supported by the model)
            ollama_host: override OLLAMA_HOST (default http://localhost:11434)
            ollama_opts: extra Ollama options (e.g., temperature, top_p, num_gpu, num_thread)
        """
        logger.info("📦 Initialisation du moteur RAG (Ollama)...")
        # Build options
        opts = dict(ollama_opts or {})
        # Common low-latency defaults; user can override via ollama_opts
        opts.setdefault("temperature", 0.1)
        # Try to pass thread hint if supported by the backend
        if "num_thread" not in opts and model_threads:
            opts["num_thread"] = int(model_threads)

        self.llm = OllamaClient(model=model_name, host=ollama_host)
        self.ollama_opts = opts

        self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

        logger.info(f"📂 Chargement des données vectorielles depuis {vector_path}")
        with open(vector_path, "rb") as f:
            chunk_texts = pickle.load(f)
        nodes = [TextNode(text=chunk) for chunk in chunk_texts]

        faiss_index = faiss.read_index(index_path)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        self.index = VectorStoreIndex(nodes=nodes, embed_model=self.embed_model, vector_store=vector_store)

        logger.info("✅ Moteur RAG (Ollama) initialisé avec succès.")

    # ---------------- LLM helpers (via Ollama) ----------------

    def _complete(self, prompt: str, stop: Optional[List[str]] = None, max_tokens: int = 128) -> str:
        text = self.llm.generate(
            prompt=prompt,
            stop=stop,
            max_tokens=max_tokens,
            stream=False,
            options=self.ollama_opts,
        )
        # Some Ollama setups may stream even when stream=False. Coerce generators to string.
        try:
            if hasattr(text, "__iter__") and not isinstance(text, (str, bytes)):
                chunks = []
                for t in text:
                    if not isinstance(t, (str, bytes)):
                        continue
                    chunks.append(t)
                text = "".join(chunks)
        except Exception:
            pass
        return (text or "").strip()

    def _complete_stream(self, prompt: str, stop: Optional[List[str]] = None, max_tokens: int = MAX_TOKENS):
        return self.llm.generate(
            prompt=prompt,
            stop=stop,
            max_tokens=max_tokens,
            stream=True,
            options=self.ollama_opts,
        )

    # ---------------- Reformulation ----------------

    def reformulate_question(self, question: str) -> str:
        logger.info("🔁 Reformulation de la question (sans contexte)...")
        prompt = f"""Tu es un assistant expert chargé de clarifier des questions floues.

Transforme la question suivante en une question claire, explicite et complète, sans ajouter d'informations extérieures.

Question floue : {question}
Question reformulée :"""
        reformulated = self._complete(prompt, stop=["\n"], max_tokens=128)
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
        reformulated = self._complete(prompt, stop=["\n"], max_tokens=128)
        logger.info(f"📝 Reformulée avec contexte : {reformulated}")
        return reformulated

    # ---------------- Retrieval ----------------

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

    # ---------------- Public API ----------------

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

        response = self._complete(prompt, stop=["### Instruction:"], max_tokens=MAX_TOKENS)
        response = response.strip().split("###")[0]
        logger.info(f"🧠 Réponse générée : {response[:120]}{{'...' if len(response) > 120 else ''}}")
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
        for token in self._complete_stream(prompt, stop=["### Instruction:"], max_tokens=MAX_TOKENS):
            print(token, end="", flush=True)
