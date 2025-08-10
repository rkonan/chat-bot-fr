import os
import json
import pickle
import textwrap
import logging
from typing import List, Optional, Dict, Any, Iterable, Tuple

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

MAX_TOKENS = 64  # bornage court sur CPU-only
DEFAULT_STOPS = ["</s>", "\n\n", "\nQuestion:", "Question:"]


class OllamaClient:
    """
    Minimal Ollama client for /api/generate (text completion) with streaming support.
    """
    def __init__(self, model: str, host: Optional[str] = None, timeout: int = 300):
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
        raw: bool = False,
    ) -> str | Iterable[str]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
        }
        if raw:
            payload["raw"] = True  # IMPORTANT: désactive le template Modelfile
        if stop:
            payload["stop"] = stop
        if max_tokens is not None:
            payload["num_predict"] = int(max_tokens)  # nommage Ollama
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
                        continue
                    # En stream, Ollama renvoie des morceaux dans "response"
                    if "response" in data and not data.get("done"):
                        yield data["response"]
                    if data.get("done"):
                        break
            return

        r = requests.post(self._gen_url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")


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
            model_name: e.g. "noushermes_rag"
            vector_path: pickle file with chunk texts list[str]
            index_path: FAISS index path
            model_threads: forwarded as a hint to Ollama options
            ollama_host: override OLLAMA_HOST (default http://localhost:11434)
            ollama_opts: extra Ollama options (temperature, num_ctx, num_batch, num_thread)
        """
        logger.info(f"🔎 rag_model_ollama source: {__file__}")
        logger.info("📦 Initialisation du moteur RAG (Ollama)...")

        # Options Ollama (par défaut optimisées CPU)
        opts = dict(ollama_opts or {})
        opts.setdefault("temperature", 0.0)
        opts.setdefault("num_ctx", 512)
        opts.setdefault("num_batch", 16)
        if "num_thread" not in opts and model_threads:
            opts["num_thread"] = int(model_threads)

        self.llm = OllamaClient(model=model_name, host=ollama_host)
        self.ollama_opts = opts

        # Embedding model pour retrieval / rerank
        self.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")

        logger.info(f"📂 Chargement des données vectorielles depuis {vector_path}")
        with open(vector_path, "rb") as f:
            chunk_texts: List[str] = pickle.load(f)
        nodes = [TextNode(text=chunk) for chunk in chunk_texts]

        faiss_index = faiss.read_index(index_path)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        self.index = VectorStoreIndex(nodes=nodes, embed_model=self.embed_model, vector_store=vector_store)

        logger.info("✅ Moteur RAG (Ollama) initialisé avec succès.")

    # ---------------- LLM helpers (via Ollama) ----------------

    def _complete(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = MAX_TOKENS,
        raw: bool = True
    ) -> str:
        text = self.llm.generate(
            prompt=prompt,
            stop=stop or DEFAULT_STOPS,
            max_tokens=max_tokens,
            stream=False,
            options=self.ollama_opts,
            raw=raw,  # toujours True pour bypass Modelfile
        )
        # Par sécurité si un générateur se glisse quand stream=False
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

    def _complete_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: int = MAX_TOKENS,
        raw: bool = True
    ) -> Iterable[str]:
        return self.llm.generate(
            prompt=prompt,
            stop=stop or DEFAULT_STOPS,
            max_tokens=max_tokens,
            stream=True,
            options=self.ollama_opts,
            raw=raw,  # toujours True pour bypass Modelfile
        )

    # ---------------- Utilities ----------------

    def _is_greeting(self, text: str) -> bool:
        s = text.lower().strip()
        return s in {"bonjour", "salut", "hello", "bonsoir", "hi", "coucou", "yo"} or len(s.split()) <= 2

    def _decide_mode(self, scores: List[float], tau: float = 0.32, is_greeting: bool = False) -> str:
        if is_greeting:
            return "llm"
        top = scores[0] if scores else 0.0
        return "rag" if top >= tau else "llm"

    def _stream_with_local_stops(self, tokens: Iterable[str], stops: List[str]) -> Iterable[str]:
        """
        Coupe localement le stream si un stop apparaît, même si le serveur ne s'arrête pas.
        """
        buffer = ""
        for chunk in tokens:
            buffer += chunk
            # Check si un des stops est présent dans le buffer
            hit = None
            for s in stops:
                idx = buffer.find(s)
                if idx != -1:
                    hit = (s, idx)
                    break
            if hit:
                s, idx = hit
                # Yield tout avant le stop, puis stoppe
                yield buffer[:idx]
                break
            else:
                # Si pas de stop, on envoie le chunk tel quel
                yield chunk

    # ---------------- Retrieval + (optional) rerank ----------------

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

    def rerank_nodes(self, question: str, retrieved_nodes, top_k: int = 3) -> Tuple[List[float], List[TextNode]]:
        logger.info(f"🔍 Re-ranking des {len(retrieved_nodes)} chunks pour la question : « {question} »")
        q_emb = self.embed_model.get_query_embedding(question)
        scored_nodes: List[Tuple[float, TextNode]] = []

        for node in retrieved_nodes:
            chunk_text = node.get_content()
            chunk_emb = self.embed_model.get_text_embedding(chunk_text)
            score = cos_sim(q_emb, chunk_emb).item()
            scored_nodes.append((score, node))

        ranked = sorted(scored_nodes, key=lambda x: x[0], reverse=True)

        logger.info("📊 Chunks les plus pertinents :")
        for i, (score, node) in enumerate(ranked[:top_k]):
            chunk_preview = textwrap.shorten(node.get_content().replace("\n", " "), width=100)
            logger.info(f"#{i+1} | Score: {score:.4f} | {chunk_preview}")

        top = ranked[:top_k]
        scores = [s for s, _ in top]
        nodes = [n for _, n in top]
        return scores, nodes

    def retrieve_context(self, question: str, top_k: int = 3) -> Tuple[str, List[TextNode], List[float]]:
        logger.info("📥 Récupération du contexte...")
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        retrieved_nodes = retriever.retrieve(question)
        scores, nodes = self.rerank_nodes(question, retrieved_nodes, top_k)
        context = "\n\n".join(n.get_content()[:500] for n in nodes)
        return context, nodes, scores

    # ---------------- Public API ----------------

    def ask(self, question_raw: str, allow_fallback: bool = True) -> str:
        logger.info(f"💬 Question reçue : {question_raw}")
        is_hello = self._is_greeting(question_raw)

        # retrieval (sauf salutations)
        context, scores = "", []
        if not is_hello:
            top_k = self.get_adaptive_top_k(question_raw)
            context, _, scores = self.retrieve_context(question_raw, top_k)

        # router RAG vs LLM
        mode = self._decide_mode(scores, tau=0.32, is_greeting=is_hello)
        logger.info(f"🧭 Mode choisi : {mode}")

        if mode == "rag":
            prompt = (
                "Instruction: Réponds uniquement à partir du contexte. "
                "Si la réponse n'est pas déductible, réponds exactement: \"Information non présente dans le contexte.\""
                "\n\nContexte :\n"
                f"{context}\n\n"
                f"Question : {question_raw}\n"
                "Réponse :"
            )

            resp = self._complete(
                prompt,
                stop=DEFAULT_STOPS,
                max_tokens=MAX_TOKENS,
                raw=True,   # ✅ bypass Modelfile/template
            ).strip()

            # fallback LLM‑pur si le RAG n'a rien trouvé
            if allow_fallback and "Information non présente" in resp:
                logger.info("↪️ Fallback LLM‑pur (hors contexte)")
                prompt_llm = (
                    "Réponds brièvement et précisément en français.\n"
                    f"Question : {question_raw}\n"
                    "Réponse :"
                )
                resp = self._complete(
                    prompt_llm,
                    stop=DEFAULT_STOPS,
                    max_tokens=MAX_TOKENS,
                    raw=True
                ).strip()

            ellipsis = "..." if len(resp) > 120 else ""
            logger.info(f"🧠 Réponse générée : {resp[:120]}{ellipsis}")
            return resp

        # LLM pur (salutation ou score faible)
        prompt_llm = (
            "Réponds brièvement et précisément en français.\n"
            f"Question : {question_raw}\n"
            "Réponse :"
        )
        resp = self._complete(
            prompt_llm,
            stop=DEFAULT_STOPS,
            max_tokens=MAX_TOKENS,
            raw=True
        ).strip()
        ellipsis = "..." if len(resp) > 120 else ""
        logger.info(f"🧠 Réponse générée : {resp[:120]}{ellipsis}")
        return resp

    def ask_stream(self, question: str, allow_fallback: bool = False) -> Iterable[str]:
        logger.info(f"💬 [Stream] Question reçue : {question}")
        is_hello = self._is_greeting(question)

        context, scores = "", []
        if not is_hello:
            top_k = self.get_adaptive_top_k(question)
            context, _, scores = self.retrieve_context(question, top_k)

        mode = self._decide_mode(scores, tau=0.32, is_greeting=is_hello)
        logger.info(f"🧭 Mode choisi (stream) : {mode}")

        stops = DEFAULT_STOPS

        if mode == "rag":
            prompt = (
                "Instruction: Réponds uniquement à partir du contexte. "
                "Si la réponse n'est pas déductible, réponds exactement: \"Information non présente dans le contexte.\""
                "\n\nContexte :\n"
                f"{context}\n\n"
                f"Question : {question}\n"
                "Réponse :"
            )

            logger.info("📡 Début du streaming de la réponse (RAG)...")
            tokens = self._complete_stream(
                prompt,
                stop=stops,
                max_tokens=MAX_TOKENS,
                raw=True,
            )
            # Blindage local: coupe si un stop apparaît
            for t in self._stream_with_local_stops(tokens, stops):
                if t:
                    yield t
            logger.info("📡 Fin du streaming de la réponse (RAG).")
            return

        # LLM pur en stream
        prompt_llm = (
            "Réponds brièvement et précisément en français.\n"
            f"Question : {question}\n"
            "Réponse :"
        )
        logger.info("📡 Début du streaming de la réponse (LLM pur)...")
        tokens = self._complete_stream(
            prompt_llm,
            stop=stops,
            max_tokens=MAX_TOKENS,
            raw=True,
        )
        for t in self._stream_with_local_stops(tokens, stops):
            if t:
                yield t
        logger.info("📡 Fin du streaming de la réponse (LLM pur).")
