import os
import pickle
import logging
from typing import List, Optional, Dict, Any, Iterable, Tuple
import requests
import faiss
import json
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

MAX_TOKENS = 64
DEFAULT_STOPS = ["### Réponse:", "\n\n", "###"]

class OllamaClient:
    def __init__(self, model: str, host: Optional[str] = None, timeout: int = 300):
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout
        self._gen_url = self.host.rstrip("/") + "/api/generate"

    def generate(self, prompt: str, stop: Optional[List[str]] = None,
                 max_tokens: Optional[int] = None, stream: bool = False,
                 options: Optional[Dict[str, Any]] = None, raw: bool = False) -> str | Iterable[str]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
        }
        if raw:
            payload["raw"] = True
        if stop:
            payload["stop"] = stop
        if max_tokens is not None:
            payload["num_predict"] = int(max_tokens)
        # ❌ Pas d'options envoyées pour laisser Ollama choisir ses defaults

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
    def __init__(self, model_name: str, vector_path: str, index_path: str,
                 model_threads: int = 4, ollama_host: Optional[str] = None,
                 ollama_opts: Optional[Dict[str, Any]] = None):

        logger.info(f"🔎 rag_model_ollama source: {__file__}")
        logger.info("📦 Initialisation du moteur RAG (Ollama)...")

        # ❌ Pas d'options Ollama stockées
        self.llm = OllamaClient(model=model_name, host=ollama_host)
        self.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")

        logger.info(f"📂 Chargement des données vectorielles depuis {vector_path}")
        with open(vector_path, "rb") as f:
            chunk_texts: List[str] = pickle.load(f)
        nodes = [TextNode(text=chunk) for chunk in chunk_texts]

        faiss_index = faiss.read_index(index_path)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        self.index = VectorStoreIndex(nodes=nodes, embed_model=self.embed_model, vector_store=vector_store)

        logger.info("✅ Moteur RAG (Ollama) initialisé avec succès.")

        # Warmup pour charger le modèle
        try:
            logger.info("⚡ Warmup du modèle Ollama...")
            for _ in self._complete_stream("Bonjour", max_tokens=8, raw=False):
                pass
        except Exception as e:
            logger.warning(f"Warmup échoué : {e}")

    def _complete_stream(self, prompt: str, stop: Optional[List[str]] = None,
                         max_tokens: int = MAX_TOKENS, raw: bool = False):
        return self.llm.generate(prompt=prompt, stop=stop, max_tokens=max_tokens,
                                 stream=True, raw=raw)

    def _complete(self, prompt: str, stop: Optional[List[str]] = None,
                  max_tokens: int = 128, raw: bool = False) -> str:
        text = self.llm.generate(prompt=prompt, stop=stop, max_tokens=max_tokens,
                                 stream=False, raw=raw)
        return (text or "").strip()

    def _is_greeting(self, text: str) -> bool:
        s = text.lower().strip()
        return s in {"bonjour", "salut", "hello", "bonsoir", "hi", "coucou", "yo"} or len(s.split()) <= 2

    def _decide_mode(self, scores: List[float], tau: float = 0.32, is_greeting: bool = False) -> str:
        if is_greeting:
            return "llm"
        top = scores[0] if scores else 0.0
        return "rag" if top >= tau else "llm"

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
        logger.info(f"🔍 Re-ranking des {len(retrieved_nodes)} chunks pour : « {question} »")
        q_emb = self.embed_model.get_query_embedding(question)
        scored_nodes: List[Tuple[float, TextNode]] = []
        for node in retrieved_nodes:
            chunk_emb = self.embed_model.get_text_embedding(node.get_content())
            score = cos_sim(q_emb, chunk_emb).item()
            scored_nodes.append((score, node))
        ranked = sorted(scored_nodes, key=lambda x: x[0], reverse=True)
        top = ranked[:top_k]
        return [s for s, _ in top], [n for _, n in top]

    def retrieve_context(self, question: str, top_k: int = 3) -> Tuple[str, List[TextNode], List[float]]:
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        retrieved_nodes = retriever.retrieve(question)
        scores, nodes = self.rerank_nodes(question, retrieved_nodes, top_k)
        context = "\n\n".join(n.get_content()[:500] for n in nodes)
        return context, nodes, scores

    def ask(self, question: str, allow_fallback: bool = False) -> str:
        """Génération non-stream"""
        logger.info(f"💬 [Non-stream] Question reçue : {question}")
        is_hello = self._is_greeting(question)
        context, scores = "", []
        if not is_hello:
            top_k = self.get_adaptive_top_k(question)
            context, _, scores = self.retrieve_context(question, top_k)

        mode = self._decide_mode(scores, tau=0.32, is_greeting=is_hello)
        logger.info(f"🧭 Mode choisi (non-stream) : {mode}")

        if mode == "rag":
            prompt = (
                "Instruction: Réponds uniquement à partir du contexte. "
                "Si la réponse n'est pas déductible, réponds exactement: \"Information non présente dans le contexte.\""
                f"\n\nContexte :\n{context}\n\nQuestion : {question}\nRéponse :"
            )
            return self._complete(prompt, stop=DEFAULT_STOPS, raw=False)

        prompt_llm = (
            "Réponds brièvement et précisément en français.\n"
            f"Question : {question}\nRéponse :"
        )
        return self._complete(prompt_llm, stop=DEFAULT_STOPS, raw=False)

    def ask_stream(self, question: str, allow_fallback: bool = False) -> Iterable[str]:
        logger.info(f"💬 [Stream] Question reçue : {question}")
        is_hello = self._is_greeting(question)
        context, scores = "", []
        if not is_hello:
            top_k = self.get_adaptive_top_k(question)
            context, _, scores = self.retrieve_context(question, top_k)

        mode = self._decide_mode(scores, tau=0.32, is_greeting=is_hello)
        logger.info(f"🧭 Mode choisi (stream) : {mode}")

        if mode == "rag":
            prompt = (
                "Instruction: Réponds uniquement à partir du contexte. "
                "Si la réponse n'est pas déductible, réponds exactement: \"Information non présente dans le contexte.\""
                f"\n\nContexte :\n{context}\n\nQuestion : {question}\nRéponse :"
            )
            logger.info("📡 Début streaming (RAG)...")
            for token in self._complete_stream(prompt, stop=DEFAULT_STOPS, raw=False):
                yield token
            logger.info("📡 Fin streaming (RAG).")
            return

        prompt_llm = (
            "Réponds brièvement et précisément en français.\n"
            f"Question : {question}\nRéponse :"
        )
        logger.info("📡 Début streaming (LLM pur)...")
        for token in self._complete_stream(prompt_llm, stop=DEFAULT_STOPS, raw=False):
            yield token
        logger.info("📡 Fin streaming (LLM pur).")
