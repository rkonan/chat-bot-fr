import os
import pickle
import logging
import time
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

# ---------- Client Ollama (use /api/generate, no options) ----------
class OllamaClient:
    def __init__(self, model: str, host: Optional[str] = None, timeout: int = 300):
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11435") #mode proxy
        #self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
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
        # ❌ aucune "options" pour laisser Ollama auto-tuner

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

# ---------- RAG Engine (lazy load + heuristique GK) ----------
class RAGEngine:
    def __init__(self, model_name: str, vector_path: str, index_path: str,
                 model_threads: int = 4, ollama_host: Optional[str] = None,
                 ollama_opts: Optional[Dict[str, Any]] = None):

        logger.info(f"🔎 rag_model_ollama source: {__file__}")
        logger.info("📦 Initialisation du moteur (lazy RAG)...")

        # LLM prêt immédiatement
        self.llm = OllamaClient(model=model_name, host=ollama_host)

        # chemins pour chargement différé
        self.vector_path = vector_path
        self.index_path = index_path

        # objets RAG paresseux
        self.embed_model: Optional[HuggingFaceEmbedding] = None
        self.index: Optional[VectorStoreIndex] = None
        self._loaded = False

        logger.info("✅ Moteur initialisé (sans charger FAISS ni chunks).")

    # ---------- lazy loader ----------
    def _ensure_loaded(self):
        if self._loaded:
            return
        t0 = time.perf_counter()
        logger.info("⏳ Chargement lazy des données RAG (FAISS + chunks + embeddings)...")

        # 1) chunks
        with open(self.vector_path, "rb") as f:
            chunk_texts: List[str] = pickle.load(f)
        nodes = [TextNode(text=chunk) for chunk in chunk_texts]

        # 2) index FAISS
        faiss_index = faiss.read_index(self.index_path)
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        # 3) modèle d'embedding
        self.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")

        # 4) index LlamaIndex
        self.index = VectorStoreIndex(nodes=nodes, embed_model=self.embed_model, vector_store=vector_store)

        self._loaded = True
        logger.info(f"✅ RAG chargé en {time.perf_counter() - t0:.2f}s (lazy).")

    # ---------- génération ----------
    def _complete_stream(self, prompt: str, stop: Optional[List[str]] = None,
                         max_tokens: int = MAX_TOKENS, raw: bool = False):
        return self.llm.generate(prompt=prompt, stop=stop, max_tokens=max_tokens,
                                 stream=True, raw=raw)

    def _complete(self, prompt: str, stop: Optional[List[str]] = None,
                  max_tokens: int = 128, raw: bool = False) -> str:
        text = self.llm.generate(prompt=prompt, stop=stop, max_tokens=max_tokens,
                                 stream=False, raw=raw)
        return (text or "").strip()

    # ---------- heuristiques ----------
    def _is_greeting(self, text: str) -> bool:
        s = text.lower().strip()
        return s in {"bonjour", "salut", "hello", "bonsoir", "hi", "coucou", "yo"} or len(s.split()) <= 2

    def _looks_general_knowledge(self, q: str) -> bool:
        q = q.lower().strip()
        gk_keywords = (
            "capitale", "date de naissance", "qui est", "qu'est-ce", "definition",
            "définition", "histoire", "pays", "ville", "math", "science", "sport"
        )
        if len(q.split()) <= 9:
            if any(k in q for k in gk_keywords) or q.startswith(("quelle est", "qui est", "qu'est-ce", "c'est quoi")):
                return True
        return False

    def _should_use_rag_fast(self, question: str) -> bool:
        """N'active RAG que si on détecte des indices 'docs' / longueur significative."""
        q = question.lower()

        # 1) GK → pas de RAG
        if self._looks_general_knowledge(q):
            return False

        # 2) indices RAG
        doc_keywords = (
            "document", "docs", "procédure", "politique", "policy",
            "manuel", "guide", "pdf", "docling", "selon", "dans le contexte",
            "page", "section", "chapitre", "référence", "références", "conformément",
            "note technique", "spécification", "spec", "architecture", "adr"
        )
        if any(k in q for k in doc_keywords):
            return True

        # 3) question longue → probable RAG
        if len(q.split()) >= 14:
            return True

        return False

    def _decide_mode(self, scores: List[float], tau: float = 0.32, is_greeting: bool = False) -> str:
        if is_greeting:
            return "llm"
        top = scores[0] if scores else 0.0
        return "rag" if top >= tau else "llm"

    # ---------- retrieval ----------
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
        assert self.embed_model is not None
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
        self._ensure_loaded()
        retriever = self.index.as_retriever(similarity_top_k=top_k)  # type: ignore
        retrieved_nodes = retriever.retrieve(question)
        scores, nodes = self.rerank_nodes(question, retrieved_nodes, top_k)
        context = "\n\n".join(n.get_content()[:500] for n in nodes)
        return context, nodes, scores

    # ---------- API publique ----------
    def ask(self, question: str, allow_fallback: bool = False) -> str:
        logger.info(f"💬 [Non-stream] Question reçue : {question}")
        is_hello = self._is_greeting(question)

        use_rag = (self._loaded and not is_hello) or (not self._loaded and self._should_use_rag_fast(question))
        if use_rag:
            top_k = self.get_adaptive_top_k(question)
            context, _, scores = self.retrieve_context(question, top_k)
            mode = self._decide_mode(scores, tau=0.32, is_greeting=is_hello)
            if mode == "rag":
                prompt = (
                    "Instruction: Réponds uniquement à partir du contexte. "
                    "Si la réponse n'est pas déductible, réponds exactement: \"Information non présente dans le contexte.\""
                    f"\n\nContexte :\n{context}\n\nQuestion : {question}\nRéponse :"
                )
                return self._complete(prompt, stop=DEFAULT_STOPS, raw=False)

        # LLM pur
        prompt_llm = (
            "Réponds brièvement et précisément en français.\n"
            f"Question : {question}\nRéponse :"
        )
        return self._complete(prompt_llm, stop=DEFAULT_STOPS, raw=False)

    def ask_stream(self, question: str, allow_fallback: bool = False) -> Iterable[str]:
        logger.info(f"💬 [Stream] Question reçue : {question}")
        is_hello = self._is_greeting(question)

        use_rag = (self._loaded and not is_hello) or (not self._loaded and self._should_use_rag_fast(question))
        if use_rag:
            top_k = self.get_adaptive_top_k(question)
            context, _, scores = self.retrieve_context(question, top_k)
            mode = self._decide_mode(scores, tau=0.32, is_greeting=is_hello)
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

        # LLM pur
        prompt_llm = (
            "Réponds brièvement et précisément en français.\n"
            f"Question : {question}\nRéponse :"
        )
        logger.info("📡 Début streaming (LLM pur)...")
        for token in self._complete_stream(prompt_llm, stop=DEFAULT_STOPS, raw=False):
            yield token
        logger.info("📡 Fin streaming (LLM pur).")
