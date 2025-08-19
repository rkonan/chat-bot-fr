from __future__ import annotations
import os, pickle, time, logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .clients.ollama_client import OllamaClient
from .retriever.base import RetrievedItem, Retriever
from .retriever.bm25 import BM25Retriever
from .retriever.dense import DenseRetriever, DenseResources
from .retriever.hybrid import HybridRetriever, HybridConfig
from .retriever.expander import QuestionExpander, ExpansionConfig

logger = logging.getLogger("RAGRefactor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(_handler)

DEFAULT_STOPS = ["### Réponse:", "", "###"]
MAX_TOKENS = 64

@dataclass
class EngineConfig:
    base_top_k: int = int(os.getenv("RAG_BASE_TOPK", "8"))
    pool_k: int     = int(os.getenv("RAG_POOL_K", "24"))
    context_clip: int = int(os.getenv("RAG_CTX_CLIP", "500"))
    tau_switch: float = 0.32

class Corpus:
    """Lazy loader of FAISS + nodes + embeddings."""
    def __init__(self, vector_path: str, index_path: str):
        self.vector_path = vector_path
        self.index_path = index_path
        self.loaded = False
        self.embed_model: Optional[HuggingFaceEmbedding] = None
        self.index: Optional[VectorStoreIndex] = None
        self.nodes: List[TextNode] = []

    def ensure_loaded(self):
        if self.loaded: return
        t0 = time.perf_counter()
        logger.info("⏳ Loading corpus (FAISS + chunks + embeddings)...")
        with open(self.vector_path, "rb") as f:
            chunk_texts: List[str] = pickle.load(f)
        self.nodes = [TextNode(text=chunk) for chunk in chunk_texts]
        faiss_index = faiss.read_index(self.index_path)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        self.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")
        self.index = VectorStoreIndex(nodes=self.nodes, embed_model=self.embed_model, vector_store=vector_store)
        self.loaded = True
        logger.info(f"✅ Corpus loaded in {time.perf_counter() - t0:.2f}s")

class RAGEngine:
    def __init__(self,
                 model_name: str,
                 vector_path: str,
                 index_path: str,
                 ollama_host: Optional[str] = None,
                 engine_cfg: EngineConfig = EngineConfig(),
                 hybrid_cfg: HybridConfig = HybridConfig(),
                 expansion_cfg: ExpansionConfig = ExpansionConfig()):
        self.cfg = engine_cfg
        self.llm = OllamaClient(model=model_name, host=ollama_host)
        self.corpus = Corpus(vector_path, index_path)
        self._retriever: Optional[Retriever] = None
        self.expansion_cfg = expansion_cfg
        self._last_stats: Dict[str, Any] = {}
        self.hybrid_cfg = hybrid_cfg

    # wiring
    def _build_retriever(self):
        self.corpus.ensure_loaded()
        dense = DenseRetriever(
            DenseResources(index=self.corpus.index, embed_model=self.corpus.embed_model, nodes=self.corpus.nodes),
            pool_k=self.cfg.pool_k,
        )
        bm25 = BM25Retriever([n.get_content() for n in self.corpus.nodes])
        # ⬇️ Expander now belongs to the retriever
        expander = QuestionExpander(self.expansion_cfg, llm=self.llm if self.expansion_cfg.strategy=="llm" else None)
        self._retriever = HybridRetriever(dense, bm25, self.corpus.embed_model, expander=expander, cfg=self.hybrid_cfg)

    # heuristics
    def _is_greeting(self, text: str) -> bool:
        s = text.lower().strip()
        return s in {"bonjour", "salut", "hello", "bonsoir", "hi", "coucou", "yo"} or len(s.split()) <= 2

    def _decide_mode(self, scores: List[float], tau: float, is_greeting: bool) -> str:
        if is_greeting: return "llm"
        top = scores[0] if scores else 0.0
        return "rag" if top >= tau else "llm"

    # public API
    def preview_context(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        if self._retriever is None: self._build_retriever()
        k = top_k or self.cfg.base_top_k
        items = self._retriever.retrieve(question, top_k=k)
        context = "".join(it.text[: self.hybrid_cfg.context_clip] for it in items)
        stats = getattr(self._retriever, "last_stats", {}) or {}
        self._last_stats = stats
        items_md = []
        for i, it in enumerate(items, start=1):
            items_md.append({
                "rank": i,
                "score": float(it.score),
                "node_id": it.node_id,
                "snippet": it.text[:400],
                "full_len": len(it.text),
            })
        return {
            "question": question,
            "top_k_effectif": k,
            "context": context,
            "items": items_md,
            "stats": stats,
        }

    def format_preview_md(self, preview: Dict[str, Any]) -> str:
        lines = []
        lines.append(f"### Question{preview['question']}")
        lines.append(f"**Top‑K**: {preview['top_k_effectif']}")
        st = preview.get("stats") or {}
        if st:
            lines.append("### Filtres appliqués (compteurs)")
            lines.append(
                f"- Variantes : **{st.get('variants')}**\n"
                f"- Pool unique : **{st.get('pool_unique')}**\n"
                f"- Après lexical : **{st.get('after_lexical')}**\n"
                f"- Après seuils : **{st.get('after_thresholds')}**\n"
                f"- Après MMR : **{st.get('after_mmr')}**\n"
                f"- Final : **{st.get('final_k')}**"
            )
        lines.append("### Top K chunks")
        for it in preview["items"]:
            snippet = it['snippet'].replace('', ' ')
            trailing = "…" if it['full_len'] > len(it['snippet']) else ""
            lines.append(
                f"- **#{it['rank']}** | score={it['score']:.4f} | id=`{it['node_id']}` | len={it['full_len']}\n\n> {snippet[:380]}{trailing}\n"
            )
        lines.append("---
### Contexte concaténé (troncature éventuelle)
")
        lines.append(preview["context"]) 
        return "
".join(lines)

    def ask(self, question: str, debug_preview: bool = False) -> str:
        is_hello = self._is_greeting(question)
        if self._retriever is None: self._build_retriever()
        k = self.cfg.base_top_k
        prev = self.preview_context(question, top_k=k)
        scores = [it["score"] for it in prev["items"]]
        mode = self._decide_mode(scores, tau=self.cfg.tau_switch, is_greeting=is_hello)
        if debug_preview:
            return self.format_preview_md(prev)
        if mode == "rag":
            prompt = (
                "Instruction: Réponds uniquement à partir du contexte. "
                "Si la réponse n'est pas déductible, réponds exactement: \"Information non présente dans le contexte.\""
                f"

Contexte :
{prev['context']}

Question : {question}
Réponse :"
            )
            return self.llm.generate(prompt, stop=DEFAULT_STOPS, max_tokens=MAX_TOKENS, stream=False) or ""
        # LLM pure
        messages = [{"role": "user", "content": question}]
        result = self.llm.chat(messages, stream=False)
        return result.get("message", {}).get("content", "")

    def ask_stream(self, question: str, debug_preview: bool = False) -> Iterable[str]:
        is_hello = self._is_greeting(question)
        if self._retriever is None: self._build_retriever()
        k = self.cfg.base_top_k
        prev = self.preview_context(question, top_k=k)
        scores = [it["score"] for it in prev["items"]]
        mode = self._decide_mode(scores, tau=self.cfg.tau_switch, is_greeting=is_hello)
        if debug_preview:
            yield self.format_preview_md(prev); return
        if mode == "rag":
            prompt = (
                "Instruction: Réponds uniquement à partir du contexte. "
                "Si la réponse n'est pas déductible, réponds exactement: \"Information non présente dans le contexte.\""
                f"

Contexte :
{prev['context']}

Question : {question}
Réponse :"
            )
            for tok in self.llm.generate(prompt, stop=DEFAULT_STOPS, max_tokens=MAX_TOKENS, stream=True):
                yield tok
            return
        # LLM pure
        messages = [{"role": "user", "content": question}]
        for tok in self.llm.chat(messages, stream=True):
            yield tok