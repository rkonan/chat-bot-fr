
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Engine — Drop-in (Hybrid + Ollama-compatible)
Keeps your previous interface:
    RAGEngine(model_name=..., ollama_host=..., vector_path=..., index_path=...)
Provides:
    - retrieve_context_hybrid(...)
    - ask(...) and ask_stream(...): now include CONTEXT -> Ollama generation
    - .model_name, .ollama_host, .ollama_client preserved
"""

import os, re, io, json, math, time, datetime, pickle, logging, requests
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("RAGHybrid")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(_h)

try:
    import faiss  # type: ignore
    from llama_index.core import VectorStoreIndex
    from llama_index.core.schema import TextNode
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except Exception as e:
    raise

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False

DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
DEFAULT_VECTOR_PKL = os.getenv("VECTOR_PKL", "vectordb/chunks.pkl")
DEFAULT_FAISS_INDEX = os.getenv("FAISS_INDEX", "vectordb/index.faiss")

# ---------- Ollama minimal client ----------
class OllamaClient:
    def __init__(self, host: str, model: str, timeout: int = 120):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, system: Optional[str] = None, options: Optional[Dict[str, Any]] = None, stop: Optional[List[str]] = None) -> str:
        url = f"{self.host}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system: payload["system"] = system
        if options: payload["options"] = options
        if stop: payload["stop"] = stop
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")

    def stream_generate(self, prompt: str, system: Optional[str] = None, options: Optional[Dict[str, Any]] = None, stop: Optional[List[str]] = None):
        url = f"{self.host}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": True}
        if system: payload["system"] = system
        if options: payload["options"] = options
        if stop: payload["stop"] = stop
        with requests.post(url, json=payload, timeout=self.timeout, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line: continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if "response" in msg:
                    yield msg["response"]

# ---------- utils ----------
_token_re = re.compile(r"\w+", re.UNICODE)
def _tokenize(text: str) -> List[str]:
    return _token_re.findall(text.lower())
def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0: return arr
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx <= mn: return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)
def _rrf(ranks: Dict[int, int], k: int = 60) -> Dict[int, float]:
    return {i: 1.0 / (k + r) for i, r in ranks.items()}
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float((a @ b) / (na * nb))
def _mmr(candidate_vecs: np.ndarray, query_vec: np.ndarray, lambda_div: float = 0.6, top_k: int = 8) -> List[int]:
    N = len(candidate_vecs)
    if N == 0: return []
    qnorm = np.linalg.norm(query_vec) + 1e-9
    cand_norms = np.linalg.norm(candidate_vecs, axis=1) + 1e-9
    query_sim = (candidate_vecs @ query_vec) / (cand_norms * qnorm)
    selected: List[int] = []
    remaining = set(range(N))
    while len(selected) < min(top_k, N):
        if not selected:
            i = int(np.argmax(query_sim)); selected.append(i); remaining.discard(i); continue
        best_score = -1e9; best_idx = None
        for i in list(remaining):
            div = max(
                (candidate_vecs[i] @ candidate_vecs[j]) / ((np.linalg.norm(candidate_vecs[i]) + 1e-9) * (np.linalg.norm(candidate_vecs[j]) + 1e-9))
                for j in selected
            )
            score = lambda_div * query_sim[i] - (1 - lambda_div) * div
            if score > best_score:
                best_score = score; best_idx = i
        selected.append(int(best_idx)); remaining.discard(int(best_idx))
    return selected
def _slugify(s: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-zA-Z0-9\-_.]+", "-", s.strip())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:max_len] if s else "preview"

# ---------- Engine ----------
class RAGEngine:
    def __init__(
        self,
        model_name: Optional[str] = None,          # legacy name for embedding or LLM model
        ollama_host: Optional[str] = None,         # preserved for generation
        vector_path: str = DEFAULT_VECTOR_PKL,
        index_path: str = DEFAULT_FAISS_INDEX,
        embed_model_name: Optional[str] = None,    # if set, overrides model_name for embeddings
        **kwargs,
    ) -> None:
        self.model_name = model_name or DEFAULT_EMBED_MODEL
        self.embed_model_name = embed_model_name or self.model_name
        self.ollama_host = ollama_host
        self.vector_path = vector_path
        self.index_path = index_path
        self.extra_kwargs = kwargs

        # LLM client (preserved API)
        self.ollama_client: Optional[OllamaClient] = None
        if self.ollama_host:
            self.ollama_client = OllamaClient(self.ollama_host, self.model_name)

        # Retrieval buffers
        self.embed_model: Optional[HuggingFaceEmbedding] = None
        self.index: Optional[VectorStoreIndex] = None
        self.faiss_index = None
        self.vector_store = None
        self.chunks: Optional[List[str]] = None
        self._bm25 = None
        self._idf = None
        self._tok_docs = None
        self._loaded = False

    # ---- load ----
    def _ensure_loaded(self) -> None:
        if self._loaded: return
        t0 = time.perf_counter()
        logger.info("⏳ Loading index+chunks+embed (embed=%s) — ollama_host=%s", self.embed_model_name, self.ollama_host)

        with open(self.vector_path, "rb") as f:
            chunk_texts: List[str] = pickle.load(f)
        self.chunks = chunk_texts

        self.faiss_index = faiss.read_index(self.index_path)
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)

        self.embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name)
        nodes = [TextNode(text=t) for t in chunk_texts]
        self.index = VectorStoreIndex(nodes=nodes, embed_model=self.embed_model, vector_store=self.vector_store)

        tokenized_corpus = [_tokenize(t) for t in self.chunks]
        if _HAS_BM25:
            self._bm25 = BM25Okapi(tokenized_corpus)
        else:
            df = {}
            for toks in tokenized_corpus:
                for w in set(toks):
                    df[w] = df.get(w, 0) + 1
            N = len(tokenized_corpus)
            self._idf = {w: math.log((N - df[w] + 0.5) / (df[w] + 0.5) + 1.0) for w in df}
            self._tok_docs = tokenized_corpus

        self._loaded = True
        logger.info("✅ RAG loaded in %.2fs", time.perf_counter() - t0)

    # ---- retrieval ----
    def _bm25_scores(self, query: str) -> np.ndarray:
        assert self.chunks is not None
        q = _tokenize(query)
        if self._bm25 is not None:
            return np.asarray(self._bm25.get_scores(q), dtype=np.float32)
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        q_set = set(q)
        for i, toks in enumerate(self._tok_docs or []):
            if not toks: continue
            tf = {}
            for w in toks: tf[w] = tf.get(w, 0) + 1
            s = 0.0
            for w in q_set:
                if w in tf and self._idf and w in self._idf:
                    s += (1 + math.log(1 + tf[w])) * self._idf[w]
            scores[i] = s
        return scores

    def retrieve_context_hybrid(self, question: str, top_k: int = 8, alpha_lexical: float = 0.35, use_rrf: bool = True, mmr_lambda: float = 0.6, pool_size: int = 64):
        self._ensure_loaded()
        assert self.index is not None and self.embed_model is not None and self.chunks is not None
        retriever = self.index.as_retriever(similarity_top_k=min(pool_size, len(self.chunks)))  # type: ignore
        retrieved_nodes = retriever.retrieve(question)
        node_texts = [n.get_content() for n in retrieved_nodes]

        q_vec = np.asarray(self.embed_model.get_query_embedding(question), dtype=np.float32)
        dense_scores = []
        for n in retrieved_nodes:
            emb = np.asarray(self.embed_model.get_text_embedding(n.get_content()), dtype=np.float32)
            dense_scores.append(_cos(q_vec, emb))
        dense_scores = np.asarray(dense_scores, dtype=np.float32)

        bm25_all = self._bm25_scores(question)
        text_to_idx = {}
        for i, t in enumerate(self.chunks):
            if t not in text_to_idx:
                text_to_idx[t] = i
        pool_global_idx = [text_to_idx.get(t, -1) for t in node_texts]
        valid = [i for i, g in enumerate(pool_global_idx) if g >= 0]
        if len(valid) < len(pool_global_idx):
            node_texts = [node_texts[i] for i in valid]
            retrieved_nodes = [retrieved_nodes[i] for i in valid]
            dense_scores = dense_scores[valid]
            pool_global_idx = [pool_global_idx[i] for i in valid]
        bm25_pool = bm25_all[pool_global_idx]

        if use_rrf:
            d_order = np.argsort(-dense_scores)
            b_order = np.argsort(-bm25_pool)
            ranks_dense = {i: int(r+1) for r, i in enumerate(d_order)}
            ranks_bm25 = {i: int(r+1) for r, i in enumerate(b_order)}
            rrf_d = _rrf(ranks_dense); rrf_b = _rrf(ranks_bm25)
            fused = {i: (1 - alpha_lexical) * rrf_d.get(i, 0.0) + alpha_lexical * rrf_b.get(i, 0.0) for i in range(len(node_texts))}
            fused_order_local = sorted(fused.keys(), key=lambda k: fused[k], reverse=True)
        else:
            d_norm = _minmax_norm(dense_scores)
            b_norm = _minmax_norm(bm25_pool)
            fused_vals = (1 - alpha_lexical) * d_norm + alpha_lexical * b_norm
            fused_order_local = list(np.argsort(-fused_vals))

        cand_texts = [node_texts[i] for i in fused_order_local[:min(pool_size, len(fused_order_local))]]
        cand_nodes = [retrieved_nodes[i] for i in fused_order_local[:min(pool_size, len(fused_order_local))]]
        cand_vecs = np.asarray([self.embed_model.get_text_embedding(t) for t in cand_texts], dtype=np.float32)
        mmr_idx = _mmr(cand_vecs, q_vec, lambda_div=mmr_lambda, top_k=top_k)

        final_nodes = [cand_nodes[i] for i in mmr_idx]
        final_scores = []
        for n in final_nodes:
            emb = np.asarray(self.embed_model.get_text_embedding(n.get_content()), dtype=np.float32)
            final_scores.append(_cos(q_vec, emb))
        context = "\n\n".join(n.get_content() for n in final_nodes)
        return context, final_nodes, final_scores

    # ---- generation (keeps your Ollama usage) ----
    def build_prompt(self, question: str, context: str) -> str:
        return f"### Contexte\n{context}\n\n### Question\n{question}\n\n### Réponse concise et précise:"

    def generate(self, question: str, context: str, system_prompt: Optional[str] = None, stop: Optional[List[str]] = None, options: Optional[Dict[str, Any]] = None) -> str:
        if not self.ollama_client:
            logger.warning("No Ollama host configured; returning context only.")
            return context
        prompt = self.build_prompt(question, context)
        return self.ollama_client.generate(prompt, system=system_prompt, options=options, stop=stop)

    def stream_generate(self, question: str, context: str, system_prompt: Optional[str] = None, stop: Optional[List[str]] = None, options: Optional[Dict[str, Any]] = None):
        if not self.ollama_client:
            yield context; return
        prompt = self.build_prompt(question, context)
        for tok in self.ollama_client.stream_generate(prompt, system=system_prompt, options=options, stop=stop):
            yield tok

    # ---- public API ----
    def ask(self, question: str, top_k: int = 8, debug_preview: bool = False) -> str:
        context, _, _ = self.retrieve_context_hybrid(question, top_k=top_k)
        return self.generate(question, context)

    def ask_stream(self, question: str, top_k: int = 8, debug_preview: bool = False) -> Iterable[str]:
        context, _, _ = self.retrieve_context_hybrid(question, top_k=top_k)
        yield from self.stream_generate(question, context)

# ------------------ CLI (debug / preview) ------------------
if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser(description="RAG Hybrid + Ollama (drop-in)")
    p.add_argument("--vector_path", type=str, default=DEFAULT_VECTOR_PKL, help="Pickle file with chunk texts")
    p.add_argument("--index_path", type=str, default=DEFAULT_FAISS_INDEX, help="FAISS index file")
    p.add_argument("--model_name", type=str, default=DEFAULT_EMBED_MODEL, help="LLM/Ollama model name")
    p.add_argument("--ollama_host", type=str, default="http://localhost:11434", help="Ollama host URL")
    p.add_argument("--preview", type=str, help="Run a retrieval+preview on this question")
    p.add_argument("--top_k", type=int, default=8, help="Top-K after MMR")
    args = p.parse_args()

    rag = RAGEngine(
        model_name=args.model_name,
        ollama_host=args.ollama_host,
        vector_path=args.vector_path,
        index_path=args.index_path,
    )

    if args.preview:
        ctx, nodes, scores = rag.retrieve_context_hybrid(args.preview, top_k=args.top_k)
        print("=== Context ===")
        print(ctx[:1000] + ("..." if len(ctx) > 1000 else ""))
        print("\n=== Nodes ===")
        for i, (n, sc) in enumerate(zip(nodes, scores), 1):
            print(f"#{i} score={sc:.4f} len={len(n.get_content())}")
        print("\n=== Answer (Ollama) ===")
        print(rag.generate(args.preview, ctx))
    else:
        print("Nothing to do. Use --preview 'your question'.")
