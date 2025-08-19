
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Engine (Hybrid Retrieval v1)
--------------------------------
Drop-in file providing:
- Hybrid retrieval (BM25 + Dense) with RRF fusion + MMR dedup/diversity
- Preview export to JSON/Markdown
- Minimal CLI to test retrieval without hitting the LLM

Assumptions:
- You already have a FAISS index built by LlamaIndex + a pickle of chunk texts aligned with the index.
- Embeddings are computed with a HuggingFace model (e.g., multilingual-e5-*). Adjust DEFAULTS below.
"""

import os
import re
import io
import json
import math
import time
import argparse
import datetime
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Logging
import logging
logger = logging.getLogger("RAGHybrid")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
if not logger.handlers:
    logger.addHandler(_handler)

# Numpy
import numpy as np

# FAISS / LlamaIndex
try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("faiss is required. Install faiss-cpu or faiss-gpu in your environment.") from e

try:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.schema import TextNode
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except Exception as e:
    raise RuntimeError("llama-index packages are required (core, vector_stores.faiss, embeddings.huggingface).") from e

# Optional BM25; with pure-Python fallback
try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False

# ------------------ DEFAULTS (edit these to fit your setup) ------------------
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
DEFAULT_VECTOR_PKL = os.getenv("VECTOR_PKL", "vectordb/chunks.pkl")   # list[str] of chunk texts
DEFAULT_FAISS_INDEX = os.getenv("FAISS_INDEX", "vectordb/index.faiss")

# ------------------ Small utils ------------------
_token_re = re.compile(r"\w+", re.UNICODE)

def _tokenize(text: str) -> List[str]:
    return _token_re.findall(text.lower())

def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)

def _rrf(ranks: Dict[int, int], k: int = 60) -> Dict[int, float]:
    # ranks: item -> position (1-based)
    return {i: 1.0 / (k + r) for i, r in ranks.items()}

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float((a @ b) / (na * nb))

def _mmr(candidate_vecs: np.ndarray, query_vec: np.ndarray, lambda_div: float = 0.6, top_k: int = 8) -> List[int]:
    """Maximal Marginal Relevance reranking.
    candidate_vecs: [N, D], query_vec: [D] (np.float32)
    Returns a list of indices into candidate_vecs (length <= top_k).
    """
    N = len(candidate_vecs)
    if N == 0:
        return []
    qnorm = np.linalg.norm(query_vec) + 1e-9
    cand_norms = np.linalg.norm(candidate_vecs, axis=1) + 1e-9
    query_sim = (candidate_vecs @ query_vec) / (cand_norms * qnorm)  # [N]

    selected: List[int] = []
    remaining = set(range(N))

    while len(selected) < min(top_k, N):
        if not selected:
            i = int(np.argmax(query_sim))
            selected.append(i)
            remaining.discard(i)
            continue

        best_score = -1e9
        best_idx = None
        for i in list(remaining):
            # diversity term = max cosine similarity to items already selected
            div = max(
                (candidate_vecs[i] @ candidate_vecs[j]) / ((np.linalg.norm(candidate_vecs[i]) + 1e-9) * (np.linalg.norm(candidate_vecs[j]) + 1e-9))
                for j in selected
            )
            score = lambda_div * query_sim[i] - (1 - lambda_div) * div
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(int(best_idx))  # type: ignore
        remaining.discard(int(best_idx))  # type: ignore

    return selected

def _slugify(s: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-zA-Z0-9\-_.]+", "-", s.strip())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:max_len] if s else "preview"

# ------------------ RAG Engine ------------------
class RAGEngine:
    def __init__(
        self,
        vector_path: str = DEFAULT_VECTOR_PKL,
        index_path: str = DEFAULT_FAISS_INDEX,
        embed_model_name: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        self.vector_path = vector_path
        self.index_path = index_path
        self.embed_model_name = embed_model_name

        self.embed_model: Optional[HuggingFaceEmbedding] = None
        self.index: Optional[VectorStoreIndex] = None
        self.faiss_index = None
        self.vector_store = None
        self._loaded = False

        # buffers for hybrid retrieval
        self.chunks: Optional[List[str]] = None
        self._bm25 = None
        self._idf = None
        self._tok_docs = None

    # ------------------ Loading ------------------
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        t0 = time.perf_counter()
        logger.info("⏳ Loading FAISS index + chunks + embedding model...")

        # 1) Chunks (aligned to FAISS vectors)
        with open(self.vector_path, "rb") as f:
            chunk_texts: List[str] = pickle.load(f)
        self.chunks = chunk_texts

        # 2) FAISS
        self.faiss_index = faiss.read_index(self.index_path)
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)

        # 3) Embedding model
        self.embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name)

        # 4) LlamaIndex VectorStoreIndex (for retriever convenience)
        nodes = [TextNode(text=t) for t in chunk_texts]
        self.index = VectorStoreIndex(nodes=nodes, embed_model=self.embed_model, vector_store=self.vector_store)

        # 5) BM25 init
        tokenized_corpus = [_tokenize(t) for t in self.chunks]
        if _HAS_BM25:
            self._bm25 = BM25Okapi(tokenized_corpus)
        else:
            # pure-Python tf*idf fallback (not exact BM25 but adequate for blending)
            df = {}
            for toks in tokenized_corpus:
                for w in set(toks):
                    df[w] = df.get(w, 0) + 1
            N = len(tokenized_corpus)
            self._idf = {w: math.log((N - df[w] + 0.5) / (df[w] + 0.5) + 1.0) for w in df}
            self._tok_docs = tokenized_corpus

        self._loaded = True
        logger.info("✅ RAG loaded in %.2fs", time.perf_counter() - t0)

    # ------------------ BM25 scoring ------------------
    def _bm25_scores(self, query: str) -> np.ndarray:
        assert self.chunks is not None
        q = _tokenize(query)
        if self._bm25 is not None:
            return np.asarray(self._bm25.get_scores(q), dtype=np.float32)
        # fallback tf*idf-ish
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        q_set = set(q)
        for i, toks in enumerate(self._tok_docs or []):
            if not toks:
                continue
            tf = {}
            for w in toks:
                tf[w] = tf.get(w, 0) + 1
            s = 0.0
            for w in q_set:
                if w in tf and self._idf and w in self._idf:
                    s += (1 + math.log(1 + tf[w])) * self._idf[w]
            scores[i] = s
        return scores

    # ------------------ Hybrid Retrieval ------------------
    def retrieve_context_hybrid(
        self,
        question: str,
        top_k: int = 8,
        alpha_lexical: float = 0.35,
        use_rrf: bool = True,
        mmr_lambda: float = 0.6,
        pool_size: int = 64,
    ) -> Tuple[str, List[TextNode], List[float]]:
        """Returns (context_text, nodes, dense_cos_scores)"""
        self._ensure_loaded()
        assert self.index is not None and self.embed_model is not None and self.chunks is not None

        # 1) Dense pool via LlamaIndex retriever (uses FAISS)
        retriever = self.index.as_retriever(similarity_top_k=min(pool_size, len(self.chunks)))  # type: ignore
        retrieved_nodes = retriever.retrieve(question)
        node_texts = [n.get_content() for n in retrieved_nodes]

        # Dense cos scores vs query
        q_vec = np.asarray(self.embed_model.get_query_embedding(question), dtype=np.float32)
        dense_scores = []
        for n in retrieved_nodes:
            emb = np.asarray(self.embed_model.get_text_embedding(n.get_content()), dtype=np.float32)
            dense_scores.append(_cos(q_vec, emb))
        dense_scores = np.asarray(dense_scores, dtype=np.float32)

        # 2) BM25 scores for same pool
        bm25_all = self._bm25_scores(question)
        # Map text -> global index; if duplicates exist, first occurrence wins
        text_to_idx = {}
        for i, t in enumerate(self.chunks):
            if t not in text_to_idx:
                text_to_idx[t] = i
        pool_global_idx = [text_to_idx.get(t, -1) for t in node_texts]
        valid = [i for i, g in enumerate(pool_global_idx) if g >= 0]
        if len(valid) < len(pool_global_idx):
            # filter out unknowns
            node_texts = [node_texts[i] for i in valid]
            retrieved_nodes = [retrieved_nodes[i] for i in valid]
            dense_scores = dense_scores[valid]
            pool_global_idx = [pool_global_idx[i] for i in valid]
        bm25_pool = bm25_all[pool_global_idx]

        # 3) Fusion (RRF by default)
        if use_rrf:
            d_order = np.argsort(-dense_scores)
            b_order = np.argsort(-bm25_pool)
            ranks_dense = {i: int(r+1) for r, i in enumerate(d_order)}
            ranks_bm25 = {i: int(r+1) for r, i in enumerate(b_order)}
            rrf_d = _rrf(ranks_dense)
            rrf_b = _rrf(ranks_bm25)
            fused = {i: (1 - alpha_lexical) * rrf_d.get(i, 0.0) + alpha_lexical * rrf_b.get(i, 0.0) for i in range(len(node_texts))}
            fused_order_local = sorted(fused.keys(), key=lambda k: fused[k], reverse=True)
        else:
            d_norm = _minmax_norm(dense_scores)
            b_norm = _minmax_norm(bm25_pool)
            fused_vals = (1 - alpha_lexical) * d_norm + alpha_lexical * b_norm
            fused_order_local = list(np.argsort(-fused_vals))

        # 4) MMR reranking on embeddings of fused candidates
        cand_texts = [node_texts[i] for i in fused_order_local[:min(pool_size, len(fused_order_local))]]
        cand_nodes = [retrieved_nodes[i] for i in fused_order_local[:min(pool_size, len(fused_order_local))]]
        cand_vecs = np.asarray([self.embed_model.get_text_embedding(t) for t in cand_texts], dtype=np.float32)
        mmr_idx = _mmr(cand_vecs, q_vec, lambda_div=mmr_lambda, top_k=top_k)

        final_nodes = [cand_nodes[i] for i in mmr_idx]
        # recompute dense cos scores for final selection (optional)
        final_scores = []
        for n in final_nodes:
            emb = np.asarray(self.embed_model.get_text_embedding(n.get_content()), dtype=np.float32)
            final_scores.append(_cos(q_vec, emb))

        context = "\n\n".join(n.get_content() for n in final_nodes)
        return context, final_nodes, final_scores

    # ------------------ Helper: adaptive Top-K (simple) ------------------
    def get_adaptive_top_k(self, question: str, base_k: int = 8) -> int:
        # Placeholder heuristic: can be replaced by entropy/length/etc.
        L = len(question)
        if L < 40:
            return base_k
        if L < 120:
            return base_k + 2
        return base_k + 4

    # ------------------ Mode decision (placeholder) ------------------
    def _decide_mode(self, scores: List[float], tau: float = 0.32, is_greeting: bool = False) -> str:
        if is_greeting:
            return "smalltalk"
        if not scores:
            return "no_context"
        if max(scores) < tau:
            return "low_conf"
        return "rag"

    def _is_greeting(self, question: str) -> bool:
        q = question.lower().strip()
        return q in {"salut", "bonjour", "hello", "hi"} or q.startswith(("salut ", "bonjour "))

    # ------------------ Preview / Debug ------------------
    def preview_context(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        k = top_k or self.get_adaptive_top_k(question)
        context, nodes, scores = self.retrieve_context_hybrid(question, top_k=k)
        mode = self._decide_mode(scores, tau=0.32, is_greeting=self._is_greeting(question))
        items = []
        for i, (sc, nd) in enumerate(zip(scores, nodes), start=1):
            node_id = getattr(nd, "node_id", None) or getattr(nd, "id_", None) or f"node_{i}"
            text = nd.get_content()
            items.append({
                "rank": i,
                "score": float(sc),
                "node_id": str(node_id),
                "len": len(text),
                "text": text[:600],
            })
        return {
            "question": question,
            "mode": mode,
            "top_k": k,
            "items": items,
            "alpha_lexical": 0.35,
            "mmr_lambda": 0.6,
            "use_rrf": True,
        }

    def format_preview_md(self, prev: Dict[str, Any]) -> str:
        buf = io.StringIO()
        buf.write(f"# Debug preview — {prev.get('mode','?')}\n\n")
        buf.write(f"**Question**: {prev.get('question','')}  \n")
        buf.write(f"Top‑K: {prev.get('top_k','?')}  \n")
        buf.write(f"alpha_lexical={prev.get('alpha_lexical')}, mmr_lambda={prev.get('mmr_lambda')}, use_rrf={prev.get('use_rrf')}  \n\n")
        buf.write("## Top K chunks\n")
        for it in prev.get("items", []):
            buf.write(f"### #{it['rank']} | score={it['score']:.4f} | len={it['len']}\n")
            buf.write(it["text"].replace("\n", " ")[:1000] + "\n\n")
        return buf.getvalue()

    def save_preview(self, prev: Dict[str, Any], base_dir: Optional[Union[str, os.PathLike]] = None, filename: Optional[str] = None) -> Dict[str, str]:
        base_path: Path = Path(base_dir) if base_dir is not None else Path(os.getenv("RAG_DEBUG_DIR", "runs/debug_previews"))
        base_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        q = prev.get("question") or prev.get("query") or ""
        stem = filename or f"{ts}_{_slugify(q)}"
        json_path = base_path / f"{stem}.json"
        md_path = base_path / f"{stem}.md"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(prev, f, ensure_ascii=False, indent=2)
        md = self.format_preview_md(prev)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info("📝 Preview saved: %s | %s", json_path, md_path)
        return {"json": str(json_path), "md": str(md_path)}

    # ------------------ Public API (LLM-free for now) ------------------
    def ask(self, question: str, debug_preview: bool = False) -> str:
        top_k = self.get_adaptive_top_k(question)
        context, _, scores = self.retrieve_context_hybrid(question, top_k=top_k)
        mode = self._decide_mode(scores, tau=0.32, is_greeting=self._is_greeting(question))
        prev = self.preview_context(question, top_k=top_k)
        if debug_preview or os.getenv("RAG_DEBUG") == "1":
            self.save_preview(prev)
        # Normally you would call the LLM here with (question, context).
        # For this file we just return the context for inspection.
        return context

    def ask_stream(self, question: str, debug_preview: bool = False) -> Iterable[str]:
        # Streaming stub; yields the context chunks (for quick check)
        output = self.ask(question, debug_preview=debug_preview)
        yield output

# ------------------ CLI ------------------
def main() -> None:
    p = argparse.ArgumentParser(description="RAG Hybrid Retrieval quick tester")
    p.add_argument("--vector_path", type=str, default=DEFAULT_VECTOR_PKL, help="Pickle file with list[str] chunk texts")
    p.add_argument("--index_path", type=str, default=DEFAULT_FAISS_INDEX, help="FAISS index (.faiss)")
    p.add_argument("--embed_model", type=str, default=DEFAULT_EMBED_MODEL, help="HF embedding model, e.g., intfloat/multilingual-e5-base")
    p.add_argument("--preview", type=str, default=None, help="Question to run retrieval on; if omitted, does nothing")
    p.add_argument("--top_k", type=int, default=8, help="Top-K after MMR")
    p.add_argument("--alpha_lexical", type=float, default=0.35)
    p.add_argument("--no_rrf", action="store_true")
    p.add_argument("--mmr_lambda", type=float, default=0.6)
    p.add_argument("--pool_size", type=int, default=64)
    args = p.parse_args()

    engine = RAGEngine(
        vector_path=args.vector_path,
        index_path=args.index_path,
        embed_model_name=args.embed_model,
    )

    if args.preview:
        prev = engine.preview_context(args.preview, top_k=args.top_k)
        paths = engine.save_preview(prev)
        print(json.dumps({"saved": paths, "mode": prev.get("mode")}, ensure_ascii=False, indent=2))
    else:
        print("Nothing to do. Pass --preview 'your question' to run a retrieval.")

if __name__ == "__main__":
    main()
