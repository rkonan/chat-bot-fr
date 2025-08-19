from __future__ import annotations
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from .base import RetrievedItem, Retriever
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from ..utils.text import keywords, tok
import numpy as np


def _mmr_select(vecs: List[List[float]], k: int, lam: float = 0.5) -> List[int]:
    V = np.asarray(vecs, dtype=np.float32)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    q = V.mean(axis=0, keepdims=True)
    rel = (V @ q.T).ravel()
    chosen, cand = [], list(range(len(vecs)))
    while cand and len(chosen) < k:
        if not chosen:
            j = int(np.argmax(rel[cand])); pick = cand[j]; cand.pop(j); chosen.append(pick); continue
        div = np.max(V[cand] @ V[chosen].T, axis=1)
        mmr = lam*rel[cand] - (1-lam)*div
        j = int(np.argmax(mmr)); pick = cand[j]; cand.pop(j); chosen.append(pick)
    return chosen

@dataclass
class HybridConfig:
    use_rrf: bool = True
    rrf_k: int = 60
    use_lexical_filter: bool = True
    min_keyword_hits: int = 1
    alpha_top: float = 0.96
    score_min: float = 0.0
    use_mmr: bool = True
    mmr_lambda: float = 0.5
    context_clip: int = 500

class HybridRetriever(Retriever):
    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, embed_model, cfg: HybridConfig = HybridConfig()):
        self.dense = dense
        self.bm25 = bm25
        self.embed_model = embed_model
        self.cfg = cfg
        self._nodes_cache = self.dense.r.nodes  # to align ids for RRF

    def _rrf(self, dense_items: List[RetrievedItem], bm25_items: List[RetrievedItem]) -> List[RetrievedItem]:
        rank_emb = {it.node_id: r+1 for r, it in enumerate(dense_items)}
        rank_bm  = {it.node_id: r+1 for r, it in enumerate(bm25_items)}
        fused: List[Tuple[float, RetrievedItem]] = []
        for it in dense_items:
            re_ = rank_emb.get(it.node_id, 10**6)
            rb_ = rank_bm.get(it.node_id, 10**6)
            s = 1.0/(self.cfg.rrf_k + re_) + 1.0/(self.cfg.rrf_k + rb_)
            fused.append((s, it))
        fused.sort(key=lambda x: -x[0])
        return [it for _, it in fused]

    def retrieve(self, question: str, top_k: int = 10) -> List[RetrievedItem]:
        # pool
        pool_k = max(top_k, 24)
        dense_items = self.dense.retrieve(question, top_k=pool_k)
        bm25_items  = self.bm25.retrieve(question, top_k=pool_k)
        items = dense_items
        if self.cfg.use_rrf:
            items = self._rrf(dense_items, bm25_items)

        # lexical filter
        if self.cfg.use_lexical_filter:
            kws = set(keywords(question))
            filtered: List[RetrievedItem] = []
            for it in items:
                hits = sum(1 for t in tok(it.text) if t in kws)
                if hits >= self.cfg.min_keyword_hits:
                    filtered.append(it)
            if filtered:
                items = filtered

        # thresholds
        if items:
            top_score = items[0].score
            items = [it for it in items if (it.score >= top_score*self.cfg.alpha_top) and (it.score >= self.cfg.score_min)]

        # MMR diversification on embeddings
        if self.cfg.use_mmr and len(items) > top_k:
            q_emb = self.embed_model.get_query_embedding(question)
            vecs: List[List[float]] = []
            for it in items:
                vecs.append(self.embed_model.get_text_embedding(it.text))
            mmr_ids = _mmr_select(vecs, top_k, self.cfg.mmr_lambda)
            items = [items[i] for i in mmr_ids]

        return items[:top_k]
