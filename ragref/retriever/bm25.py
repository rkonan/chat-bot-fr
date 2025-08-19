from __future__ import annotations
import math
from typing import Dict, List, Tuple
from .base import RetrievedItem, Retriever
from ..utils.text import tok, keywords

class BM25Okapi:
    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.df: Dict[str, int] = {}
        self.tf: List[Dict[str, int]] = []
        self.doc_len: List[int] = []
        for tokens in docs:
            tf_i: Dict[str, int] = {}
            for t in tokens: tf_i[t] = tf_i.get(t, 0) + 1
            self.tf.append(tf_i)
            dl = sum(tf_i.values()); self.doc_len.append(dl)
            for t in tf_i: self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = sum(self.doc_len)/self.N if self.N else 0.0
        self.idf = {t: math.log(1 + (self.N - df + 0.5)/(df + 0.5)) for t, df in self.df.items()}

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0]*self.N
        for i, tf_i in enumerate(self.tf):
            denom = self.k1 * (1 - self.b + self.b * (self.doc_len[i]/(self.avgdl + 1e-9)))
            s = 0.0
            for t in query_tokens:
                if t not in tf_i: continue
                idf_t = self.idf.get(t, 0.0)
                num = tf_i[t]*(self.k1 + 1)
                s += idf_t * (num/(tf_i[t] + denom + 1e-9))
            scores[i] = s
        return scores

    def search(self, query_tokens: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        sc = self.get_scores(query_tokens)
        if not sc: return []
        import numpy as _np
        k = min(top_k, len(sc))
        idx = _np.argpartition(-_np.array(sc, dtype=_np.float32), k-1)[:k]
        idx = idx[_np.argsort(-_np.array([sc[i] for i in idx]))]
        return [(int(i), float(sc[i])) for i in idx]

class BM25Retriever(Retriever):
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.docs_tok = [tok(t) for t in corpus]
        self.bm25 = BM25Okapi(self.docs_tok)

    def retrieve(self, question: str, top_k: int = 10) -> List[RetrievedItem]:
        q_tok = keywords(question)
        hits = self.bm25.search(q_tok, top_k=top_k)
        out: List[RetrievedItem] = []
        for idx, score in hits:
            text = self.corpus[idx]
            out.append(RetrievedItem(node_id=str(idx), text=text, score=float(score), meta={"backend":"bm25"}))
        return out