from __future__ import annotations
from typing import List
from dataclasses import dataclass
from .base import RetrievedItem, Retriever
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers.util import cos_sim

@dataclass
class DenseResources:
    index: VectorStoreIndex
    embed_model: HuggingFaceEmbedding
    nodes: List[TextNode]

class DenseRetriever(Retriever):
    def __init__(self, resources: DenseResources, pool_k: int = 24):
        self.r = resources
        self.pool_k = pool_k

    def retrieve(self, question: str, top_k: int = 10) -> List[RetrievedItem]:
        retriever = self.r.index.as_retriever(similarity_top_k=max(top_k, self.pool_k))
        retrieved_nodes = retriever.retrieve(question)
        q_emb = self.r.embed_model.get_query_embedding(question)
        scored = []
        for n in retrieved_nodes:
            emb = self.r.embed_model.get_text_embedding(n.get_content())
            s = cos_sim(q_emb, emb).item()
            scored.append((s, n))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[RetrievedItem] = []
        for s, n in scored[:top_k]:
            nid = getattr(n, "node_id", None) or getattr(n, "id_", None) or ""
            out.append(RetrievedItem(node_id=str(nid), text=n.get_content(), score=float(s), meta={"backend":"dense"}))
        return out
