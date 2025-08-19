from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

@dataclass
class RetrievedItem:
    node_id: str
    text: str
    score: float
    meta: Optional[Dict[str, Any]] = None

class Retriever(Protocol):
    def retrieve(self, question: str, top_k: int = 10) -> List[RetrievedItem]: