from __future__ import annotations
from typing import List
from dataclasses import dataclass
from ..clients.ollama_client import OllamaClient

@dataclass
class ExpansionConfig:
    strategy: str = "none"   # none|llm
    n: int = 2               # number of rewrites

class QuestionExpander:
    """Small, swappable service to expand/rewriter queries.
    NOTE: In the refactor, **HybridRetriever** owns the expander and fuses
    results across expansions (not the engine).
    """
    def __init__(self, cfg: ExpansionConfig, llm: OllamaClient | None = None):
        self.cfg = cfg
        self.llm = llm

    def expand(self, q: str) -> List[str]:
        if self.cfg.strategy == "llm" and self.llm:
            prompt = (
                "Tu es un réécrivain de requêtes pour la recherche dense + BM25.\n"
                "Réécris la question suivante en {n} variantes courtes et spécifiques, une par ligne, sans numérotation.\n\n"
                f"Question: {q}\nVariantes:".replace("{n}", str(self.cfg.n))
            )
            out = self.llm.generate(prompt, max_tokens=256, stream=False)
            lines = [l.strip(" -•	") for l in (out or "").splitlines() if l.strip()]
            # Always include the original at index 0
            return [q] + lines[: self.cfg.n]
        return [q]