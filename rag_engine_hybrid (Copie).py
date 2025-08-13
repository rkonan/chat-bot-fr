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
import argparse

# === Logger ===
logger = logging.getLogger("RAGHybrid")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)

MAX_TOKENS = 64
DEFAULT_STOPS = ["### Réponse:", "\n\n", "###"]

# ---------- Client Ollama ----------
class OllamaClient:
    def __init__(self, model: str, host: Optional[str] = None, timeout: int = 300):
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout
        self._gen_url = self.host.rstrip("/") + "/api/generate"
        self._chat_url = self.host.rstrip("/") + "/api/chat"

    # /api/generate (RAG)
    def generate(self, prompt: str, stop: Optional[List[str]] = None,
                 max_tokens: Optional[int] = None, stream: bool = False,
                 raw: bool = False) -> str | Iterable[str]:
        payload: Dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": stream}
        if raw:
            payload["raw"] = True
        if stop:
            payload["stop"] = stop
        if max_tokens is not None:
            payload["num_predict"] = int(max_tokens)

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
        return r.json().get("response", "")

    # /api/chat (LLM pur)
    def chat(self, messages: list, stream: bool = False) -> Dict | Iterable[str]:
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": stream}

        if stream:
            def token_gen():
                with requests.post(self._chat_url, json=payload, stream=True, timeout=self.timeout) as r:
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        data = json.loads(line)
                        if "message" in data and not data.get("done"):
                            yield data["message"]["content"]
                        if data.get("done"):
                            break
            return token_gen()

        r = requests.post(self._chat_url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

# ---------- RAG Engine (lazy) ----------
class RAGEngine:
    def __init__(self, model_name: str, vector_path: str, index_path: str,
                 model_threads: int = 4, ollama_host: Optional[str] = None):
        logger.info("📦 Initialisation du moteur (hybride, lazy, no-opts)...")

        self.llm = OllamaClient(model=model_name, host=ollama_host)

        # chemins pour chargement différé
        self.vector_path = vector_path
        self.index_path = index_path

        # objets RAG (chargés à la demande)
        self.embed_model: Optional[HuggingFaceEmbedding] = None
        self.index: Optional[VectorStoreIndex] = None
        self._loaded = False

        # petit cache de chat (historique simple si tu veux l’étendre plus tard)
        self.chat_history: List[Dict[str, str]] = []

        logger.info("✅ Moteur prêt (FAISS/chunks non chargés).")

    # ---------- lazy loader ----------
    def _ensure_loaded(self):
        if self._loaded:
            return
        t0 = time.perf_counter()
        logger.info("⏳ Chargement lazy des données RAG (FAISS + chunks + embeddings)...")

        with open(self.vector_path, "rb") as f:
            chunk_texts: List[str] = pickle.load(f)
        nodes = [TextNode(text=chunk) for chunk in chunk_texts]

        faiss_index = faiss.read_index(self.index_path)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        self.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")
        self.index = VectorStoreIndex(nodes=nodes, embed_model=self.embed_model, vector_store=vector_store)

        self._loaded = True
        logger.info(f"✅ RAG chargé en {time.perf_counter() - t0:.2f}s (lazy).")

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
        q = question.lower()

        # GK → pas de RAG
        if self._looks_general_knowledge(q):
            return False

        # indices “docs”
        doc_keywords = (
            "document", "docs", "procédure", "politique", "policy",
            "manuel", "guide", "pdf", "docling", "selon", "dans le contexte",
            "page", "section", "chapitre", "référence", "références", "conformément",
            "note technique", "spécification", "spec", "architecture", "adr"
        )
        if any(k in q for k in doc_keywords):
            return True

        # longueur → probable RAG
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

    # ---------- PREVIEW / DEBUG (sans LLM) ----------
    def preview_context(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Retourne un dict avec:
          - mode_decision: 'rag' ou 'llm' selon _decide_mode
          - top_k_effectif
          - context (concaténé)
          - items: liste de {rank, score, node_id, snippet, full_len}
        Ne fait AUCUN appel LLM.
        """
        k = top_k or self.get_adaptive_top_k(question)
        context, nodes, scores = self.retrieve_context(question, k)
        mode = self._decide_mode(scores, tau=0.32, is_greeting=self._is_greeting(question))

        items = []
        for i, (sc, nd) in enumerate(zip(scores, nodes), start=1):
            node_id = getattr(nd, "node_id", None) or getattr(nd, "id_", None) or f"node_{i}"
            text = nd.get_content()
            items.append({
                "rank": i,
                "score": float(sc),
                "node_id": str(node_id),
                "snippet": text[:400],
                "full_len": len(text),
            })

        preview = {
            "question": question,
            "mode_decision": mode,
            "top_k_effectif": k,
            "context": context,
            "items": items,
        }
        return preview

    def format_preview_md(self, preview: Dict[str, Any]) -> str:
        """
        Rend une vue Markdown lisible du preview_context.
        """
        lines = []
        lines.append(f"### Question\n{preview['question']}\n")
        lines.append(f"**Mode décidé**: `{preview['mode_decision']}`   |   **Top‑K**: {preview['top_k_effectif']}\n")
        lines.append("### Top K chunks")
        for it in preview["items"]:
            snippet = it['snippet'].replace('\n', ' ')
            trailing = "…" if it['full_len'] > len(it['snippet']) else ""
            lines.append(
                f"- **#{it['rank']}** | score={it['score']:.4f} | id=`{it['node_id']}` | len={it['full_len']}\n"
                f"  \n> {snippet[:380]}{trailing}\n"
            )
        lines.append("\n---\n### Contexte concaténé (troncature éventuelle)\n")
        lines.append(preview["context"])
        return "\n".join(lines)

    def log_preview(self, preview: Dict[str, Any]) -> None:
        """
        Log synthétique (INFO) des chunks sélectionnés.
        """
        logger.info(f"🔎 Mode: {preview['mode_decision']} | K={preview['top_k_effectif']}")
        for it in preview["items"]:
            head = it['snippet'][:120].replace('\n', ' ')
            tail = "…" if it['full_len'] > 120 else ""
            logger.info(f"  #{it['rank']} score={it['score']:.4f} id={it['node_id']} | {head}{tail}")

    # ---------- API publique ----------
    # Non-stream
    def ask(self, question: str, debug_preview: bool = False) -> str:
        logger.info(f"💬 [Non-stream] Question : {question}")
        is_hello = self._is_greeting(question)

        use_rag = (self._loaded and not is_hello) or (not self._loaded and self._should_use_rag_fast(question))
        if use_rag:
            top_k = self.get_adaptive_top_k(question)
            context, _, scores = self.retrieve_context(question, top_k)
            mode = self._decide_mode(scores, tau=0.32, is_greeting=is_hello)

            # Debug preview (sans LLM)
            if debug_preview or os.getenv("RAG_DEBUG") == "1":
                prev = self.preview_context(question, top_k=top_k)
                self.log_preview(prev)
                if debug_preview:
                    return self.format_preview_md(prev)

            if mode == "rag":
                prompt = (
                    "Instruction: Réponds uniquement à partir du contexte. "
                    "Si la réponse n'est pas déductible, réponds exactement: \"Information non présente dans le contexte.\""
                    f"\n\nContexte :\n{context}\n\nQuestion : {question}\nRéponse :"
                )
                return self.llm.generate(prompt, stop=DEFAULT_STOPS, max_tokens=MAX_TOKENS, stream=False) or ""

        # LLM pur → /api/chat (pas d’options)
        messages = self.chat_history + [{"role": "user", "content": question}]
        result = self.llm.chat(messages, stream=False)
        content = result.get("message", {}).get("content", "")
        # maj historique (simple, limite 8 tours pour rester léger)
        self.chat_history.extend([{"role": "user", "content": question},
                                  {"role": "assistant", "content": content}])
        if len(self.chat_history) > 16:
            self.chat_history = self.chat_history[-16:]
        return content

    # Stream
    def ask_stream(self, question: str, debug_preview: bool = False) -> Iterable[str]:
        logger.info(f"💬 [Stream] Question : {question}")
        is_hello = self._is_greeting(question)

        use_rag = (self._loaded and not is_hello) or (not self._loaded and self._should_use_rag_fast(question))
        if use_rag:
            top_k = self.get_adaptive_top_k(question)
            context, _, scores = self.retrieve_context(question, top_k)
            mode = self._decide_mode(scores, tau=0.32, is_greeting=is_hello)

            # Debug preview (sans LLM)
            if debug_preview or os.getenv("RAG_DEBUG") == "1":
                prev = self.preview_context(question, top_k=top_k)
                self.log_preview(prev)
                if debug_preview:
                    yield self.format_preview_md(prev)
                    return

            if mode == "rag":
                prompt = (
                    "Instruction: Réponds uniquement à partir du contexte. "
                    "Si la réponse n'est pas déductible, réponds exactement: \"Information non présente dans le contexte.\""
                    f"\n\nContexte :\n{context}\n\nQuestion : {question}\nRéponse :"
                )
                logger.info("📡 Début streaming (RAG via /api/generate)...")
                for tok in self.llm.generate(prompt, stop=DEFAULT_STOPS, max_tokens=MAX_TOKENS, stream=True):
                    yield tok
                logger.info("📡 Fin streaming (RAG).")
                return

        # LLM pur → /api/chat (stream)
        messages = self.chat_history + [{"role": "user", "content": question}]
        logger.info("📡 Début streaming (LLM pur via /api/chat)...")
        acc = ""
        for tok in self.llm.chat(messages, stream=True):
            acc += tok
            yield tok
        logger.info("📡 Fin streaming (LLM pur).")
        # maj historique
        self.chat_history.extend([{"role": "user", "content": question},
                                  {"role": "assistant", "content": acc}])
        if len(self.chat_history) > 16:
            self.chat_history = self.chat_history[-16:]

# ---------- CLI utilitaire ----------
def _cli():
    parser = argparse.ArgumentParser(description="RAG Engine Hybrid - Preview/Ask")
    parser.add_argument("--model", default="nous-hermes-2-mistral-7b-dpo.Q4_K_M")
    parser.add_argument("--ollama_host", default=None)
    parser.add_argument("--vector_path", required=True, help="Chemin vers le pickle des chunks (ex: vectordb_docling/chunks.pkl)")
    parser.add_argument("--index_path", required=True, help="Chemin vers l'index FAISS (ex: vectordb_docling/index.faiss)")
    parser.add_argument("--preview", type=str, help="Question à prévisualiser (contexte uniquement, pas d'appel LLM)")
    parser.add_argument("--ask", type=str, help="Question avec exécution complète (peut basculer RAG/LLM)")
    parser.add_argument("--stream", action="store_true", help="Mode streaming pour --ask")
    parser.add_argument("--debug_preview", action="store_true", help="Forcer l'impression du contexte dans ask/ask_stream")
    args = parser.parse_args()

    eng = RAGEngine(
        model_name=args.model,
        vector_path=args.vector_path,
        index_path=args.index_path,
        ollama_host=args.ollama_host
    )

    if args.preview:
        prev = eng.preview_context(args.preview)
        print(eng.format_preview_md(prev))
        return

    if args.ask:
        if args.stream:
            for tok in eng.ask_stream(args.ask, debug_preview=args.debug_preview):
                print(tok, end="", flush=True)
            print()
        else:
            out = eng.ask(args.ask, debug_preview=args.debug_preview)
            print(out)
        return

    parser.print_help()

if __name__ == "__main__":
    _cli()
