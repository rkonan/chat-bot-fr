# test_preview.py
import argparse
from rag_engine_hybrid import RAGEngine

def format_preview(question, context, nodes, scores):
    lines = []
    lines.append(f"### Question\n{question}\n")
    lines.append(f"**Top‑K**: {len(nodes)}\n")
    lines.append("### Top K chunks (rerank cos_sim)")
    for i, (sc, nd) in enumerate(zip(scores, nodes), start=1):
        node_id = getattr(nd, "node_id", None) or getattr(nd, "id_", None) or f"node_{i}"
        txt = nd.get_content()
        snippet = txt.replace("\n", " ")[:380]
        more = "…" if len(txt) > 380 else ""
        lines.append(
            f"- **#{i}** | score={sc:.4f} | id=`{node_id}` | len={len(txt)}\n"
            f"  \n> {snippet}{more}\n"
        )
    lines.append("\n---\n### Contexte concaténé (troncature éventuelle)\n")
    lines.append(context)
    return "\n".join(lines)

def run_once(engine, question, k):
    context, nodes, scores = engine.retrieve_context(question, top_k=k)
    print(format_preview(question, context, nodes, scores))
    print("\n" + "="*80 + "\n")

def main():
    ap = argparse.ArgumentParser("RAG Preview (sans LLM)")
    ap.add_argument("--vector", required=True, help="ex: vectordb_docling/chunks.pkl")
    ap.add_argument("--index", required=True, help="ex: vectordb_docling/index.faiss")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--q", nargs="+", required=True, help="Une ou plusieurs questions")
    ap.add_argument("--model", default="nous-hermes-2-mistral-7b-dpo.Q4_K_M")  # ignoré ici
    ap.add_argument("--ollama_host", default=None)
    args = ap.parse_args()

    eng = RAGEngine(
        model_name=args.model,
        vector_path=args.vector,
        index_path=args.index,
        ollama_host=args.ollama_host,
    )

    for q in args.q:
        run_once(eng, q, args.k)

if __name__ == "__main__":
    main()
