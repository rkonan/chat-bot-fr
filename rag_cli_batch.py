
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# CLI wrapper around rag_engine_hybrid_v1_dropin.py with batch mode, retrieval-only, and Ollama options.

import os, sys, json
from pathlib import Path
from datetime import datetime
import argparse

# Import engine (expects the file to be colocated or on PYTHONPATH)
try:
    from rag_engine_hybrid_v1_dropin import RAGEngine, DEFAULT_VECTOR_PKL, DEFAULT_FAISS_INDEX, DEFAULT_EMBED_MODEL
except Exception as e:
    print("Failed to import RAGEngine from rag_engine_hybrid_v1_dropin.py:", e, file=sys.stderr)
    sys.exit(2)

def main():
    p = argparse.ArgumentParser(description="RAG Hybrid + Ollama — CLI (batch + preview)")
    p.add_argument("--vector_path", type=str, default=DEFAULT_VECTOR_PKL, help="Pickle file with chunk texts (list[str])")
    p.add_argument("--index_path", type=str, default=DEFAULT_FAISS_INDEX, help="FAISS index file")
    # Two model knobs: embedding vs LLM
    p.add_argument("--embed_model", type=str, default=DEFAULT_EMBED_MODEL, help="HF embedding model (for retrieval)")
    p.add_argument("--model_name", type=str, default=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"), help="Ollama model name for generation")
    p.add_argument("--ollama_host", type=str, default=os.getenv("OLLAMA_HOST", "http://localhost:11434"), help="Ollama host URL")
    p.add_argument("--ollama_timeout", type=int, default=int(os.getenv("OLLAMA_TIMEOUT", "300")), help="Ollama HTTP timeout (seconds)")

    p.add_argument("--top_k", type=int, default=8, help="Top-K after MMR")
    p.add_argument("--preview", type=str, default=None, help="Single question to preview")
    p.add_argument("--questions_file", type=str, default=None, help="Path to a text file with one question per line")
    p.add_argument("--output", type=str, default=None, help="Output JSONL path (batch mode). Default: runs/batch_YYYYMMDD-HHMMSS.jsonl")
    p.add_argument("--no_generate", action="store_true", help="Retrieval only (do not call Ollama)")
    args = p.parse_args()

    # Construct engine with separated embed vs LLM models
    rag = RAGEngine(
        model_name=args.model_name,
        ollama_host=args.ollama_host,
        vector_path=args.vector_path,
        index_path=args.index_path,
        embed_model_name=args.embed_model,
    )
    # Patch timeout if available
    if getattr(rag, "ollama_client", None) is not None:
        rag.ollama_client.timeout = args.ollama_timeout

    # Single preview mode
    if args.preview:
        ctx, nodes, scores = rag.retrieve_context_hybrid(args.preview, top_k=args.top_k)
        print("=== Context ===")
        print(ctx[:1200] + ("..." if len(ctx) > 1200 else ""))
        print("\n=== Nodes ===")
        for i, (n, sc) in enumerate(zip(nodes, scores), 1):
            print(f"#{i} score={sc:.4f} len={len(n.get_content())}")
        if not args.no_generate:
            print("\n=== Answer (Ollama) ===")
            try:
                print(rag.generate(args.preview, ctx))
            except Exception as e:
                print(f"[GENERATION ERROR] {e}", file=sys.stderr)
        return

    # Batch mode
    if args.questions_file:
        infile = Path(args.questions_file)
        if not infile.exists():
            print(f"Questions file not found: {infile}", file=sys.stderr)
            sys.exit(2)

        outdir = Path("runs")
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = Path(args.output) if args.output else outdir / f"batch_{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"

        n = 0
        with open(infile, "r", encoding="utf-8") as fin, open(outpath, "w", encoding="utf-8") as fout:
            for line in fin:
                q = line.strip()
                if not q:
                    continue
                try:
                    ctx, _, _ = rag.retrieve_context_hybrid(q, top_k=args.top_k)
                    if args.no_generate:
                        rec = {"question": q, "answer": "", "context": ctx, "answer_error": ""}
                    else:
                        try:
                            ans = rag.generate(q, ctx)
                            rec = {"question": q, "answer": ans, "context": ctx, "answer_error": ""}
                        except Exception as ge:
                            rec = {"question": q, "answer": "", "context": ctx, "answer_error": f"{type(ge).__name__}: {ge}"}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                except Exception as e:
                    # Retrieval failure shouldn't kill the whole batch
                    rec = {"question": q, "answer": "", "context": "", "answer_error": f"RETRIEVAL {type(e).__name__}: {e}"}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
        print(f"Wrote {n} results to {outpath}")
        return

    print("Nothing to do. Use --preview 'question' or --questions_file path.")

if __name__ == "__main__":
    main()
