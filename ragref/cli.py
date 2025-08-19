from __future__ import annotations
import argparse
from .engine import RAGEngine


def main():
    parser = argparse.ArgumentParser(description="RAGRefactor — Engine + Retriever (separated)")
    parser.add_argument("--model", default="nous-hermes-2-mistral-7b-dpo.Q4_K_M")
    parser.add_argument("--ollama_host", default=None)
    parser.add_argument("--vector_path", required=True)
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--preview", type=str)
    parser.add_argument("--ask", type=str)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--debug_preview", action="store_true")
    args = parser.parse_args()

    eng = RAGEngine(
        model_name=args.model,
        vector_path=args.vector_path,
        index_path=args.index_path,
        ollama_host=args.ollama_host,
    )

    if args.preview:
        prev = eng.preview_context(args.preview)
        print(eng.format_preview_md(prev)); return

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
    main()