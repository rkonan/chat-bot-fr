#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_rewriters.py
Benchmark plusieurs modèles Ollama pour la réécriture de question (3 variantes en un seul appel).
- Mesure le temps d'appel et calcule tokens/sec si dispo.
- Parse la sortie au format numéroté 1)/2)/3).
- Supporte plusieurs runs par modèle pour moyenne/écart-type.
- Sorties JSON et CSV optionnelles.

Usage:
  python benchmark_rewriters.py "Votre question ici"
  python benchmark_rewriters.py "Votre question ici" --models phi3:mini qwen2:1.5b gemma2:2b --runs 3 --json out.json --csv out.csv
"""

import argparse
import csv
import json
import re
import statistics
import sys
import time
from typing import List, Dict, Any

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT_TEMPLATE = """Donne exactement trois réécritures DIFFÉRENTES de la question suivante, en français clair et concis,
sans ajouter d’informations ni y répondre.
Réponds uniquement au format :
1) ...
2) ...
3) ...

Question : {question}
"""

def parse_numbered_list(text: str) -> List[str]:
    """Extrait les lignes 1)/2)/3) et nettoie."""
    items = re.findall(r'^\\s*[123]\\)\\s*(.+?)\\s*$', text, flags=re.M)
    items = [re.sub(r'\\s+', ' ', s).strip().strip('"\'' ) for s in items if s.strip()]
    # dédup en gardant l'ordre
    seen, uniq = set(), []
    for s in items:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq[:3]

def call_ollama_once(model: str, question: str, url: str, temperature: float = 0.35,
                     num_predict: int = 120, top_k: int = 40, repeat_penalty: float = 1.1,
                     timeout: int = 180) -> Dict[str, Any]:
    """Fait un appel streaming à /api/generate et retourne mesures + résultats."""
    prompt = PROMPT_TEMPLATE.format(question=question)
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty
        }
    }

    t0 = time.time()
    r = requests.post(url, json=payload, stream=True, timeout=timeout)
    chunks, last_meta = [], {}
    for line in r.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        # accumulate text
        chunks.append(data.get("response", ""))
        # keep last meta for token counts if present
        for k in ("eval_count", "eval_duration", "prompt_eval_count", "prompt_eval_duration", "total_duration"):
            if k in data:
                last_meta[k] = data[k]
        if data.get("done"):
            break
    t1 = time.time()

    raw_text = "".join(chunks).strip()
    queries = parse_numbered_list(raw_text)
    # si <3, compléter simplement sans nouvel appel (copie/variantes minimales)
    while len(queries) < 3:
        base = queries[-1] if queries else question
        cand = base
        if not cand.endswith("?"):
            cand += " ?"
        if cand not in queries:
            queries.append(cand)
        else:
            # ajoute une petite variation bénigne
            cand2 = cand.replace("Quel", "Lequel") if "Quel" in cand else cand + " "
            if cand2 not in queries:
                queries.append(cand2)
            else:
                break

    # Token/sec si méta dispo
    eval_count = last_meta.get("eval_count")
    eval_duration_ns = last_meta.get("eval_duration")
    tps = None
    if isinstance(eval_count, int) and isinstance(eval_duration_ns, int) and eval_duration_ns > 0:
        tps = eval_count / (eval_duration_ns / 1e9)

    return {
        "ok": len(queries) >= 1,
        "model": model,
        "question": question,
        "queries": queries[:3],
        "gen_time_sec": round(t1 - t0, 3),
        "tokens": eval_count,
        "tokens_per_sec": round(tps, 2) if tps is not None else None,
        "meta": last_meta,
        "raw": raw_text[:5000]
    }

def run_benchmark(question: str, models: List[str], runs: int, url: str) -> Dict[str, Any]:
    results = []
    for model in models:
        per_runs = []
        for i in range(runs):
            try:
                res = call_ollama_once(model, question, url)
            except Exception as e:
                res = {
                    "ok": False,
                    "model": model,
                    "question": question,
                    "error": str(e),
                    "queries": [],
                    "gen_time_sec": None,
                    "tokens": None,
                    "tokens_per_sec": None,
                    "meta": {}
                }
            per_runs.append(res)
            # print progress per run
            print(f"[{model}] run {i+1}/{runs}: ok={per_runs[-1]['ok']} time={per_runs[-1]['gen_time_sec']}s tps={per_runs[-1]['tokens_per_sec']}")
        # agrégation
        times = [r["gen_time_sec"] for r in per_runs if r["gen_time_sec"] is not None]
        tps_vals = [r["tokens_per_sec"] for r in per_runs if r["tokens_per_sec"] is not None]
        agg = {
            "model": model,
            "runs": runs,
            "ok_runs": sum(1 for r in per_runs if r["ok"]),
            "avg_time_sec": round(statistics.mean(times), 3) if times else None,
            "p95_time_sec": round(statistics.quantiles(times, n=20)[18], 3) if len(times) >= 20 else (max(times) if times else None),
            "avg_tps": round(statistics.mean(tps_vals), 2) if tps_vals else None,
            "examples": per_runs[0]["queries"] if per_runs and per_runs[0].get("queries") else [],
            "details": per_runs,
        }
        results.append(agg)
        print(f"==> {model}: avg_time={agg['avg_time_sec']}s avg_tps={agg['avg_tps']} ok_runs={agg['ok_runs']}/{runs}")
    return {"question": question, "results": results}

def write_json(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_csv(path: str, bench: Dict[str, Any]):
    rows = []
    for r in bench["results"]:
        rows.append({
            "model": r["model"],
            "runs": r["runs"],
            "ok_runs": r["ok_runs"],
            "avg_time_sec": r["avg_time_sec"],
            "p95_time_sec": r["p95_time_sec"],
            "avg_tps": r["avg_tps"],
            "example_1": (r["examples"][0] if len(r["examples"]) > 0 else ""),
            "example_2": (r["examples"][1] if len(r["examples"]) > 1 else ""),
            "example_3": (r["examples"][2] if len(r["examples"]) > 2 else ""),
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "model","runs","ok_runs","avg_time_sec","p95_time_sec","avg_tps","example_1","example_2","example_3"
        ])
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, help="Question à réécrire")
    parser.add_argument("--models", nargs="*", default=[
        "phi3:mini",
        "phi3",
        "qwen2:1.5b-instruct",
        "gemma2:2b",
        "stablelm2:1.6b",
        "qwen2.5:3b-instruct-q4_K_M",
    ], help="Liste des modèles Ollama à tester (ils doivent être déjà pull)")
    parser.add_argument("--runs", type=int, default=1, help="Nombre de runs par modèle (>=3 recommandé pour moyenne stable)")
    parser.add_argument("--json", type=str, default=None, help="Chemin de sortie JSON")
    parser.add_argument("--csv", type=str, default=None, help="Chemin de sortie CSV")
    parser.add_argument("--url", type=str, default=DEFAULT_OLLAMA_URL, help="Endpoint Ollama /api/generate")
    args = parser.parse_args()

    bench = run_benchmark(args.question, args.models, args.runs, args.url)

    # impression console courte
    print("\\n=== Résumé ===")
    for r in bench["results"]:
        print(f"{r['model']}: avg_time={r['avg_time_sec']}s avg_tps={r['avg_tps']} ok_runs={r['ok_runs']}/{r['runs']}")

    if args.json:
        write_json(args.json, bench)
        print(f"JSON écrit: {args.json}")
    if args.csv:
        write_csv(args.csv, bench)
        print(f"CSV écrit: {args.csv}")

if __name__ == "__main__":
    main()
