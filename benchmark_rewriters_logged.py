#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_rewriters_logged.py
Benchmark de réécriture de questions (3 variantes / 1 appel) pour plusieurs modèles Ollama,
avec LOGS détaillés pour comprendre chaque étape.

Fonctions :
- Appel /api/generate en streaming (un seul appel par modèle).
- Logs : payload (en debug), statut HTTP, temps jusqu'au premier octet (TTFB), durée de stream, temps total, nb de chunks.
- Parse la sortie au format numéroté 1)/2)/3) ; complète si <3 sans nouvel appel.
- Mesure tokens/sec si méta disponible (eval_count, eval_duration).
- Métriques qualité simples : longueur moyenne, diversité (distinct-2).
- Répétition (--runs N) + agrégation (moyenne, min, max). Exporte JSON/CSV.
- Verbosité contrôlable : --verbose / --debug.

Exemples :
  python benchmark_rewriters_logged.py "Quel modèle est le meilleur modèle etudié ?"
  python benchmark_rewriters_logged.py "Quel modèle..." --models phi3:mini qwen2:1.5b-instruct gemma2:2b stablelm2:1.6b --runs 3
  python benchmark_rewriters_logged.py "Quel modèle..." --runs 3 --json bench.json --csv bench.csv --verbose
"""

import argparse
import csv
import json
import logging
import re
import statistics
import sys
import time
from typing import List, Dict, Any, Tuple

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

# Liste par défaut alignée sur tes modèles installés
DEFAULT_MODELS = [
    "phi3:mini",
    "qwen2:1.5b-instruct",
    "gemma2:2b",
    "stablelm2:1.6b",
    "aroxima/gte-qwen2-1.5b-instruct:q5_k_m",
    "phi3",
    "tinydolphin",
    "tinyllama",
    "qwen2.5:3b-instruct-q4_K_M",
    "noushermes_rag",
    "granite3.3",
    "gemma3",
    "deepseek-r1",
    "mistral",


]

PROMPT_TEMPLATE = """Donne exactement trois réécritures DIFFÉRENTES de la question suivante, en français clair et concis,
sans ajouter d’informations ni y répondre.
Réponds uniquement au format :
1) ...
2) ...
3) ...

Question : {question}
"""

# -------------------------------
# Utils logging & metrics
# -------------------------------

def setup_logger(level: str) -> None:
    lvl = level.upper()
    if lvl not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        lvl = "INFO"
    logging.basicConfig(
        level=getattr(logging, lvl),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S"
    )

def distinct_2(texts: List[str]) -> float:
    """Calcule la diversité distinct-2 moyenne (bigrams uniques / total)."""
    if not texts:
        return 0.0
    scores = []
    for t in texts:
        toks = t.strip().split()
        if len(toks) < 2:
            scores.append(0.0)
            continue
        bigrams = [tuple(toks[i:i+2]) for i in range(len(toks)-1)]
        total = len(bigrams)
        uniq = len(set(bigrams))
        scores.append(uniq / total if total > 0 else 0.0)
    return round(sum(scores) / len(scores), 3)

def avg_len_words(texts: List[str]) -> float:
    if not texts:
        return 0.0
    return round(sum(len(t.split()) for t in texts) / len(texts), 2)

def parse_numbered_list(text: str) -> List[str]:
    """Extrait les lignes 1)/2)/3) et nettoie."""
    items = re.findall(r'^\s*[123]\)\s*(.+?)\s*$', text, flags=re.M)
    items = [re.sub(r'\s+', ' ', s).strip().strip('"\'' ) for s in items if s.strip()]
    # dédup en gardant l'ordre
    seen, uniq = set(), []
    for s in items:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq[:3]

def safe_complete_to_three(queries: List[str], question: str) -> List[str]:
    """Complète à 3 éléments sans nouvel appel (variantes bénignes)."""
    q = queries[:]
    while len(q) < 3:
        base = q[-1] if q else question
        cand = base.strip()
        if not cand.endswith("?"):
            cand += " ?"
        if cand not in q:
            q.append(cand)
        else:
            # petite variation : Quel -> Lequel, ou ajout ponctuel bénin
            c2 = cand.replace("Quel", "Lequel") if "Quel" in cand else (cand + " ...").strip()
            if c2 not in q:
                q.append(c2)
            else:
                break
    return q[:3]

# -------------------------------
# Core: single call with rich logs
# -------------------------------

def call_ollama_once(model: str, question: str, url: str,
                     temperature: float = 0.35, num_predict: int = 120,
                     top_k: int = 40, repeat_penalty: float = 1.1,
                     timeout: int = 180, verbose: bool = False) -> Dict[str, Any]:
    """Fait un appel streaming à /api/generate et retourne mesures + résultats + logs détaillés."""
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

    logging.info(f"[{model}] Prépare requête -> {url}")
    if verbose:
        logging.debug(f"[{model}] Payload: {json.dumps(payload, ensure_ascii=False)}")

    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, stream=True, timeout=timeout)
    except requests.RequestException as e:
        logging.error(f"[{model}] ERREUR réseau/HTTP: {e}")
        return {"ok": False, "model": model, "error": str(e), "queries": [], "gen_time_sec": None}

    logging.info(f"[{model}] HTTP status: {resp.status_code}")
    if resp.status_code != 200:
        try:
            err_body = resp.text[:500]
        except Exception:
            err_body = "<no-body>"
        logging.error(f"[{model}] HTTP {resp.status_code} body: {err_body}")
        return {"ok": False, "model": model, "error": f"HTTP {resp.status_code}: {err_body}", "queries": [], "gen_time_sec": None}

    # Stream
    chunks = []
    last_meta = {}
    first_chunk_time = None
    lines = 0
    for raw in resp.iter_lines():
        if not raw:
            continue
        if first_chunk_time is None:
            first_chunk_time = time.time()
            logging.info(f"[{model}] Premier octet reçu (TTFB): {first_chunk_time - t0:.3f}s")
        lines += 1
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as je:
            logging.warning(f"[{model}] JSON chunk mal formé (ligne {lines}): {je}")
            continue
        chunk_text = data.get("response", "")
        if chunk_text:
            chunks.append(chunk_text)
            if verbose and lines <= 5:
                logging.debug(f"[{model}] Chunk#{lines}: {chunk_text[:120]!r}")
        # récupérer les métadonnées utiles si présentes
        for k in ("eval_count", "eval_duration", "prompt_eval_count", "prompt_eval_duration", "total_duration"):
            if k in data:
                last_meta[k] = data[k]
        if data.get("done"):
            logging.info(f"[{model}] Signal 'done' reçu après {lines} chunks")
            break

    t1 = time.time()
    if first_chunk_time is None:
        first_chunk_time = t1  # pas de flux ?

    ttfb = first_chunk_time - t0
    stream_dur = t1 - first_chunk_time
    total = t1 - t0
    logging.info(f"[{model}] Durées: TTFB={ttfb:.3f}s | stream={stream_dur:.3f}s | total={total:.3f}s | chunks={lines}")

    raw_text = "".join(chunks).strip()
    if verbose:
        logging.debug(f"[{model}] Sortie brute (tronc.) = {raw_text[:300]!r}")

    queries = parse_numbered_list(raw_text)
    if len(queries) < 3:
        logging.warning(f"[{model}] Moins de 3 réécritures détectées ({len(queries)}). Complétion locale.")
        queries = safe_complete_to_three(queries, question)

    # Token/sec si méta dispo
    eval_count = last_meta.get("eval_count")
    eval_duration_ns = last_meta.get("eval_duration")
    tps = None
    if isinstance(eval_count, int) and isinstance(eval_duration_ns, int) and eval_duration_ns > 0:
        tps = eval_count / (eval_duration_ns / 1e9)

    # métriques simples
    d2 = distinct_2(queries)
    avg_len = avg_len_words(queries)

    tps_str = f"{tps:.2f}" if tps is not None else "N/A"
    logging.info(f"[{model}] Résumé run: time={total:.2f}s | tps={tps_str} | distinct-2={d2} | avg_len={avg_len}")
    return {
        "ok": True,
        "model": model,
        "question": question,
        "queries": queries[:3],
        "gen_time_sec": round(total, 3),
        "ttfb_sec": round(ttfb, 3),
        "stream_sec": round(stream_dur, 3),
        "tokens": eval_count,
        "tokens_per_sec": float(f"{tps:.2f}") if tps is not None else None,
        "distinct_2": d2,
        "avg_len_words": avg_len,
        "meta": last_meta,
        "raw": raw_text[:5000]
    }

# -------------------------------
# Orchestration
# -------------------------------

def aggregate_model_runs(per_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_runs = [r for r in per_runs if r.get("ok")]
    if not ok_runs:
        return {
            "runs": len(per_runs),
            "ok_runs": 0,
            "avg_time_sec": None,
            "min_time_sec": None,
            "max_time_sec": None,
            "avg_tps": None,
            "avg_distinct_2": None,
            "avg_len_words": None,
            "examples": []
        }
    times = [r["gen_time_sec"] for r in ok_runs if r.get("gen_time_sec") is not None]
    tps_vals = [r["tokens_per_sec"] for r in ok_runs if r.get("tokens_per_sec") is not None]
    d2_vals = [r["distinct_2"] for r in ok_runs if r.get("distinct_2") is not None]
    len_vals = [r["avg_len_words"] for r in ok_runs if r.get("avg_len_words") is not None]

    return {
        "runs": len(per_runs),
        "ok_runs": len(ok_runs),
        "avg_time_sec": round(statistics.mean(times), 3) if times else None,
        "min_time_sec": round(min(times), 3) if times else None,
        "max_time_sec": round(max(times), 3) if times else None,
        "avg_tps": round(statistics.mean(tps_vals), 2) if tps_vals else None,
        "avg_distinct_2": round(statistics.mean(d2_vals), 3) if d2_vals else None,
        "avg_len_words": round(statistics.mean(len_vals), 2) if len_vals else None,
        "examples": ok_runs[0]["queries"] if ok_runs and ok_runs[0].get("queries") else []
    }

def run_benchmark(question: str, models: List[str], runs: int, url: str, verbose: bool) -> Dict[str, Any]:
    bench_results = []
    for model in models:
        logging.info(f"==== Modèle: {model} | runs={runs} ====")
        per_runs = []
        for i in range(runs):
            logging.info(f"[{model}] RUN {i+1}/{runs} ...")
            try:
                res = call_ollama_once(
                    model=model, question=question, url=url,
                    temperature=0.35, num_predict=120, top_k=40, repeat_penalty=1.1,
                    timeout=180, verbose=verbose
                )
            except Exception as e:
                logging.exception(f"[{model}] Exception non gérée: {e}")
                res = {"ok": False, "model": model, "error": str(e), "queries": [], "gen_time_sec": None}
            per_runs.append(res)
            ok = res.get("ok")
            t = res.get("gen_time_sec")
            tps = res.get("tokens_per_sec")
            logging.info(f"[{model}] Résultat RUN {i+1}: ok={ok} time={t if t is not None else 'N/A'}s tps={tps if tps is not None else 'N/A'}")
        agg = aggregate_model_runs(per_runs)
        model_summary = {"model": model, "details": per_runs, **agg}
        bench_results.append(model_summary)
        logging.info(f"==> {model}: avg_time={agg['avg_time_sec']}s (min={agg['min_time_sec']}s, max={agg['max_time_sec']}s) "
                     f"| avg_tps={agg['avg_tps']} | avg_distinct2={agg['avg_distinct_2']} | ok_runs={agg['ok_runs']}/{agg['runs']}")
    return {"question": question, "results": bench_results}

def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_csv(path: str, bench: Dict[str, Any]) -> None:
    rows = []
    for r in bench["results"]:
        rows.append({
            "model": r["model"],
            "runs": r["runs"],
            "ok_runs": r["ok_runs"],
            "avg_time_sec": r["avg_time_sec"],
            "min_time_sec": r["min_time_sec"],
            "max_time_sec": r["max_time_sec"],
            "avg_tps": r["avg_tps"],
            "avg_distinct_2": r["avg_distinct_2"],
            "avg_len_words": r["avg_len_words"],
            "example_1": (r["examples"][0] if len(r["examples"]) > 0 else ""),
            "example_2": (r["examples"][1] if len(r["examples"]) > 1 else ""),
            "example_3": (r["examples"][2] if len(r["examples"]) > 2 else ""),
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model","runs","ok_runs","avg_time_sec","min_time_sec","max_time_sec",
                      "avg_tps","avg_distinct_2","avg_len_words","example_1","example_2","example_3"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, help="Question à réécrire")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                        help="Liste des modèles Ollama à tester (ils doivent être déjà pull)")
    parser.add_argument("--runs", type=int, default=1, help="Nombre de runs par modèle (>=3 recommandé)")
    parser.add_argument("--json", type=str, default=None, help="Chemin fichier sortie JSON")
    parser.add_argument("--csv", type=str, default=None, help="Chemin fichier sortie CSV")
    parser.add_argument("--url", type=str, default=DEFAULT_OLLAMA_URL, help="Endpoint Ollama /api/generate")
    parser.add_argument("--verbose", action="store_true", help="Logs verbeux (chunks, payload tronqué)")
    parser.add_argument("--debug", action="store_true", help="Logs DEBUG (très verbeux)")
    args = parser.parse_args()

    # Niveau de log
    level = "INFO"
    if args.debug:
        level = "DEBUG"
    elif args.verbose:
        level = "INFO"
    setup_logger(level)

    logging.info("=== Bench rewriters (single-call, 3 variantes) ===")
    logging.info(f"Question  : {args.question}")
    logging.info(f"Modèles   : {', '.join(args.models)}")
    logging.info(f"Runs/mod. : {args.runs}")
    logging.info(f"Endpoint  : {args.url}")
    bench = run_benchmark(args.question, args.models, args.runs, args.url, verbose=args.verbose)

    # Résumé console
    logging.info("\n=== Résumé global ===")
    for r in bench["results"]:
        logging.info(f"{r['model']}: avg_time={r['avg_time_sec']}s (min={r['min_time_sec']}s, max={r['max_time_sec']}s) "
                     f"| avg_tps={r['avg_tps']} | avg_distinct2={r['avg_distinct_2']} | ok_runs={r['ok_runs']}/{r['runs']}")

    if args.json:
        write_json(args.json, bench)
        logging.info(f"JSON écrit: {args.json}")
    if args.csv:
        write_csv(args.csv, bench)
        logging.info(f"CSV écrit: {args.csv}")

if __name__ == "__main__":
    main()
