#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_rewriters_logged_plus.py

Améliorations 
- Extraction automatique d'un lot de questions à partir d'un document (PDF/texte)
- Exécution du benchmark sur TOUTES ces questions
- Exports JSON/CSV consolidés (une ligne par modèle et par question)
- Compatibilité avec le script existant (mêmes métriques + nouveaux champs)

Dépendances facultatives pour l'extraction PDF :
- PyMuPDF (fitz) OU pdfminer.six
(Si aucune n'est trouvée, affichage d'un message clair.)

Exemples :
  python benchmark_rewriters_logged_plus.py \
      --questions-from-doc DST_Rapport_final_Reco_plant.pdf \
      --models phi3:mini qwen2:1.5b-instruct gemma2:2b \
      --runs 2 --json bench_multi.json --csv bench_multi.csv --verbose

  python benchmark_rewriters_logged_plus.py "Quel modèle a le mieux géré Cassava ?" \
      --runs 3 --json bench_one.json --csv bench_one.csv

"""

import argparse
import csv
import json
import logging
import os
import re
import statistics
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

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
    items = [re.sub(r'\s+', ' ', s).strip().strip('\"\'' ) for s in items if s.strip()]
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
# PDF/text extraction & question mining
# -------------------------------

def _extract_text_with_pymupdf(path: str) -> Optional[str]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None
    try:
        with fitz.open(path) as doc:
            texts = []
            for page in doc:
                texts.append(page.get_text("text"))
        return "\n".join(texts)
    except Exception as e:
        logging.warning(f"PyMuPDF a échoué: {e}")
        return None


def _extract_text_with_pdfminer(path: str) -> Optional[str]:
    try:
        from pdfminer.high_level import extract_text
    except Exception:
        return None
    try:
        return extract_text(path)
    except Exception as e:
        logging.warning(f"pdfminer.six a échoué: {e}")
        return None


def extract_text_from_doc(path: str) -> str:
    """Retourne le texte d'un PDF/texte. Lève une exception si impossible."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    _, ext = os.path.splitext(path.lower())
    if ext in {".txt", ".md"}:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == ".pdf":
        text = _extract_text_with_pymupdf(path)
        if not text:
            text = _extract_text_with_pdfminer(path)
        if not text:
            raise RuntimeError(
                "Impossible d'extraire le texte du PDF. Installe PyMuPDF (fitz) ou pdfminer.six."
            )
        return text
    # Fallback: lire comme texte
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def harvest_questions_from_text(text: str, max_q: int = 20) -> List[str]:
    """Détecte des questions existantes et en génère d'autres à partir de titres/sections.
    Heuristiques simples, adaptées au rapport fourni (Cassava, modèles, métriques...).
    """
    lines = [normalize_space(l) for l in text.splitlines()]

    # 1) Questions déjà présentes (phrases finissant par '?')
    existing = re.findall(r"([^\n\r\?]{8,}?\?)", text)
    existing = [normalize_space(q) for q in existing]

    # 2) Titres/sections -> gabarits de questions
    headings = [l for l in lines if l and (l.isupper() or re.match(r"^(Partie|Objectifs|Conclusion|Résultats|Stratégies|Étude|Modèles|Métriques|Performances|Bilan|Suite du projet)\b", l, re.I))]
    generated: List[str] = []

    def add(q: str):
        qn = normalize_space(q)
        if len(qn) >= 12 and qn.endswith("?"):
            generated.append(qn)

    # Gabarits fréquents pour ce rapport
    add("Quels modèles ont été testés dans le projet ?")
    add("Quelles métriques principales et secondaires ont été utilisées ?")
    add("Quels sont les objectifs du projet et le contexte d'usage ?")
    add("Quelle est la performance globale atteinte par EfficientNetV2M ?")
    add("Quelles difficultés spécifiques ont été rencontrées sur la classe Cassava ?")
    add("Quelles stratégies de traitement des images Cassava ont été évaluées ?")
    add("Quel a été l'impact du soft voting par rapport aux modèles individuels ?")
    add("Quels jeux de données ont été retenus ou exclus et pourquoi ?")
    add("Quels critères de qualité d'image ont été utilisés (flou, contraste, luminosité) ?")
    add("Quel est le bilan global et les pistes d'amélioration proposées ?")
    add("Quel modèle offre le meilleur compromis précision / vitesse d'inférence ?")
    add("Pourquoi le stacking n'a-t-il pas surpassé le soft voting ?")
    add("Comment les cartes de saillance ont-elles aidé l'interprétation des modèles ?")
    add("Quelles limites du dataset ont impacté la généralisation ?")
    add("Quelles sont les conclusions sur l'inclusion de la classe Cassava dans le modèle final ?")

    # 3) Détections orientées mots-clés
    kw_templates = [
        (r"Cassav[ae]", [
            "Quelle a été la performance sur les sous-classes Cassava ?",
            "Quelles méthodes ont amélioré (ou dégradé) le F1-score sur Cassava ?",
        ]),
        (r"EfficientNetV2M|ResNet50V2|Swin|ViT", [
            "Quels écarts de performance entre EfficientNetV2M, ResNet50V2 et SwinTransformer ?",
        ]),
        (r"Grad[Cc]am|Saillanc", [
            "Que révèlent les cartes de saillance/Grad-CAM sur les zones d'attention ?",
        ]),
        (r"Accuracy|F1|métrique", [
            "Pourquoi avoir privilégié l'accuracy et le F1-score dans l'évaluation ?",
        ]),
    ]
    low_text = text
    for pat, qs in kw_templates:
        if re.search(pat, low_text, re.I):
            for q in qs:
                add(q)

    # Consolidation & dédup
    all_q = existing + generated
    seen = set()
    uniq = []
    for q in all_q:
        qn = normalize_space(q)
        if len(qn) >= 10 and qn.endswith("?") and qn not in seen:
            seen.add(qn)
            uniq.append(qn)
        if len(uniq) >= max_q:
            break
    return uniq

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
        return {"ok": False, "model": model, "error": str(e), "queries": [], "gen_time_sec": None, "question": question}

    logging.info(f"[{model}] HTTP status: {resp.status_code}")
    if resp.status_code != 200:
        try:
            err_body = resp.text[:500]
        except Exception:
            err_body = "<no-body>"
        logging.error(f"[{model}] HTTP {resp.status_code} body: {err_body}")
        return {"ok": False, "model": model, "error": f"HTTP {resp.status_code}: {err_body}", "queries": [], "gen_time_sec": None, "question": question}

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
        "raw": raw_text[:5000],
    }

# -------------------------------
# Orchestration (multi-questions)
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
                res = {"ok": False, "model": model, "error": str(e), "queries": [], "gen_time_sec": None, "question": question}
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

# -------------------------------
# Writers (multi-questions)
# -------------------------------

def write_json_multi(path: str, 
                     source_doc: Optional[str], 
                     questions: List[str], 
                     benchmarks: List[Dict[str, Any]]) -> None:
    payload = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_doc": source_doc,
        "questions": questions,
        "benchmarks": benchmarks,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv_multi(path: str, benchmarks: List[Dict[str, Any]]) -> None:
    rows = []
    for b in benchmarks:
        q = b.get("question", "")
        for r in b.get("results", []):
            rows.append({
                "question": q,
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
        fieldnames = [
            "question","model","runs","ok_runs","avg_time_sec","min_time_sec","max_time_sec",
            "avg_tps","avg_distinct_2","avg_len_words","example_1","example_2","example_3"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# -------------------------------
# CLI
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, nargs="?", default=None, help="Question à réécrire (si pas de doc)")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                        help="Liste des modèles Ollama à tester (ils doivent être déjà pull)")
    parser.add_argument("--runs", type=int, default=1, help="Nombre de runs par modèle (>=3 recommandé)")
    parser.add_argument("--json", type=str, default=None, help="Chemin fichier sortie JSON (consolidé)")
    parser.add_argument("--csv", type=str, default=None, help="Chemin fichier sortie CSV (consolidé)")
    parser.add_argument("--url", type=str, default=DEFAULT_OLLAMA_URL, help="Endpoint Ollama /api/generate")
    parser.add_argument("--verbose", action="store_true", help="Logs verbeux (chunks, payload tronqué)")
    parser.add_argument("--debug", action="store_true", help="Logs DEBUG (très verbeux)")

    # Nouvelles options
    parser.add_argument("--questions-from-doc", type=str, default=None,
                        help="Chemin vers un PDF/texte depuis lequel extraire/générer les questions")
    parser.add_argument("--max-questions", type=int, default=20, help="Nombre max de questions extraites")
    parser.add_argument("--questions-output", type=str, default=None,
                        help="Écrit la liste de questions extraites dans ce fichier (txt)")
    parser.add_argument("--questions-file", type=str, default=None,
                        help="Lit une liste de questions (une par ligne) depuis un fichier texte")

    args = parser.parse_args()

    # Niveau de log
    level = "INFO"
    if args.debug:
        level = "DEBUG"
    elif args.verbose:
        level = "INFO"
    setup_logger(level)

    # Prépare la/les questions
    questions: List[str] = []
    source_doc = None

    if args.questions_from_doc:
        source_doc = args.questions_from_doc
        logging.info(f"Extraction du texte depuis: {source_doc}")
        text = extract_text_from_doc(source_doc)
        questions = harvest_questions_from_text(text, max_q=args.max_questions)
        if not questions:
            logging.error("Aucune question n'a pu être extraite/générée depuis le document.")
            sys.exit(2)
        logging.info(f"{len(questions)} question(s) préparée(s) depuis le document.")
        if args.questions_output:
            with open(args.questions_output, "w", encoding="utf-8") as f:
                f.write("\n".join(questions))
            logging.info(f"Questions écrites dans: {args.questions_output}")

    elif args.questions_file:
        source_doc = args.questions_file
        with open(args.questions_file, "r", encoding="utf-8") as f:
            questions = [normalize_space(l) for l in f if normalize_space(l)]
        logging.info(f"{len(questions)} question(s) chargée(s) depuis {args.questions_file}")

    elif args.question:
        questions = [args.question]

    else:
        logging.error("Merci de fournir soit une question, soit --questions-from-doc, soit --questions-file.")
        sys.exit(2)

    logging.info("=== Bench rewriters (single-call, 3 variantes) ===")
    logging.info(f"Modèles   : {', '.join(args.models)}")
    logging.info(f"Runs/mod. : {args.runs}")
    logging.info(f"Endpoint  : {args.url}")

    all_benchmarks: List[Dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        logging.info("\n" + "="*80)
        logging.info(f"[Question {i}/{len(questions)}] {q}")
        bench = run_benchmark(q, args.models, args.runs, args.url, verbose=args.verbose)
        all_benchmarks.append(bench)

        # Résumé console par question
        logging.info("\n--- Résumé par modèle ---")
        for r in bench["results"]:
            logging.info(f"{r['model']}: avg_time={r['avg_time_sec']}s (min={r['min_time_sec']}s, max={r['max_time_sec']}s) "
                         f"| avg_tps={r['avg_tps']} | avg_distinct2={r['avg_distinct_2']} | ok_runs={r['ok_runs']}/{r['runs']}")

    # Exports consolidés
    if args.json:
        write_json_multi(args.json, source_doc, questions, all_benchmarks)
        logging.info(f"JSON écrit: {args.json}")
    if args.csv:
        write_csv_multi(args.csv, all_benchmarks)
        logging.info(f"CSV écrit: {args.csv}")


if __name__ == "__main__":
    main()
