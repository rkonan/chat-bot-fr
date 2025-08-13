#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
import re
import requests
import time
from typing import List

OLLAMA_URL = "http://localhost:11434/api/generate"
#MODEL = "qwen2.5:3b-instruct-q4_K_M"

MODEL = "qwen2:1.5b-instruct"

PROMPT_TEMPLATE = """Donne exactement trois réécritures DIFFÉRENTES de la question suivante, en français clair et concis,
sans ajouter d’informations ni y répondre.
Réponds uniquement au format :
1) ...
2) ...
3) ...

Question : {question}
"""

def parse_numbered_list(text: str) -> List[str]:
    items = re.findall(r'^\s*[123]\)\s*(.+?)\s*$', text, flags=re.M)
    items = [re.sub(r'\s+', ' ', s).strip().strip('"\'' ) for s in items if s.strip()]
    seen, uniq = set(), []
    for s in items:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

def local_variants(seed_phrase: str, need: int) -> List[str]:
    q = seed_phrase.strip().rstrip(" ?")
    cands = [
        q,
        re.sub(r'\btrès rapide\b', 'rapide', q, flags=re.I),
        re.sub(r'\bOllama\b', 'la plateforme Ollama', q, flags=re.I),
        q.replace("Quel", "Lequel"),
        q.replace("Quel", "Quel est le modèle"),
        q.replace("pour la reformulation", "pour reformuler rapidement")
    ]
    out = []
    for s in cands:
        s = s.strip()
        if not s.endswith("?"):
            s += " ?"
        if s not in out:
            out.append(s)
        if len(out) >= need:
            break
    return out[:need]

def three_rewrites_one_call(question: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(question=question)

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "options": {
            "temperature": 0.4,
            "num_predict": 160,
            "top_k": 40,
            "repeat_penalty": 1.1
        }
    }

    print(f"[INFO] Lancement génération pour : \"{question}\"")
    call_start = time.time()
    r = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=180)
    chunks = []
    for line in r.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        chunks.append(data.get("response", ""))
        if data.get("done"):
            break
    call_end = time.time()
    print(f"[PERF] Temps appel Ollama : {call_end - call_start:.2f} sec")

    text = "".join(chunks).strip()
    rewrites = parse_numbered_list(text)

    if len(rewrites) < 3:
        need = 3 - len(rewrites)
        base = rewrites[-1] if rewrites else question
        rewrites.extend(local_variants(base, need))

    seen, final = set(), []
    for s in rewrites:
        if s not in seen and s:
            seen.add(s)
            final.append(s)
        if len(final) == 3:
            break

    return {"queries": final, "perf_seconds": call_end - call_start}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rewriter3_onecall.py \"Votre question ici\"")
        sys.exit(1)

    total_start = time.time()
    question = sys.argv[1]
    result = three_rewrites_one_call(question)
    total_end = time.time()

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[PERF] Temps total script : {total_end - total_start:.2f} sec")
