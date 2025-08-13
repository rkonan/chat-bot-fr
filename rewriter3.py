import sys, json, requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b-instruct-q4_K_M"

#SYSTEM = ("Réécris UNIQUEMENT cette question en français, plus claire et concise, "
#          "sans y répondre et sans ajouter d'information. "
#          "Donne une seule phrase, rien d'autre.")


SYSTEM = ("Réécris UNIQUEMENT cette question en français, plus claire et concise, "
          "sans y répondre et sans ajouter d'information. "
          "Donne une seule phrase, rien d'autre.")

def one_rewrite(question: str, seed: int) -> str:
    payload = {
        "model": MODEL,
        "prompt": f"{SYSTEM}\n\nQuestion: {question}",
        "options": {
            "temperature": 0.2,   # assez bas pour rester fidèle, assez haut pour varier
            "num_predict": 64,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "seed": seed
        }
    }
    r = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
    text = []
    for line in r.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        text.append(chunk.get("response", ""))
        if chunk.get("done"):
            break
    out = "".join(text).strip()
    # Nettoyage léger
    out = out.strip(' "\n').replace("  ", " ")
    return out

def three_rewrites(question: str):
    seeds = [11, 22, 33]
    out = []
    for s in seeds:
        rw = one_rewrite(question, s)
        if rw and rw not in out:
            out.append(rw)
    # Fallback si peu de diversité
    if len(out) < 3:
        out.append(one_rewrite(question, 44))
    if len(out) < 3:
        out.append(question)  # dernière sécurité
    return out[:3]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 rewriter3.py \"Votre question ici\"")
        sys.exit(1)
    q = sys.argv[1]
    rewrites = three_rewrites(q)
    print(json.dumps({"queries": rewrites}, ensure_ascii=False, indent=2))
