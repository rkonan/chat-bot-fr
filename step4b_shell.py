import os
import signal
import sys

from step4_faiss import load_index, search_index
from step3_embed import embedder
from llama_cpp import Llama

MODEL_PATH = "models/Nous-Hermes-2-Mistral-7B-DPO.Q4_K_M.gguf"
INDEX_PATH = "vectordb/index.faiss"
CHUNKS_PATH = "vectordb/chunks.pkl"

# Initialisation LLM
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4)

# Chargement index et chunks
index, chunks = load_index(INDEX_PATH, CHUNKS_PATH)

if any("nicolas" in chunk.lower() for chunk in chunks):
    print ("Nicolas est dans les chunks")
# Interruption propre
def exit_gracefully(sig, frame):
    print("\n👋 Au revoir !")
    sys.exit(0)

signal.signal(signal.SIGINT, exit_gracefully)

print("🧠 RAG CLI interactif (CTRL+C pour quitter)")
print("Pose ta question :")
while True:
    question = input("\n❓> ").strip()
    if not question:
        continue
    query_embedding = embedder.encode([question], convert_to_numpy=True)
    indices, _ = search_index(index, query_embedding, top_k=3)
    for i in indices:
        print(f"\n--- Chunk {i} ---\n{chunks[i]}")
    context = "\n\n".join([chunks[i] for i in indices])
    MAX_CONTEXT_CHARS = 3000
    truncated_context = context[:MAX_CONTEXT_CHARS]
    prompt = f"""### Instruction: En te basant uniquement sur le contexte ci-dessous, réponds à la question de manière précise et en français.
Contexte :
{truncated_context}
Question : {question}
### Réponse:"""
    # output = llm(prompt, max_tokens=128, stop=["### Instruction:"])
    # response = output["choices"][0]["text"].strip().split("###")[0]
    # print(f"\n💬 Réponse : {response}")
