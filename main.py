from llama_cpp import Llama

llm = Llama(
    model_path="models/Nous-Hermes-2-Mistral-7B-DPO.Q4_K_M.gguf",  # ajuste le chemin si nécessaire
    n_ctx=2048,
    n_threads=4
)

response = llm("### Instruction: Quelle est la capitale du Sénégal ?\n### Réponse:", max_tokens=128)
print(response["choices"][0]["text"])
