import os
import pickle
import faiss
import numpy as np
from step3_embed import embedder

# === Chemins par défaut ===
INDEX_PATH = "vectordb/index.faiss"
CHUNKS_PATH = "vectordb/chunks.pkl"

# === Création de l'index FAISS depuis des chunks de texte ===
def create_faiss_index(chunks, index_path=INDEX_PATH, chunks_path=CHUNKS_PATH):
    print("🔍 Génération des embeddings...")
    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    print("🧠 Création de l'index FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print("💾 Sauvegarde de l'index et des chunks...")
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Index créé avec succès.")

# === Chargement de l'index FAISS + chunks ===
def load_index(index_path=INDEX_PATH, chunks_path=CHUNKS_PATH):
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError("Index ou chunks non trouvés. Veuillez d'abord exécuter la création.")

    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# === Recherche dans l'index ===
def search_index(index, query_embedding, top_k=3):
    D, I = index.search(query_embedding, top_k)
    return I[0], D[0]


if __name__ =="__main__":
    from step1_read_pdf import read_pdf
    from step2_chunk import chunk_text
    from step3_embed import embed_chunks
    #Lecture du document
    text=read_pdf("data/DST_Rapport_final_Reco_plant.pdf")
    chunks =chunk_text(text,chunk_size=300,overlap=50)

    #Embedding 
    embeddings=embed_chunks(chunks)
    create_faiss_index(chunks)
