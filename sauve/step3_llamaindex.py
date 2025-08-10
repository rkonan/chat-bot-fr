import os
import pickle
import faiss
import numpy as np
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Paramètres
DOCS_DIR = "data"
VECTOR_DIR = "vectordb"
INDEX_FILE = os.path.join(VECTOR_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(VECTOR_DIR, "chunks.pkl")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(VECTOR_DIR, exist_ok=True)

# Étape 1 — Lecture
print("📥 Chargement des documents...")
documents = SimpleDirectoryReader(input_dir=DOCS_DIR).load_data()

# Étape 2 — Chunking avec overlap (512 tokens, 64 d'overlap)
print("✂️ Découpage structuré avec overlap...")

parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=64,
    break_on_newlines=True  # 👈 Important ici
)
#parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
nodes = parser.get_nodes_from_documents(documents)

# Étape 3 — Embedding + FAISS
print("🔢 Génération des embeddings et indexation FAISS...")
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

# Créer un index brut FAISS
#dimension = embed_model.get_query_embedding("test").shape[0]
embedding_dim = np.array(embed_model.get_query_embedding("test")).shape[0]
faiss_index = faiss.IndexFlatL2(embedding_dim)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# Création de l’index LlamaIndex avec FAISS
index = VectorStoreIndex(nodes, embed_model=embed_model, vector_store=vector_store)

# Étape 4 — Sauvegarde
print("💾 Sauvegarde de l’index et des chunks...")
#vector_store.save(INDEX_FILE)
faiss.write_index(faiss_index, INDEX_FILE)
chunks = [node.get_content() for node in nodes]
with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ {len(chunks)} chunks sauvegardés dans {CHUNKS_FILE}")
