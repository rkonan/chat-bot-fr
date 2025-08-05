import os
import pickle
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
import textwrap

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex

# 📁 Paramètres
DOCS_DIR = "data"
VECTOR_DIR = "vectordb"
INDEX_FILE = os.path.join(VECTOR_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(VECTOR_DIR, "chunks.pkl")
EMBEDDING_MODEL = "dangvantuan/french-document-embedding"

os.makedirs(VECTOR_DIR, exist_ok=True)

# 📥 Chargement manuel des PDF
print("📥 Lecture des fichiers PDF...")
reader = PDFReader()
documents = []

for pdf_path in Path(DOCS_DIR).glob("*.pdf"):
    print(f" - 📄 {pdf_path.name}")
    docs = reader.load_data(pdf_path)  # ✅ CORRECTION : path au lieu de file
    documents.extend(docs)

print(f"✅ {len(documents)} documents PDF chargés.")

# ✂️ Chunking par taille de tokens (plus stable que par phrases)
print("✂️ Chunking avec SentenceSplitter (512 tokens, overlap 64)...")
parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
nodes = parser.get_nodes_from_documents(documents)

print(f"✅ {len(nodes)} chunks générés.")

# 🔍 Aperçu des 5 premiers chunks
print("\n🧩 Aperçu des 5 premiers chunks :\n")
for i, node in enumerate(nodes[:5]):
    preview = textwrap.shorten(node.get_content().replace("\n", " "), width=200)
    print(f"Chunk {i+1:>2}: {preview}")

# 🔢 Embedding + FAISS
print("\n🔢 Génération des embeddings et indexation FAISS...")
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL, trust_remote_code=True)
embedding_dim = np.array(embed_model.get_query_embedding("test")).shape[0]
faiss_index = faiss.IndexFlatL2(embedding_dim)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# 🧠 Construction de l’index vectoriel
index = VectorStoreIndex(nodes, embed_model=embed_model, vector_store=vector_store)

# 💾 Sauvegarde
print("💾 Sauvegarde de l’index et des chunks...")
faiss.write_index(faiss_index, INDEX_FILE)
chunks = [node.get_content() for node in nodes]
with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(chunks, f)

print(f"\n✅ {len(chunks)} chunks sauvegardés dans {CHUNKS_FILE}")
