import os
import pickle
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex

# 📁 Paramètres
DOCS_DIR = "data"
VECTOR_DIR = "vectordb_docling"
INDEX_FILE = os.path.join(VECTOR_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(VECTOR_DIR, "chunks.pkl")
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

os.makedirs(VECTOR_DIR, exist_ok=True)

# 📥 Conversion avec Docling
print("📥 Conversion des documents avec Docling...")
converter = DocumentConverter()
dl_docs = []

for pdf_path in Path(DOCS_DIR).glob("*.pdf"):
    print(f" - 📄 {pdf_path.name}")
    docling_doc = converter.convert(str(pdf_path)).document
    dl_docs.append(docling_doc)

# ✂️ Chunking sémantique via HybridChunker
print("✂️ Chunking intelligent avec HybridChunker (Docling)...")
chunker = HybridChunker()
text_nodes = []

for dl_doc in dl_docs:
    chunks = chunker.chunk(dl_doc=dl_doc)
    for chunk in chunks:
        text_nodes.append(TextNode(text=chunk.text))

print(f"✅ {len(text_nodes)} chunks générés.")

# 🔢 Embedding + FAISS index
print("🔢 Génération des embeddings et indexation FAISS...")
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
embedding_dim = np.array(embed_model.get_query_embedding("test")).shape[0]
faiss_index = faiss.IndexFlatL2(embedding_dim)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# 🧠 Construction de l’index vectoriel
index = VectorStoreIndex(text_nodes, embed_model=embed_model, vector_store=vector_store)

# 💾 Sauvegarde
print("💾 Sauvegarde de l’index et des chunks...")
faiss.write_index(faiss_index, INDEX_FILE)
chunks = [node.get_content() for node in text_nodes]

with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ {len(chunks)} chunks sauvegardés dans {CHUNKS_FILE}")
