import os
import re
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
VECTOR_DIR = "chatbot-models/vectordb_docling"  # même emplacement que ton app
INDEX_FILE = os.path.join(VECTOR_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(VECTOR_DIR, "chunks.pkl")
META_FILE = os.path.join(VECTOR_DIR, "meta.pkl")
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

os.makedirs(VECTOR_DIR, exist_ok=True)

# ---------- Helpers de nettoyage ----------
DOT_LEADER = re.compile(r"\.{5,}")                   # ..........
PAGE_FOOTER = re.compile(r"^\s*(page|p\.)\s*\d+\s*$", re.I)
JUST_NUM = re.compile(r"^\s*[\dIVXLC]+(\s*[-–]\s*[\dIVXLC]+)?\s*$", re.I)
MULTI_DOT_TOC = re.compile(r".*\d.*\.{3,}.*\d.*")    # "Titre ..... 19"
WS = re.compile(r"\s+")

def dehyphenate(text: str) -> str:
    # enlève "mot-\nsuite" -> "motsuite"
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

def clean_text(raw: str) -> str:
    t = raw.replace("\r", "")
    t = dehyphenate(t)
    lines = []
    for line in t.split("\n"):
        s = line.strip()
        if not s:
            continue
        if DOT_LEADER.search(s):        # lignes avec suite de points (sommaire)
            continue
        if MULTI_DOT_TOC.match(s):      # "Titre ..... 19"
            continue
        if PAGE_FOOTER.match(s) or JUST_NUM.match(s):  # "page 12" / "XII"
            continue
        if len(s) < 3:
            continue
        lines.append(s)
    t = " ".join(lines)
    t = WS.sub(" ", t).strip()
    return t

def cap_and_filter(text: str, min_chars=200, max_chars=1500) -> str | None:
    if not text:
        return None
    t = text[:max_chars]
    if len(t) < min_chars:
        return None
    return t

# ---------- Pipeline ----------
print("📥 Conversion des documents avec Docling...")
converter = DocumentConverter()
dl_docs = []
doc_paths = list(Path(DOCS_DIR).glob("*.pdf"))
for pdf_path in doc_paths:
    print(f" - 📄 {pdf_path.name}")
    docling_doc = converter.convert(str(pdf_path)).document
    dl_docs.append((pdf_path.name, docling_doc))

print("✂️ Chunking intelligent avec HybridChunker (Docling)...")
chunker = HybridChunker()

text_nodes: list[TextNode] = []
metas: list[dict] = []

for doc_name, dl_doc in dl_docs:
    chunks = chunker.chunk(dl_doc=dl_doc)
    # Docling expose souvent des champs utiles : chunk.text, chunk.page_no, chunk.title (selon version)
    for chunk in chunks:
        raw_text = getattr(chunk, "text", "")
        cleaned = clean_text(raw_text)
        capped = cap_and_filter(cleaned)
        if not capped:
            continue

        node = TextNode(text=capped)
        # métadonnées utiles (selon ce que Docling fournit)
        meta = {
            "doc": doc_name,
            "page": getattr(chunk, "page_no", None),
            "title": getattr(chunk, "title", None),
        }
        text_nodes.append(node)
        metas.append(meta)

print(f"✅ {len(text_nodes)} chunks (propres) avant dé‑duplication.")

# Dé‑duplication naïve par hash
print("🧹 Dé‑duplication simple…")
seen = set()
dedup_nodes: list[TextNode] = []
dedup_metas: list[dict] = []
for node, meta in zip(text_nodes, metas):
    h = hash(node.get_content())
    if h in seen:
        continue
    seen.add(h)
    dedup_nodes.append(node)
    dedup_metas.append(meta)

text_nodes = dedup_nodes
metas = dedup_metas
print(f"✅ {len(text_nodes)} chunks après dé‑duplication.")

# 🔢 Embedding + FAISS (cosine)
print("🔢 Génération des embeddings et indexation FAISS (cosine)…")
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
# LlamaIndex s’occupe d’embarquer les embeddings; mais on force ici le type d’index
embedding_dim = np.array(embed_model.get_query_embedding("test")).shape[0]

# IMPORTANT : cosine = normaliser + IndexFlatIP
faiss_index = faiss.IndexFlatIP(embedding_dim)
vector_store = FaissVectorStore(faiss_index=faiss_index)
index = VectorStoreIndex(text_nodes, embed_model=embed_model, vector_store=vector_store)

# 💾 Sauvegarde
print("💾 Sauvegarde de l’index et des chunks + metas…")
faiss.write_index(faiss_index, INDEX_FILE)

chunks = [node.get_content() for node in text_nodes]
with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(chunks, f)

with open(META_FILE, "wb") as f:
    pickle.dump(metas, f)

print(f"✅ {len(chunks)} chunks sauvegardés dans {CHUNKS_FILE}")
print(f"✅ Métadonnées sauvegardées dans {META_FILE}")
print(f"✅ Index FAISS: {INDEX_FILE}")
