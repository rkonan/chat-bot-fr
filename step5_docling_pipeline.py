
import os
import re
import sys
import json
import time
import math
import csv
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterable

# Optional deps: yaml, pytesseract, pdf2image
try:
    import yaml  # pyyaml
except Exception:
    yaml = None

import numpy as np
import faiss

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# OCR optional
def _has_ocr():
    try:
        import pytesseract  # noqa: F401
        from pdf2image import convert_from_path  # noqa: F401
        return True
    except Exception:
        return False

def ocr_pdf_to_texts(path: Path, dpi: int = 200) -> List[str]:
    """Return list of page texts using pytesseract/pdf2image if available; else empty."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except Exception:
        return []
    try:
        images = convert_from_path(str(path), dpi=dpi)
    except Exception:
        return []
    texts = []
    for img in images:
        try:
            txt = pytesseract.image_to_string(img) or ""
        except Exception:
            txt = ""
        texts.append(txt)
    return texts

# ---------------- Cleaning/Filters ----------------
DOT_LEADER = re.compile(r"\.{5,}")                   # ..........
PAGE_FOOTER = re.compile(r"^\s*(page|p\.)\s*\d+\s*$", re.I)
JUST_NUM = re.compile(r"^\s*[\dIVXLC]+(\s*[-–]\s*[\dIVXLC]+)?\s*$", re.I)
MULTI_DOT_TOC = re.compile(r".*\d.*\.{3,}.*\d.*")    # "Titre ..... 19"
WS = re.compile(r"\s+")

def dehyphenate(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

def clean_text(raw: str) -> str:
    t = (raw or "").replace("\r", "")
    t = dehyphenate(t)
    lines = []
    for line in t.split("\n"):
        s = line.strip()
        if not s:
            continue
        if DOT_LEADER.search(s):
            continue
        if MULTI_DOT_TOC.match(s):
            continue
        if PAGE_FOOTER.match(s) or JUST_NUM.match(s):
            continue
        if len(s) < 3:
            continue
        lines.append(s)
    t = " ".join(lines)
    t = WS.sub(" ", t).strip()
    return t

def cap_and_filter(text: str, min_chars=200, max_chars=1500) -> Optional[str]:
    if not text:
        return None
    t = text[:max_chars]
    if len(t) < min_chars:
        return None
    return t

# ---------------- Utils ----------------
def sha256sum(path: Path, chunk_size: int = 1<<20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def list_docs(
    docs_dir: Path,
    exts=(".pdf", ".docx"),
    exclude_dirs: Iterable[str] = (),
    exclude_globs: Iterable[str] = (),
    max_size_mb: Optional[float] = None,
) -> List[Path]:
    outs = []
    for p in docs_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        # exclude dirs
        parts = set([x.lower() for x in p.parts])
        if any(ed.lower() in parts for ed in exclude_dirs):
            continue
        # exclude globs
        name = p.name.lower()
        if any(re.fullmatch(gl.replace("*", ".*").lower(), name) for gl in exclude_globs):
            continue
        if max_size_mb is not None and p.stat().st_size > max_size_mb * (1<<20):
            continue
        if p.name.startswith("~$"):
            continue
        outs.append(p)
    return sorted(outs)

# ---------------- SimHash (near-dup) ----------------
def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

def simhash(text: str, hashbits: int = 64) -> int:
    # Simple SimHash over tokens
    from collections import Counter
    v = [0] * hashbits
    for token, cnt in Counter(_tokenize(text)).items():
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        hv = int(h, 16)
        for i in range(hashbits):
            bit = 1 if (hv >> i) & 1 else -1
            v[i] += bit * cnt
    fingerprint = 0
    for i, val in enumerate(v):
        if val > 0:
            fingerprint |= (1 << i)
    return fingerprint

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def dedup_simhash(texts: List[str], metas: List[dict], max_hamming: int = 3) -> tuple[list[str], list[dict]]:
    # Greedy dedup: keep first occurrence, drop near-dups within Hamming threshold
    out_t, out_m = [], []
    seen: list[int] = []
    for t, m in zip(texts, metas):
        fp = simhash(t)
        if any(hamming(fp, x) <= max_hamming for x in seen):
            continue
        seen.append(fp)
        out_t.append(t)
        out_m.append(m)
    return out_t, out_m

# ---------------- Main ----------------
def run(cfg: dict):
    docs_dir = Path(cfg["input"]["docs_dir"])
    out_dir = Path(cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = out_dir / "storage"
    index_file = out_dir / "index.faiss"
    chunks_file = out_dir / "chunks.pkl"
    meta_file = out_dir / "meta.pkl"
    cache_file = out_dir / "ingest_cache.json"
    stats_json = out_dir / "stats.json"
    stats_csv = out_dir / "per_file_stats.csv"

    # config shortcuts
    exts = tuple(cfg["input"].get("extensions", [".pdf", ".docx"]))
    exclude_dirs = cfg["input"].get("exclude_dirs", [])
    exclude_globs = cfg["input"].get("exclude_globs", [])
    max_size_mb = cfg["input"].get("max_size_mb")
    min_chars = int(cfg["chunking"].get("min_chars", 200))
    max_chars = int(cfg["chunking"].get("max_chars", 1500))
    simhash_hamming = int(cfg["dedup"].get("max_hamming", 3))
    enable_ocr = bool(cfg.get("ocr", {}).get("enable", False))
    ocr_dpi = int(cfg.get("ocr", {}).get("dpi", 200))
    embed_model = cfg["embedding"].get("model", "intfloat/multilingual-e5-base")

    # cache
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    else:
        cache = {}

    docs = list_docs(docs_dir, exts=exts, exclude_dirs=exclude_dirs, exclude_globs=exclude_globs, max_size_mb=max_size_mb)
    if not docs:
        print(f"[!] Aucun document trouvé sous {docs_dir}")
        return

    converter = DocumentConverter()
    chunker = HybridChunker()

    texts: List[str] = []
    metas: List[dict] = []

    # per-file stats
    pf_stats = []

    processed = 0
    skipped = 0
    for p in docs:
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        entry = cache.get(str(p))
        if entry and entry.get("sha256") == sha:
            skipped += 1
            continue

        t0 = time.time()
        pages_extracted = 0
        pages_ocr = 0
        chunks_kept = 0
        try:
            dldoc = converter.convert(str(p)).document
            pieces = chunker.chunk(dl_doc=dldoc)
            # Decide whether to OCR after observing page texts are weak
            ocr_used_for_file = False
            for ch in pieces:
                raw = getattr(ch, "text", "") or ""
                cleaned = clean_text(raw)
                if len(cleaned) < 40 and enable_ocr and p.suffix.lower()==".pdf" and not ocr_used_for_file:
                    # low-text → try OCR for the entire file once
                    ocr_pages = ocr_pdf_to_texts(p, dpi=ocr_dpi) if _has_ocr() else []
                    if ocr_pages:
                        ocr_used_for_file = True
                        pages_ocr += len(ocr_pages)
                        for ot in ocr_pages:
                            oc = clean_text(ot)
                            capped = cap_and_filter(oc, min_chars=min_chars, max_chars=max_chars)
                            if not capped:
                                continue
                            texts.append(capped)
                            metas.append({"doc": p.name, "relpath": str(p.relative_to(docs_dir)), "page": None, "title": getattr(ch, "title", None), "sha256": sha, "source": "ocr"})
                            chunks_kept += 1
                        break  # skip docling chunks if OCR succeeded
                capped = cap_and_filter(cleaned, min_chars=min_chars, max_chars=max_chars)
                if not capped:
                    continue
                texts.append(capped)
                metas.append({"doc": p.name, "relpath": str(p.relative_to(docs_dir)), "page": getattr(ch, "page_no", None), "title": getattr(ch, "title", None), "sha256": sha, "source": "docling"})
                chunks_kept += 1
                pages_extracted += 1
        except Exception as e:
            print(f"[WARN] Échec conversion {p.name}: {e}")
            continue

        elapsed = time.time() - t0
        pf_stats.append({
            "path": str(p),
            "sha256": sha,
            "size_bytes": p.stat().st_size,
            "chunks_kept": chunks_kept,
            "pages_extracted": pages_extracted,
            "pages_ocr": pages_ocr,
            "seconds": round(elapsed, 3),
        })

        cache[str(p)] = {"sha256": sha, "mtime": int(p.stat().st_mtime)}
        processed += 1

    print(f"Docs: {len(docs)} | convertis: {processed} | inchangés (cache): {skipped}")
    print(f"Chunks (avant dédup): {len(texts)}")

    # SimHash near-dup filter
    texts, metas = dedup_simhash(texts, metas, max_hamming=simhash_hamming)
    print(f"Chunks (après SimHash hamming≤{simhash_hamming}): {len(texts)}")

    if not texts:
        print("[!] Aucun chunk à indexer après filtrage.")
        cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    # Embedding + FAISS cosine
    embed_model = HuggingFaceEmbedding(model_name=embed_model)
    dim = np.array(embed_model.get_query_embedding("test")).shape[0]
    faiss_index = faiss.IndexFlatIP(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    nodes = [TextNode(text=t) for t in texts]
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

    # Persist artifacts
    import pickle
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(index_file))
    storage_context.persist(persist_dir=str(storage_dir))
    with open(chunks_file, "wb") as f:
        pickle.dump(texts, f)
    with open(meta_file, "wb") as f:
        pickle.dump(metas, f)

    # Save cache
    cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    # Stats globales
    total_chars = sum(len(t) for t in texts)
    avg_chars = int(total_chars / max(1, len(texts)))
    lens = [len(t) for t in texts]
    try:
        import numpy as _np
        p50 = int(_np.percentile(lens, 50))
        p90 = int(_np.percentile(lens, 90))
        p99 = int(_np.percentile(lens, 99))
    except Exception:
        lens_sorted = sorted(lens)
        def percentile(seq, q):
            k = int((q/100) * (len(seq)-1))
            return seq[min(max(k,0), len(seq)-1)]
        p50 = percentile(lens_sorted, 50)
        p90 = percentile(lens_sorted, 90)
        p99 = percentile(lens_sorted, 99)

    stats = {
        "docs_total": len(docs),
        "docs_converted": processed,
        "docs_skipped_cache": skipped,
        "chunks_total": len(texts),
        "total_chars": total_chars,
        "avg_chars_per_chunk": avg_chars,
        "p50_chars": p50,
        "p90_chars": p90,
        "p99_chars": p99,
        "faiss_index": str(index_file),
        "storage_dir": str(storage_dir),
        "chunks_file": str(chunks_file),
        "meta_file": str(meta_file),
        "simhash_max_hamming": simhash_hamming,
        "ocr_enabled": enable_ocr and _has_ocr(),
    }
    stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # Per-file CSV
    with open(stats_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path","sha256","size_bytes","chunks_kept","pages_extracted","pages_ocr","seconds"])
        w.writeheader()
        for row in pf_stats:
            w.writerow(row)

    print("---- Stats ----")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

def load_config(path: str) -> dict:
    p = Path(path)
    if p.suffix.lower() in [".yml", ".yaml"]:
        if yaml is None:
            raise RuntimeError("pyyaml non installé: pip install pyyaml")
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Step5 Docling pipeline robuste (configurable, OCR, SimHash, stats)")
    ap.add_argument("--config", required=True, help="Chemin du fichier de config (YAML ou JSON)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg)
