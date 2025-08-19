
import os
import re
import sys
import json
import time
import csv
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterable

# Optional deps
try:
    import yaml
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

# ---- Logger ----
logger = logging.getLogger("DoclingPipeline")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

def set_log_level(level_str: str):
    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)
    logger.info(f"Log level set to {logging.getLevelName(logger.level)}")

# ---- OCR (optional) ----
def _has_ocr():
    try:
        import pytesseract  # noqa: F401
        from pdf2image import convert_from_path  # noqa: F401
        return True
    except Exception:
        return False

def ocr_pdf_to_texts(path: Path, dpi: int = 200) -> List[str]:
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except Exception:
        logger.debug("OCR deps not available; skipping OCR.")
        return []
    try:
        images = convert_from_path(str(path), dpi=dpi)
    except Exception as e:
        logger.warning(f"OCR convert_from_path failed for {path.name}: {e}")
        return []
    texts = []
    for i, img in enumerate(images, 1):
        try:
            txt = pytesseract.image_to_string(img) or ""
        except Exception as e:
            logger.warning(f"OCR failed on {path.name} page {i}: {e}")
            txt = ""
        texts.append(txt)
    return texts

# ---- Cleaning/Filters ----
DOT_LEADER = re.compile(r"\.{5,}")
PAGE_FOOTER = re.compile(r"^\s*(page|p\.)\s*\d+\s*$", re.I)
JUST_NUM = re.compile(r"^\s*[\dIVXLC]+(\s*[-–]\s*[\dIVXLC]+)?\s*$", re.I)
MULTI_DOT_TOC = re.compile(r".*\d.*\.{3,}.*\d.*")
WS = re.compile(r"\s+")

def dehyphenate(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

def clean_text(raw: str, line_drop_regexes: List[re.Pattern]) -> str:
    t = (raw or "").replace("\r", "")
    t = dehyphenate(t)
    lines = []
    for line in t.split("\n"):
        s = line.strip()
        if not s:
            continue
        if any(r.search(s) for r in line_drop_regexes):
            logger.debug(f"Dropping line by regex: {s[:80]}")
            continue
        if DOT_LEADER.search(s) or MULTI_DOT_TOC.match(s) or PAGE_FOOTER.match(s) or JUST_NUM.match(s):
            logger.debug(f"Dropping boilerplate/toc/footer: {s[:80]}")
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

# ---- Utils ----
def sha256sum(path: Path, chunk_size: int = 1<<20) -> str:
    import hashlib
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
        parts = set([x.lower() for x in p.parts])
        if any(ed.lower() in parts for ed in exclude_dirs):
            logger.debug(f"Excluding by dir rule: {p}")
            continue
        name = p.name.lower()
        if any(re.fullmatch(gl.replace("*", ".*").lower(), name) for gl in exclude_globs):
            logger.debug(f"Excluding by glob rule: {p}")
            continue
        if max_size_mb is not None and p.stat().st_size > max_size_mb * (1<<20):
            logger.info(f"Skipping {p.name} (size>{max_size_mb}MB)")
            continue
        if p.name.startswith("~$"):
            continue
        outs.append(p)
    return sorted(outs)

# ---- SimHash ----
def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

def simhash(text: str, hashbits: int = 64) -> int:
    import hashlib
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
    out_t, out_m = [], []
    seen: list[int] = []
    dropped = 0
    for i, (t, m) in enumerate(zip(texts, metas)):
        fp = simhash(t)
        if any(hamming(fp, x) <= max_hamming for x in seen):
            dropped += 1
            logger.debug(f"Dropping near-dup chunk #{i} (simhash within {max_hamming})")
            continue
        seen.append(fp)
        out_t.append(t)
        out_m.append(m)
    logger.info(f"SimHash dedup dropped {dropped} chunks")
    return out_t, out_m

# ---- HTML Report ----
def _html_escape(s: str) -> str:
    import html
    return html.escape(str(s), quote=True)

def generate_html_report(stats: dict, per_file_rows: List[dict], out_path: Path):
    css = """
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
    h1 { font-size: 1.6rem; margin-bottom: 0.4rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; }
    table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
    th, td { border-bottom: 1px solid #eee; text-align: left; padding: 6px 8px; }
    th { background: #fafafa; position: sticky; top: 0; }
    .muted { color: #666; }
    code { background: #f6f8fa; padding: 2px 6px; border-radius: 6px; }
    .bar { height: 10px; background: #eaeaea; border-radius: 6px; }
    .fill { height: 10px; display:block; border-radius:6px; }
    """
    html_parts = []
    html_parts.append(f"<html><head><meta charset='utf-8'><style>{css}</style><title>Docling Ingestion Report</title></head><body>")
    html_parts.append("<h1>Docling Ingestion Report</h1>")
    html_parts.append("<div class='grid'>")
    keys = [
        ("Documents total", "docs_total"),
        ("Convertis", "docs_converted"),
        ("En cache (inchangés)", "docs_skipped_cache"),
        ("Chunks", "chunks_total"),
        ("Chars total", "total_chars"),
        ("Chars/chunk (moyen)", "avg_chars_per_chunk"),
        ("p50 chars", "p50_chars"),
        ("p90 chars", "p90_chars"),
        ("p99 chars", "p99_chars"),
        ("OCR activé", "ocr_enabled"),
        ("SimHash hamming ≤", "simhash_max_hamming"),
        ("Index", "faiss_index"),
        ("Docstore", "storage_dir"),
    ]
    for label, key in keys:
        val = stats.get(key, "")
        html_parts.append(f"<div class='card'><div class='muted'>{_html_escape(label)}</div><div><code>{_html_escape(val)}</code></div></div>")
    html_parts.append("</div>")
    rows = sorted(per_file_rows, key=lambda r: (-int(r.get('chunks_kept',0)), r.get('path','')))[:200]
    html_parts.append("<h2>Par fichier (Top 200)</h2>")
    html_parts.append("<div class='card'><div style='max-height:520px; overflow:auto;'>")
    html_parts.append("<table><thead><tr><th>Fichier</th><th>Taille</th><th>Chunks</th><th>Pages extraites</th><th>Pages OCR</th><th>Temps (s)</th></tr></thead><tbody>")
    max_chunks = max([int(r.get("chunks_kept",0)) for r in rows] or [1])
    for r in rows:
        chunks = int(r.get("chunks_kept",0))
        width = int(100 * chunks / max_chunks) if max_chunks else 0
        bar = f"<div class='bar'><span class='fill' style='width:{width}%; background:#9ecbff;'></span></div>"
        html_parts.append(
            "<tr>"
            f"<td>{_html_escape(r.get('path',''))}</td>"
            f"<td>{_html_escape(r.get('size_bytes',''))}</td>"
            f"<td>{chunks}{bar}</td>"
            f"<td>{_html_escape(r.get('pages_extracted',''))}</td>"
            f"<td>{_html_escape(r.get('pages_ocr',''))}</td>"
            f"<td>{_html_escape(r.get('seconds',''))}</td>"
            "</tr>"
        )
    html_parts.append("</tbody></table></div></div>")
    html_parts.append("</body></html>")
    out_path.write_text("".join(html_parts), encoding="utf-8")
    logger.info(f"HTML report written to: {out_path}")

# ---- Config loader ----
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

# ---- Main ----
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
    report_html = out_dir / "report.html"

    exts = tuple(cfg["input"].get("extensions", [".pdf", ".docx"]))
    exclude_dirs = cfg["input"].get("exclude_dirs", [])
    exclude_globs = cfg["input"].get("exclude_globs", [])
    max_size_mb = cfg["input"].get("max_size_mb")
    min_chars = int(cfg["chunking"].get("min_chars", 200))
    max_chars = int(cfg["chunking"].get("max_chars", 1500))
    simhash_hamming = int(cfg["dedup"].get("max_hamming", 3))
    enable_ocr = bool(cfg.get("ocr", {}).get("enable", False))
    ocr_dpi = int(cfg.get("ocr", {}).get("dpi", 200))
    embed_model_name = cfg["embedding"].get("model", "intfloat/multilingual-e5-base")

    #embed_model_name = cfg["embedding"].get("model", "intfloat/multilingual-e5-base")

    # Regex-based cleaning excludes
    line_drop_patterns = cfg.get("cleaning", {}).get("line_drop_regex", [])
    chunk_drop_patterns = cfg.get("cleaning", {}).get("chunk_drop_regex", [])
    line_drop_regexes = [re.compile(p, re.I) for p in line_drop_patterns]
    chunk_drop_regexes = [re.compile(p, re.I) for p in chunk_drop_patterns]

    # Cache
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    else:
        cache = {}

    logger.info(f"Scanning docs in {docs_dir} (exts={exts})")
    docs = list_docs(docs_dir, exts=exts, exclude_dirs=exclude_dirs, exclude_globs=exclude_globs, max_size_mb=max_size_mb)
    logger.info(f"Found {len(docs)} candidate files")
    if not docs:
        logger.warning(f"Aucun document trouvé sous {docs_dir}")
        return

    converter = DocumentConverter()
    chunker = HybridChunker()

    texts: List[str] = []
    metas: List[dict] = []
    lens_for_stats: List[int] = []
    pf_stats = []

    processed = 0
    skipped = 0
    logger.info("Debut traitements des docs")
    for p in docs:
        sha = sha256sum(p)
        logger.info("Fin calcul hash document")
        entry = cache.get(str(p))
        if entry and entry.get("sha256") == sha:
            skipped += 1
            logger.debug(f"Skip (cache hit): {p.name}")
            continue

        logger.info(f"Processing {p.name} (size={p.stat().st_size/1e6:.2f} MB)")
        t0 = time.time()
        pages_extracted = 0
        pages_ocr = 0
        chunks_kept = 0
        try:
            dldoc = converter.convert(str(p)).document
            logger.debug("Conversion of doc ok")
            pieces = chunker.chunk(dl_doc=dldoc)
            logger.info("Chunks ok")
            logger.debug(f"{p.name}: HybridChunker yielded {len(pieces)} fragments")
            used_ocr = False
            for idx, ch in enumerate(pieces):
                raw = getattr(ch, "text", "") or ""
                cleaned = clean_text(raw, line_drop_regexes)
                if len(cleaned) < 40 and enable_ocr and p.suffix.lower()==".pdf" and not used_ocr:
                    logger.info(f"{p.name}: low-text detected (frag#{idx}), trying OCR fallback…")
                    ocr_pages = ocr_pdf_to_texts(p, dpi=ocr_dpi) if _has_ocr() else []
                    if ocr_pages:
                        used_ocr = True
                        pages_ocr += len(ocr_pages)
                        for ot in ocr_pages:
                            oc = clean_text(ot, line_drop_regexes)
                            capped = cap_and_filter(oc, min_chars=min_chars, max_chars=max_chars)
                            if not capped:
                                continue
                            if any(r.search(capped) for r in chunk_drop_regexes):
                                logger.debug("Dropping chunk by chunk_drop_regex (OCR)")
                                continue
                            texts.append(capped)
                            metas.append({"doc": p.name, "relpath": str(p.relative_to(docs_dir)), "page": None, "title": getattr(ch, "title", None), "sha256": sha, "source": "ocr"})
                            lens_for_stats.append(len(capped))
                            chunks_kept += 1
                        logger.info(f"{p.name}: OCR kept {chunks_kept} chunks")
                        break
                capped = cap_and_filter(cleaned, min_chars=min_chars, max_chars=max_chars)
                if not capped:
                    continue
                if any(r.search(capped) for r in chunk_drop_regexes):
                    logger.debug("Dropping chunk by chunk_drop_regex")
                    continue
                texts.append(capped)
                metas.append({"doc": p.name, "relpath": str(p.relative_to(docs_dir)), "page": getattr(ch, "page_no", None), "title": getattr(ch, "title", None), "sha256": sha, "source": "docling"})
                lens_for_stats.append(len(capped))
                chunks_kept += 1
                pages_extracted += 1
        except Exception as e:
            logger.warning(f"Conversion failed for {p.name}: {e}")
            continue

        elapsed = time.time() - t0
        logger.info(f"{p.name}: kept {chunks_kept} chunks (pages={pages_extracted}, ocr_pages={pages_ocr}) in {elapsed:.2f}s")

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

    logger.info(f"Docs: {len(docs)} | converted: {processed} | cached: {skipped}")
    logger.info(f"Chunks before dedup: {len(texts)}")

    # SimHash near-dup filter
    texts, metas = dedup_simhash(texts, metas, max_hamming=simhash_hamming)
    logger.info(f"Chunks after dedup: {len(texts)}")

    if not texts:
        logger.warning("No chunk to index after filtering.")
        cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    # Embedding + FAISS
    logger.info(f"Embedding with model: {embed_model_name}")
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    dim = np.array(embed_model.get_query_embedding("test")).shape[0]
    faiss_index = faiss.IndexFlatIP(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    nodes = [TextNode(text=t) for t in texts]
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

    # Persist
    import pickle
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(index_file))
    storage_context.persist(persist_dir=str(storage_dir))
    with open(chunks_file, "wb") as f:
        pickle.dump(texts, f)
    with open(meta_file, "wb") as f:
        pickle.dump(metas, f)
    cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Persisted FAISS index -> {index_file}")
    logger.info(f"Persisted docstore -> {storage_dir}")

    # Stats
    total_chars = sum(len(t) for t in texts)
    avg_chars = int(total_chars / max(1, len(texts)))
    lens = lens_for_stats or [len(t) for t in texts]
    p50 = int(np.percentile(lens, 50))
    p90 = int(np.percentile(lens, 90))
    p99 = int(np.percentile(lens, 99))
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
        "chunk_lengths": lens[:10000],
    }
    stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Global stats:\n" + json.dumps(stats, ensure_ascii=False, indent=2))

    # Per-file CSV
    with open(stats_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path","sha256","size_bytes","chunks_kept","pages_extracted","pages_ocr","seconds"])
        w.writeheader()
        for row in pf_stats:
            w.writerow(row)
    logger.info(f"Wrote per-file stats CSV -> {stats_csv}")

    # HTML report
    generate_html_report(stats, pf_stats, report_html)

def main():
    ap = argparse.ArgumentParser(description="Step6 Docling (logged): regex excludes + HTML report + detailed logs")
    ap.add_argument("--config", required=True, help="Config YAML/JSON")
    ap.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    args = ap.parse_args()
    set_log_level(args.log_level)
    cfg = load_config(args.config)
    run(cfg)

if __name__ == "__main__":
    main()
