
# Step 6 (Logged) — Docling + Regex + HTML + Detailed Logs

## Run
```bash
python step6_docling_pipeline_logged.py --config config_step6.yml --log-level DEBUG
```
Log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`.

## What you get
- Structured logs (timestamps, levels) at each stage
- SimHash dedup stats, per-file processing summaries
- OCR fallback details (if enabled)
- Persisted FAISS + LlamaIndex, CSV + JSON stats, HTML report
