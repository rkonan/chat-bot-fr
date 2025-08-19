
# Step 6 — Docling robuste + Regex cleaning + Rapport HTML

Nouveaux apports par rapport au Step 5 :
- **Exclusions par regex** pilotées par YAML :
  - `cleaning.line_drop_regex`: lignes ignorées pendant le nettoyage (ex: *mentions légales*, *pieds de page*).
  - `cleaning.chunk_drop_regex`: chunks entiers supprimés si une regex matche (ex: *pages de garde*).
- **Rapport HTML autonome** `report.html` : cartes de stats + tableau top fichiers.

## Dépendances
```
pip install docling llama-index faiss-cpu "transformers<5" sentence-transformers pyyaml
# OCR optionnel
pip install pytesseract pdf2image
# + système: tesseract-ocr, poppler-utils
```

## Config (exemple)
```yaml
input:
  docs_dir: data
  extensions: [".pdf", ".docx"]
  exclude_dirs: ["tmp"]
  exclude_globs: ["*.bak.pdf"]
  max_size_mb: 100

chunking:
  min_chars: 220
  max_chars: 1400

dedup:
  max_hamming: 3

ocr:
  enable: true
  dpi: 200

cleaning:
  line_drop_regex:
    - "^confidentiel\\b"
    - "^(page|p\\.)\\s*\\d+$"
    - "mentions?\\s+légales"
  chunk_drop_regex:
    - "^table\\s+des\\s+matières"
    - "^sommaire$"

embedding:
  model: intfloat/multilingual-e5-base

output:
  out_dir: chatbot-models/vectordb_docling
```

## Exécuter
```
python step6_docling_pipeline.py --config config_step6.yml
```
Le rapport HTML est écrit dans `.../out_dir/report.html`.
