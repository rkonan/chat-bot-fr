from pathlib import Path
from pdfminer.high_level import extract_text

PDF_DIR = "data"
OUT_DIR = "converted_txt"
Path(OUT_DIR).mkdir(exist_ok=True)

for pdf_path in Path(PDF_DIR).glob("*.pdf"):
    text = extract_text(str(pdf_path))
    output_file = Path(OUT_DIR) / (pdf_path.stem + ".txt")
    output_file.write_text(text, encoding="utf-8")
    print(f"✅ {pdf_path.name} → {output_file.name}")
