import fitz
import sys
import os

def read_pdf(path):
    if not os.path.exists(path):
        print(f" Fichier non trouvé : {path}")
        sys.exit(1)  

    doc = fitz.open(path)
    all_text=""

    for i,page in enumerate(doc):
        text=page.get_text()
        print(f"Page {i+1} - {len(text)}  caractères") 
        print("-"*50)
        print(text[:500])  # Affiche les 500 premiers caractères
        print("\n")
        all_text += text + "\n"
    return all_text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ Utilisation : python step1_read_pdf.py chemin/vers/fichier.pdf")
        sys.exit(1)

    file_path = sys.argv[1]
    text = read_pdf(file_path)
    print(f"\n✅ Extraction terminée. {len(text)} caractères récupérés.")