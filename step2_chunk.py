import re

def chunk_text(text,chunk_size=300,overlap=50):
    words =text.split()
    chunks=[]
    i=0
    while i<len(words):
        chunk=words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i+=chunk_size-overlap
    return chunks

if __name__ =="__main__":
    from step1_read_pdf import read_pdf

    text=read_pdf("data/DST_Rapport_final_Reco_plant.pdf")

    print(f"\n Longueur totale du texte : {len(text)} caractères")
    chunks =chunk_text(text,chunk_size=300,overlap=50)
    print(f"Nombre de chunks  {len(chunks)}")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n Chunk {i+1} ({len(chunks)})")
        print(chunk[:500], "..." if len(chunk)>500 else "")
