from sentence_transformers import SentenceTransformer
import numpy as np 

EMBED_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
embedder=SentenceTransformer(EMBED_MODEL_NAME)

def embed_chunks(chunks):
    embeddings=embedder.encode(chunks,convert_to_numpy=True,show_progress_bar=True)
    return embeddings

if __name__=="__main__":
    from step1_read_pdf import read_pdf
    from step2_chunk import chunk_text
    #Lecture du document
    text=read_pdf("data/DST_Rapport_final_Reco_plant.pdf")
    chunks =chunk_text(text,chunk_size=300,overlap=50)

    #Embedding 
    embeddings=embed_chunks(chunks)

    print(f"\n✅ Embeddings générés : {embeddings.shape[0]} vectors de {embeddings.shape[1]} dimensions")
    print(f"Exemple (1er vecteur) :\n{embeddings[0][:5]}...")

