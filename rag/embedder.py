import os
import pickle
import sqlite3
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter




load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH     = os.path.join(BASE_DIR, "database", "olist.db")
INDEX_PATH  = os.path.join(BASE_DIR, "database", "faiss.index")
CHUNKS_PATH = os.path.join(BASE_DIR, "database", "chunks.pkl")

embedding_model =  os.getenv("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
chunk_size = int(os.getenv("chunk_size", 1000))
chunk_overlap = int(os.getenv("chunk_overlap", 200))


def split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
    splitter =RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def get_embedding(text: str) -> list:
    """Embed a single text string — used by retriever for query lookup."""
    embedder = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"device": "cpu"})
    return embedder.embed_query(text)

def get_embeddings_batch(texts: list) -> list:
    """Embed a list of texts — used by build_faiss_index."""
    embedder = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"device": "cpu"})
    return embedder.embed_documents(texts)


def build_faiss_index():
    """read reivew from sqlite ,chunk,embed ,save faiss index +chunks.pkl"""
  
#    step A load retrival data from DB

    con =sqlite3.connect(DB_PATH)
    cursor= con.execute("SELECT order_id, review_comment_message FROM order_reviews "
        "WHERE review_comment_message IS NOT NULL "
        "AND TRIM(review_comment_message) != ''"
    )
    rows =cursor.fetchall()
    con.close()
    print(f"Loaded {len(rows)} reviews from database")

    #  Step B chunk  each review ,keep ordr_id metadata

    all_chunks=[]
    for order_id, text in rows:
        pieces  = split_text (text)
        for pice in pieces :
            all_chunks.append({"order_id": order_id, "text": pice})

    print(f"Total chunks after splitting: {len(all_chunks)}")

    # Step C  embed all check (load model once)
    texts = [c["text"] for c in all_chunks]
    vectors =get_embeddings_batch(texts)
    matrix = np.array(vectors, dtype="float32") 
    print(f"Embedding matrix shape: {matrix.shape}")

    # Step D Build faiss index
    
    dimension =matrix.shape[1]           # 384 for all-MiniLM-L6-v2
    index =faiss.IndexFlatL2(dimension)   # L2 distance (cosine also works)
    index.add(matrix)
    print(f"FAISS index size: {index.ntotal} vectors")

    # step e: save index and chunks to disk  
    faiss.write_index(index,INDEX_PATH)
    with open(CHUNKS_PATH,"wb") as f:
        pickle.dump(all_chunks,f)
    
    print(f"Saved index  → {INDEX_PATH}")
    print(f"Saved chunks → {CHUNKS_PATH}")

if __name__ == "__main__":
    build_faiss_index()