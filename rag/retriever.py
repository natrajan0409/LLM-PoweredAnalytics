import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from rag.embedder import get_embedding

load_dotenv()

class FaissRetriever:

    def __init__(self, index_path, chunks_path):
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

    def retrieve(self, query_text, top_k=5):
        query_embedding = np.array([get_embedding(query_text)], dtype="float32")
        distances, indexes = self.index.search(query_embedding, top_k)
        results = []
        for idx, score in zip(indexes[0], distances[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({"text": chunk["text"], "order_id": chunk["order_id"], "score": float(score)})
        return results
