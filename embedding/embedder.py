from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_chunks(self, chunks):
        return self.model.encode(chunks)
    
    def save_embeddings(self, embeddings, path="vector_store/pdf_embeddings.npy"):
        np.save(path, embeddings)
    
    def load_embeddings(self, path="vector_store/pdf_embeddings.npy"):
        return np.load(path)