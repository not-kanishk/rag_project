from sentence_transformers import util

def retrieve(query, embedder, embeddings, chunks_with_metadata, top_k=5):
    query_vec = embedder.model.encode(query)
    scores = util.cos_sim(query_vec, embeddings)[0].cpu().numpy()
    top_indices = scores.argsort()[-top_k:][::-1]
    
    # Return the chunk dictionaries, not just the text
    return [chunks_with_metadata[i] for i in top_indices]