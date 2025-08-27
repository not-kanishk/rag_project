import os
import sys
from PyPDF2 import PdfReader
from chunking.chunker import chunk_text
from embedding.embedder import Embedder
from vector_store.store import save_chunks, load_chunks
from retriever.retriever import retrieve
from generator.generator import answer
from sentence_transformers import util

# ----------------------------
# Step 1: Load PDF and extract text
# ----------------------------
def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text_with_metadata = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        text_with_metadata.append({"text": text, "page": page_num})
    return text_with_metadata

# ----------------------------
# Step 2: Chunk text
# ----------------------------
def create_chunks(text_with_metadata, chunk_size=500, overlap=50):
    return chunk_text(text_with_metadata, chunk_size=chunk_size, overlap=overlap)

# ----------------------------
# Step 3: Embed chunks
# ----------------------------
def embed_and_store(chunks, embedder, embeddings_path):
    embeddings = embedder.embed_chunks(chunks)
    embedder.save_embeddings(embeddings, embeddings_path)
    return embeddings

# ----------------------------
# Step 4: Main
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_rag.py path_to_pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    # Create a unique name for the embeddings and chunks based on the PDF file name
    pdf_name = os.path.basename(pdf_path).split('.')[0]
    
    os.makedirs("vector_store", exist_ok=True)
    embeddings_path = f"vector_store/{pdf_name}_embeddings.npy"
    chunks_path = f"vector_store/{pdf_name}_chunks.json"

    # Step A: Extract & chunk PDF (if files don't exist)
    if not os.path.exists(embeddings_path) or not os.path.exists(chunks_path):
        print("Extracting text and metadata from PDF...")
        text_with_metadata = extract_text(pdf_path)
        chunks_with_metadata = create_chunks(text_with_metadata)
        print(f"Created {len(chunks_with_metadata)} chunks with metadata.")
        
        print("Embedding chunks and saving to vector_store...")
        embedder = Embedder()
        
        # Pass only the text to the embedder for embeddings
        chunks_text_only = [chunk['text'] for chunk in chunks_with_metadata]
        embeddings = embed_and_store(chunks_text_only, embedder, embeddings_path)

        # Save the chunks with metadata separately
        save_chunks(chunks_with_metadata, chunks_path)
    else:
        print("Loading existing embeddings and chunks...")
        embedder = Embedder()
        embeddings = embedder.load_embeddings(embeddings_path)
        chunks_with_metadata = load_chunks(chunks_path)

    # Step B: Interactive Q&A
    print("\nRAG Pipeline ready. Ask questions about your PDF (type 'exit' to quit).")
    print("Ensure Ollama is running in the background.")
    while True:
        q = input("\nEnter question: ")
        if q.lower() == "exit":
            break
        
        # Use the imported functions
        retrieved_items = retrieve(q, embedder, embeddings, chunks_with_metadata)
        answer(q, retrieved_items)