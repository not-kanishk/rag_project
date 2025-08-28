import streamlit as st
import os
import sys

# Ensure the project root is in the path for imports
sys.path.append(os.path.abspath('.'))

# Import core RAG functions
from PyPDF2 import PdfReader
from chunking.chunker import chunk_text
from embedding.embedder import Embedder
from vector_store.store import save_chunks, load_chunks
from retriever.retriever import retrieve
from generator.generator import answer
from sentence_transformers import util
from pipeline import embed_and_store

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "chunks_with_metadata" not in st.session_state:
    st.session_state.chunks_with_metadata = None

# --- Streamlit UI ---
st.set_page_config(layout="wide")

st.title("📄 Local RAG Chatbot")

# Use columns for a better layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file and not st.session_state.embedder:
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            pdf_name = os.path.splitext(uploaded_file.name)[0]
            embeddings_path = f"vector_store/{pdf_name}_embeddings.npy"
            chunks_path = f"vector_store/{pdf_name}_chunks.json"

            if os.path.exists(embeddings_path) and os.path.exists(chunks_path):
                st.info("Loading existing document embeddings and chunks.")
                st.session_state.embedder = Embedder()
                st.session_state.embeddings = st.session_state.embedder.load_embeddings(embeddings_path)
                st.session_state.chunks_with_metadata = load_chunks(chunks_path)
            else:
                st.info("Embedding new document and saving to vector store...")

                temp_dir = "temp_pdf"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                def extract_text(pdf_path):
                    reader = PdfReader(pdf_path)
                    text_with_metadata = []
                    for page_num, page in enumerate(reader.pages, start=1):
                        text = page.extract_text()
                        text_with_metadata.append({"text": text, "page": page_num})
                    return text_with_metadata

                text_with_metadata = extract_text(temp_path)
                chunks_with_metadata = chunk_text(text_with_metadata)

                st.session_state.embedder = Embedder()
                chunks_text_only = [chunk['text'] for chunk in chunks_with_metadata]

                embeddings = embed_and_store(chunks_text_only, st.session_state.embedder, embeddings_path)

                st.session_state.embeddings = embeddings
                st.session_state.chunks_with_metadata = chunks_with_metadata
                save_chunks(chunks_with_metadata, chunks_path)

                os.remove(temp_path)

        st.success(f"Successfully processed '{uploaded_file.name}'")


with col2:
    # User input for the question
    # This input is now disabled until a document is processed
    prompt = st.chat_input("Ask a question about the document:", disabled=not st.session_state.embedder)

    # Process user input and generate a response
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate a response ONLY if the embedder is ready
        if st.session_state.embedder:
            with st.spinner("Generating answer..."):
                retrieved_items = retrieve(prompt, st.session_state.embedder, st.session_state.embeddings, st.session_state.chunks_with_metadata)
                response = answer(prompt, retrieved_items)

            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Please upload a PDF first."})

    # Display chat messages from session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
