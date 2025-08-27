import os
import json

def save_chunks(chunks, path="vector_store/pdf_chunks.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4)

def load_chunks(path="vector_store/pdf_chunks.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)