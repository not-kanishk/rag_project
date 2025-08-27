def chunk_text(pages_with_metadata, chunk_size=500, overlap=50):
    chunks_with_metadata = []
    
    # Simple chunking by page
    for page in pages_with_metadata:
        words = page['text'].split()
        page_num = page['page']
        
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            chunks_with_metadata.append({"text": chunk_text, "page": page_num})
            start += chunk_size - overlap
            
    return chunks_with_metadata