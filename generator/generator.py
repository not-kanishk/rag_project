import ollama

def answer(query, retrieved_items):
    # Format the prompt with the retrieved chunks and their page numbers
    context = ""
    for item in retrieved_items:
        context += f"Page {item['page']}: {item['text']}\n"

    # Add a clear instruction to the LLM to cite the source
    prompt = f"Using only the following context, answer the question: {query}\n\nContext:\n{context}\n\nCite the page number(s) from the context when you refer to information."

    # Use the local LLM to generate an answer
    try:
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        generated_answer = response['message']['content']
        
        # Now, return the full answer with evidence
        response_text = f"**Generated Answer:**\n{generated_answer}\n"
        
        # Add the supporting evidence to the response string
        response_text += "\n**Supporting Evidence:**\n"
        for item in retrieved_items:
            response_text += f"\n- Page {item['page']}: {item['text']}"
        
        return response_text
    
    except ollama.OllamaConnectionError:
        return "**Error:** Ollama server is not running. Please ensure Ollama is active."
    except Exception as e:
        return f"**An error occurred:** {e}"