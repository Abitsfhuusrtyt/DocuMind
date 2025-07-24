from flask import Flask, request, jsonify, render_template
import json
import numpy as np
import faiss
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# --- Load environment variables ---
load_dotenv()

# --- Configure Gemini API ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel(model_name='gemini-1.5-flash')  # or use 'gemini-1.5-flash'

# --- Flask App ---
app = Flask(__name__)

# --- Load files and models on startup ---
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('corpus.index')

with open('chunk_ids.json', 'r') as f:
    chunk_ids = json.load(f)
with open('chunks.json', 'r') as f:
    chunks = json.load(f)

# --- Function to search similar chunks ---
def search_relevant_chunks(query, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=True)
    distances, indices = index.search(query_embedding.cpu().numpy(), top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'id': chunk_ids[idx],
            'text': chunks[idx],
            'distance': distances[0][i].item()
        })
    return results

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_relevant_chunks', methods=['POST'])
def get_relevant_chunks():
    data = request.get_json()
    prompt = data.get('prompt')
    print(f"[get_relevant_chunks] Received prompt: {prompt}")

    if not prompt:
        return jsonify({'error': 'Prompt is required.'}), 400

    relevant_chunks = search_relevant_chunks(prompt)
    return jsonify({'relevant_chunks': relevant_chunks})

# --- Call Gemini API with context ---
def call_llm_api(prompt, context_chunks):
    system_prompt = (
        "You are an AI assistant for DocuMind, specializing in OpManager and related IT infrastructure topics. Your task is to answer the user's question comprehensively.\n\n"
        "1.  **Prioritize the Provided Context:** First, base your answer on the 'CONTEXT CHUNKS' provided below. Synthesize the information from these chunks to form a coherent answer.\n"
        "2.  **Use Your General Knowledge:** If the context chunks do not contain enough information to fully answer the question, supplement your response with your own general knowledge about OpManager, network monitoring, and related technologies.\n"
        "3.  **Indicate Your Sources:** Clearly state when your answer is based on the provided context and when it is based on your general knowledge.\n"
        "4.  **Provide a Relevance Score:** After the answer, provide a score from 1 to 10 indicating how relevant the provided context was for answering the user's question (1 = Not relevant at all, 10 = Perfectly answered using only the context)."
    )

    formatted_chunks = "\n\n---\n\n".join(context_chunks)
    final_prompt = f"{system_prompt}\n\nCONTEXT CHUNKS:\n{formatted_chunks}\n\nUSER'S QUESTION:\n{prompt}"

    print("[generate] Sending prompt to Gemini:")
    print(final_prompt[:300] + "..." if len(final_prompt) > 300 else final_prompt)  # Print partial prompt

    try:
        response = gemini_model.generate_content(final_prompt)
        return response.text
    except Exception as e:
        print(f"[Gemini ERROR] {e}")
        return "Error generating response from Gemini API."

@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.get_json()
    prompt = data.get('prompt')
    context_chunks = data.get('chunks')

    if not prompt or context_chunks is None:
        return jsonify({'error': 'Prompt and chunks are required.'}), 400

    print(f"[generate] Prompt: {prompt}")
    print(f"[generate] Chunks preview: {context_chunks[:1]}")

    llm_output = call_llm_api(prompt, context_chunks)
    return jsonify({'response': llm_output})

# --- Run App ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)