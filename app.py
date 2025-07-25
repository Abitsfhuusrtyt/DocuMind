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
# Make sure to set your GOOGLE_API_KEY in a .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel(model_name='gemini-1.5-flash')

# --- Flask App ---
app = Flask(__name__)

# --- Load files and models on startup ---
# These print statements help confirm that the files are loaded correctly on startup
print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

print("Loading FAISS index...")
index = faiss.read_index('corpus.index')
print("Index loaded.")

print("Loading chunk data...")
with open('chunk_ids.json', 'r') as f:
    chunk_ids = json.load(f)
with open('chunks.json', 'r') as f:
    chunks = json.load(f)
print("Chunk data loaded.")

# --- Core Functions ---

def search_relevant_chunks(query, top_k=5):
    """Encodes the query and searches the FAISS index for top_k most similar chunks."""
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        # Ensure the index is within the bounds of your data
        if idx < len(chunk_ids) and idx < len(chunks):
            results.append({
                'id': chunk_ids[idx],
                'text': chunks[idx],
                'distance': distances[0][i].item()
            })
    return results

def get_llm_response(prompt):
    """Calls the Gemini API with only the user's prompt."""
    # This system prompt is now much simpler as we are not sending context.
    system_prompt = (
        "You are an AI assistant for DocuMind, specializing in OpManager and related IT infrastructure topics. "
        "Answer the user's question based on your general knowledge."
    )
    
    final_prompt = f"{system_prompt}\n\nUSER'S QUESTION:\n{prompt}"

    print("[get_llm_response] Sending simplified prompt to Gemini (NO CHUNKS)...")
    
    try:
        response = gemini_model.generate_content(final_prompt)
        return response.text
    except Exception as e:
        print(f"[Gemini ERROR] {e}")
        return "An error occurred while generating the response from the AI model."

# --- API Routes ---

@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Main endpoint to handle user interaction.
    1. Receives a prompt.
    2. Finds relevant chunks from the local vector database.
    3. Gets a response from the LLM based ONLY on the prompt.
    4. Returns both chunks and the LLM response to the UI.
    """
    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'error': 'A prompt is required.'}), 400

    print(f"\n[chat] Received prompt: {prompt}")

    # Step 1: Get relevant chunks locally
    relevant_chunks = search_relevant_chunks(prompt)
    print(f"[chat] Found {len(relevant_chunks)} relevant chunks.")

    # Step 2: Get LLM response using only the prompt to save API tokens
    llm_output = get_llm_response(prompt)
    
    # Step 3: Send both back to the UI in a single response
    return jsonify({
        'relevant_chunks': relevant_chunks,
        'llm_response': llm_output
    })

# --- Run App ---
if __name__ == '__main__':
    # Use the PORT environment variable if available, otherwise default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Set debug=False for production environments
    app.run(debug=False, host="0.0.0.0", port=port)