import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. Load a pre-trained model for creating embeddings.
#    'all-MiniLM-L6-v2' is a good starting point for general purpose tasks.
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Load your text chunks from the JSONL file.
chunks = []
chunk_ids = []
with open('1corpus.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        chunks.append(data['text'])
        chunk_ids.append(data['_id'])

print(f"Successfully loaded {len(chunks)} chunks from 1corpus.jsonl.")

# 3. Generate embeddings for all the chunks.
#    This may take some time depending on the number of chunks and your hardware.
print("Generating embeddings for the chunks...")
chunk_embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)

# 4. Create a FAISS index.
#    IndexFlatL2 is a simple index that performs a brute-force L2 distance search.
embedding_dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)

# Add the chunk embeddings to the index.
index.add(chunk_embeddings.cpu().numpy())

print(f"Number of chunks in the FAISS index: {index.ntotal}")

# 5. Save the FAISS index and the chunk IDs to files.
faiss.write_index(index, 'corpus.index')
with open('chunk_ids.json', 'w') as f:
    json.dump(chunk_ids, f)

print("FAISS index and chunk IDs have been saved.")

# Optional: To map index positions back to original text, you can save the chunks as well.
with open('chunks.json', 'w') as f:
    json.dump(chunks, f)

print("Original chunks saved for easy retrieval.")