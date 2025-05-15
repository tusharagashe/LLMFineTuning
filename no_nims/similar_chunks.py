import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def retrieve_chunks(query, k=10, faiss_path="./faiss_db/faiss.index", metadata_path="./faiss_db/metadata.jsonl"):
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    index = faiss.read_index(faiss_path)

    with open(metadata_path, 'r') as f:
        metadata = [json.loads(line) for line in f]

    query_vec = embedder.encode([query], convert_to_numpy=True)

    D, I = index.search(query_vec, k)
    results = [metadata[i] for i in I[0]]

    return results

top_chunks = retrieve_chunks("Does FOXM1 stratify TNBC response?", k=10)

