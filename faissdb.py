import os
import faiss
import json
import sqlite3
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# might need to test with different models like BioBERTa, etc.
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'  
SQLITE_DB_PATH = './sba_1c_database.sqlite'
TABLE_NAME = 'sba_entries'
OUTPUT_DIR = './faiss_db/'

os.makedirs(OUTPUT_DIR, exist_ok=True)

conn = sqlite3.connect(SQLITE_DB_PATH)
cursor = conn.cursor()

query = f"SELECT id, header, subheader, text FROM {TABLE_NAME}"
cursor.execute(query)
rows = cursor.fetchall()

texts = []
metadata = []

for idx, row in enumerate(rows):
    id_, header, subheader, text = row
    if text:  # skip empty
        texts.append(text)
        metadata.append({
            "id": id_,
            "header": header,
            "subheader": subheader,
            "text": text
        })

print(f"Loaded {len(texts)} text chunks.")

# load embedding model
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# embed chunks
embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
embedding_dim = embeddings.shape[1]

# faiss index   
# using IndexFlatL2 for now since it's fast for small datasets, but apparently it doesn't scale so for now
# we can use IndexIVFFlat, IndexIVFPQ, or IndexHNSWFlat (might need to test which is better) 
# for larger datasets once we have more data
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# SAVE
faiss_index_path = os.path.join(OUTPUT_DIR, 'faiss.index')
faiss.write_index(index, faiss_index_path)
print(f"FAISS index saved to {faiss_index_path}")

metadata_path = os.path.join(OUTPUT_DIR, 'metadata.jsonl')
with open(metadata_path, 'w') as f:
    for meta in metadata:
        f.write(json.dumps(meta) + '\n')

print(f"Metadata saved to {metadata_path}")
print("done")
