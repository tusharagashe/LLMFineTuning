from pymilvus import MilvusClient
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import sqlite3
from dotenv import load_dotenv
import os
import json

load_dotenv()

EMBEDDING_API_KEY = os.getenv("NVIDIA_EMBEDDING_API_KEY")

def populate_milvus_from_sqlite(sqlite_path, table_name="sba_entries", collection_name="risk_chunks", db_path="milvus_lite.db"):
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT id, header, subheader, text, word_count, sentence_count FROM {table_name}")
    rows = cursor.fetchall()
    #print(len(rows))

    ids = [row[0] for row in rows]
    # text is at index 3
    texts = [row[3] for row in rows] 

    embedder = NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        api_key=EMBEDDING_API_KEY,
        truncate="NONE"
    )
    vectors = embedder.embed_documents(texts)
    #print(len(vectors))

    data = [{
        "id": int(row[0]),
        "vector": vectors[i],
        "text": row[3],
        "header": row[1],
        "subheader": row[2],
        "word_count": row[4],
        "sentence_count": row[5]
    } for i, row in enumerate(rows)]

    client = MilvusClient(db_path)
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # collection called "risk_chunks"
    client.create_collection(
        collection_name=collection_name,
        dimension=len(vectors[0]),
        metric_type="COSINE"
    )

    client.insert(collection_name=collection_name, data=data)

    # json file so we can view data
    results = client.query(collection_name, output_fields=["id", "header", "subheader", "word_count", "sentence_count", "text"], limit=100)
    with open("milvus_export.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    populate_milvus_from_sqlite("sba_1c_database.sqlite")
