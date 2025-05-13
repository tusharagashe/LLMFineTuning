import sqlite3
import os
import argparse
import json
from pymilvus import MilvusClient
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from dotenv import load_dotenv

load_dotenv()
EMBED_KEY = os.getenv("NVIDIA_EMBEDDING_API_KEY")

def checking_sqlite(sqlite_path):
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    # find user tables
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    tables = [r[0] for r in cur.fetchall()]
    if not tables:
        raise RuntimeError("No user tables found in SQLite DB")
    table = tables[0]
    cur.execute(f"PRAGMA table_info({table});")
    cols = cur.fetchall()  # (cid, name, type, notnull, dflt, pk)
    conn.close()

    # id column: pk or first integer
    id_col = next(
        (n for (_, n, t, _, _, pk) in cols if pk == 1),
        next((n for (_, n, t, _, _, _) in cols if "INT" in t.upper()), cols[0][1])
    )
    # text column: first TEXT-like
    text_col = next(
        (n for (_, n, t, _, _, _) in cols
         if any(x in t.upper() for x in ("CHAR", "CLOB", "TEXT"))),
        None
    )
    if not text_col:
        raise RuntimeError("No TEXT column found.")
    # metadata = all others
    meta_cols = [n for (_, n, _, _, _, _) in cols if n not in (id_col, text_col)]
    return table, id_col, text_col, meta_cols

def populate_and_export(sqlite_path):
    base = os.path.splitext(os.path.basename(sqlite_path))[0]
    os.makedirs("db", exist_ok=True)
    os.makedirs("db/json", exist_ok=True)

    milvus_db = f"db/{base}_milvus.db"
    collection = "risk_chunks"
    export_file = f"db/json/{base}_export.json"

    table, id_col, text_col, metas = checking_sqlite(sqlite_path)
    print(table, id_col, text_col, metas)

    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    fields = [id_col, text_col] + metas
    cur.execute(f"SELECT {','.join(fields)} FROM {table}")
    rows = cur.fetchall()
    conn.close()
    #print(len(rows))

    # embed texts
    embedder = NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        api_key=EMBED_KEY,
        truncate="NONE"
    )
    texts = [r[1] or "" for r in rows]
    vectors = embedder.embed_documents(texts)
    dim = len(vectors[0])
    #print(dim)

    # build Milvus entities
    entities = []
    for i, r in enumerate(rows):
        ent = {"id": int(r[0]), "vector": vectors[i], text_col: r[1] or ""}
        for j, m in enumerate(metas):
            ent[m] = r[2 + j]
        entities.append(ent)

    client = MilvusClient(uri=milvus_db, local=True)
    if client.has_collection(collection):
        client.drop_collection(collection)

    client.create_collection(collection_name=collection, dimension=dim, metric_type="COSINE")
    client.insert(collection_name=collection, data=entities)

    # export to JSON
    results = client.query(
        collection,
        output_fields=[id_col, text_col] + metas,
        limit=len(entities)
    )
    with open(export_file, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="function to populate and export a SQLite DB")
    parser.add_argument("--db", required=True, help="path to the SQLite DB file")
    args = parser.parse_args()
    populate_and_export(args.db)
