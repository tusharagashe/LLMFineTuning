import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from pymilvus import MilvusClient
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

load_dotenv()  
EMBED_KEY = os.getenv("NVIDIA_EMBEDDING_API_KEY")

# makes sure Milvus DB filename is <=35 characters bc milvus db filename limit is @ 35 lul
def make_milvus_name(base: str, suffix="_milvus.db", max_len=35) -> str:
    candidate = base + suffix

    if len(candidate) <= max_len:
        return candidate
    
    avail = max_len - len(suffix)

    return base[:avail] + suffix

def load_and_introspect(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    first = data[0]
    id_key = "element_id" if "element_id" in first else None

    meta_keys = []
    if isinstance(first.get("metadata"), dict):
        meta_keys = list(first["metadata"].keys())
        
    return data, id_key, "text", meta_keys

def populate_and_export(json_path):
    data, id_key, text_key, meta_keys = load_and_introspect(json_path)
    #print(f"Loaded {len(data)} entries; id_key={id_key}, text_key='{text_key}', metadata={meta_keys}")
    print(len(data), id_key, text_key, meta_keys)

    base = Path(json_path).stem
    milvus_db_name = make_milvus_name(base)
    milvus_db = Path("db") / milvus_db_name
    collection = "risk_chunks"
    export_file = Path("db/json") / f"{base}_export.json"
    milvus_db.parent.mkdir(parents=True, exist_ok=True)
    export_file.parent.mkdir(parents=True, exist_ok=True)

    # extract texts + gather metadata + og elem id
    texts = []
    metas = []
    for idx, rec in enumerate(data):
        texts.append(rec[text_key] or "")
        md = {}
        if id_key:
            md["element_id"] = rec[id_key]
        for k in meta_keys:
            md[k] = rec.get("metadata", {}).get(k)
        metas.append(md)

    # embedding nim 
    embedder = NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        api_key=EMBED_KEY,
        truncate="NONE"
    )
    vectors = embedder.embed_documents(texts)
    dim = len(vectors[0])
    print(dim)

    entities = []
    for i, txt in enumerate(texts):
        ent = {
            "id": i,          
            "vector": vectors[i],
            text_key: txt
        }
        ent.update(metas[i])  
        entities.append(ent)

    print(milvus_db_name, collection)
    client = MilvusClient(uri=str(milvus_db), local=True)
    if client.has_collection(collection_name=collection):
        client.drop_collection(collection_name=collection)
        print(f"DROPPED EXISTING COLLECTION `{collection}`")
    client.create_collection(collection_name=collection, dimension=dim, metric_type="COSINE")
    client.insert(collection_name=collection, data=entities)

    # visual check w/ JSON
    print(collection,export_file)
    out = client.query(
        collection_name=collection,
        output_fields=["id", text_key] + meta_keys + (["element_id"] if id_key else []),
        limit=len(entities)
    )
    with open(export_file, "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ingest JSON to Milvus Lite w/ export JSON")
    p.add_argument("--json", required=True, help="Path to input JSON file")
    args = p.parse_args()
    populate_and_export(args.json)
