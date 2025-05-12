import argparse
import os
from dotenv import load_dotenv
from pymilvus import MilvusClient
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from langchain_core.documents import Document

# Load API keys from .env
load_dotenv()
EMBEDDING_API_KEY = os.getenv("NVIDIA_EMBEDDING_API_KEY")
RERANKER_API_KEY = os.getenv("NVIDIA_RERANKER_API_KEY")

def search_milvus(query, milvus_db, collection_name, top_k=10):
    print(milvus_db, collection_name)
    client = MilvusClient(uri=milvus_db, local=True)

    embedder = NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        api_key=EMBEDDING_API_KEY,
        truncate="NONE"
    )
    query_vec = embedder.embed_query(query)

    res = client.search(
        collection_name=collection_name,
        data=[query_vec],
        limit=top_k,
        output_fields=["text"]
    )
    return [hit["entity"]["text"] for hit in res[0]]

def rerank(query, passages, top_n=5):
    client = NVIDIARerank(
        model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
        api_key=RERANKER_API_KEY,
    )
    docs = [Document(page_content=p) for p in passages]
    ranked = client.compress_documents(query=query, documents=docs)
    return [doc.page_content for doc in ranked[:top_n]]

def main():
    parser = argparse.ArgumentParser(description="function to retrieve and rerank passages from a Milvus Lite DB.")
    parser.add_argument("--db", required=True, help="path to Milvus Lite DB file")
    parser.add_argument("--collection", default="risk_chunks", help="Milvus collection name to search, should all be called risk_chunks")
    parser.add_argument("--query", required=True, help="search query string")
    parser.add_argument("--top_k", type=int, default=10, help="num of chunks to retrieve from Milvus")
    parser.add_argument("--top_n", type=int, default=5, help="num of top passages to return after reranking")
    args = parser.parse_args()

    retrieved = search_milvus(args.query, args.db, args.collection, args.top_k)
    top_chunks = rerank(args.query, retrieved, args.top_n)

    print("\nTop Reranked Chunks:")
    for idx, chunk in enumerate(top_chunks, 1):
        print(f"\nRank {idx}:\n{chunk[:500]}...\n")

if __name__ == "__main__":
    main()