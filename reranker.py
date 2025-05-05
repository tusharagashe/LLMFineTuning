import argparse
from pymilvus import MilvusClient
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

EMBEDDING_API_KEY = os.getenv("NVIDIA_EMBEDDING_API_KEY")
RERANKER_API_KEY = os.getenv("NVIDIA_RERANKER_API_KEY")
MILVUS_DB_PATH = "milvus_lite.db"
COLLECTION_NAME = "risk_chunks"
VECTOR_DIM = 4096  

def search_milvus(query, top_k=10):
    print(query)
    client = MilvusClient(uri=MILVUS_DB_PATH)

    embedder = NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        api_key=EMBEDDING_API_KEY,  
        truncate="NONE"
    )
    query_vec = embedder.embed_query(query)

    res = client.search(
        collection_name=COLLECTION_NAME,
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

    documents = [Document(page_content=p) for p in passages]
    results = client.compress_documents(query=query, documents=documents)
    return [doc.page_content for doc in results[:top_n]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top_k", type=int, default=10, help="Top K retrieved from Milvus")
    parser.add_argument("--top_n", type=int, default=5, help="Top N reranked chunks")
    args = parser.parse_args()

    retrieved = search_milvus(args.query, args.top_k)
    top_chunks = rerank(args.query, retrieved, args.top_n)

    print("\nTOP RERANKED CHUNKS:")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"\nRank {i}:\n{chunk[:500]}...\n")

if __name__ == "__main__":
    main()
