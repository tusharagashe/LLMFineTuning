import os
import argparse
from dotenv import load_dotenv
from pymilvus import MilvusClient
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from langchain_core.documents import Document

load_dotenv()
EMBEDDING_API_KEY = os.getenv("NVIDIA_EMBEDDING_API_KEY")
RERANKER_API_KEY = os.getenv("NVIDIA_RERANKER_API_KEY")

def search_milvus(query: str, milvus_db: str, collection_name: str, top_k: int = 10):
    client = MilvusClient(uri=milvus_db, local=True)

    if not client.has_collection(collection_name=collection_name):
        existing = client.list_collections()
        raise RuntimeError(
            f"Collection {collection_name!r} not found in {milvus_db!r}.\n"
            f"Existing collections: {existing}"
        )

    embedder = NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        api_key=EMBEDDING_API_KEY,
        truncate="NONE",
    )
    qvec = embedder.embed_query(query)

    results = client.search(
        collection_name=collection_name,
        data=[qvec],
        limit=top_k,
        output_fields=["text"],
    )
    return [hit.entity.get("text") for hit in results[0]]

def rerank(query: str, passages: list[str], top_n: int = 5):
    print(len(passages))
    client = NVIDIARerank(
        model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
        api_key=RERANKER_API_KEY,
    )
    docs = [Document(page_content=p) for p in passages]
    ranked = client.compress_documents(query=query, documents=docs)
    return [d.page_content for d in ranked[:top_n]]

def main():
    parser = argparse.ArgumentParser(
        description="retrieve + rerank passages from a Milvus-Lite DB using NVIDIA NIMs"
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to milvus .db file",
    )
    parser.add_argument(
        "--collection",
        default="risk_chunks",
        help="milvus collection name to search, default is set to 'risk_chunks'",
    )
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="how many passages to retrieve before reranking",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="how many to return after reranking",
    )
    args = parser.parse_args()

    # step 1 w/ searching thru milvus
    passages = search_milvus(args.query, args.db, args.collection, top_k=args.top_k)
#   if not passages:
#       return

    # rerank w/ nim
    top_chunks = rerank(args.query, passages, top_n=args.top_n)

    # output, only prints out the first 1000 chars of each chunk 
    print("\nTOP RERANKED CHUNKS\n")
    for i, chunk in enumerate(top_chunks, start=1):
        print(f"RANK {i}")
        #print(chunk, "...\n")
        print(chunk[:1000], "...\n")

if __name__ == "__main__":
    main()
