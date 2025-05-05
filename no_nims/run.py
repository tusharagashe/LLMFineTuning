from similar_chunks import retrieve_chunks
from reranker import rerank_chunks
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="End-to-end retrieval and reranking pipeline.")
    parser.add_argument("--query", type=str, required=True, help="User query for retrieval.")
    parser.add_argument("--faiss_index", type=str, default="./faiss_db/faiss.index", help="Path to FAISS index.")
    parser.add_argument("--metadata", type=str, default="./faiss_db/metadata.jsonl", help="Path to metadata JSONL.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of chunks to retrieve from FAISS.")
    parser.add_argument("--top_n", type=int, default=5, help="Number of reranked chunks to keep.")

    args = parser.parse_args()

    print(f"\nRunning query: {args.query}")
    retrieved = retrieve_chunks(args.query, k=args.top_k, faiss_path=args.faiss_index, metadata_path=args.metadata)
    reranked = rerank_chunks(args.query, retrieved, top_n=args.top_n)


if __name__ == "__main__":
    main()
