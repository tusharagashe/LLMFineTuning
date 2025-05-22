from reranker import search_milvus, rerank

class RerankerIntegration:
    def __init__(self, milvus_db: str, collection_name: str):
        self.milvus_db = milvus_db
        self.collection_name = collection_name

    def query_and_rerank(self, query: str, top_k: int = 10, top_n: int = 5):

        # embeds the chunks into vectors using the nvidia embedding nim
        passages = search_milvus(query, self.milvus_db, self.collection_name, top_k=top_k)
        # reranks the embeddings using the nvidia reranker nim
        return rerank(query, passages, top_n=top_n) 