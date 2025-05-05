from sentence_transformers import CrossEncoder
import torch
import numpy as np

def rerank_chunks(
    query,
    chunks,
    model_name='cross-encoder/nli-deberta-v3-base',
    top_n=5,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=True
):
    
    reranker = CrossEncoder(model_name, device=device)
    pairs = []
    for chunk in chunks:
        text = chunk['text']
        pair = (query, text)
        pairs.append(pair)

    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        if isinstance(score, (list, np.ndarray)):
            base_score = float(score[0])
        else:
            base_score = float(score)

        reason = []

        chunk['rerank_score'] = base_score
        chunk['rerank_reason'] = ", ".join(reason)

    reranked = sorted(chunks, key=lambda x: -x['rerank_score'])
    top_chunks = reranked[:top_n]

    if verbose:
        print(f"\nTop {top_n} Reranked Chunks for Query: \"{query}\"\n")
        for i, chunk in enumerate(top_chunks):
            print(f"Rank {i+1} | Score: {chunk['rerank_score']:.4f}")
            print(f"→ {chunk['text'].strip()}\n")

    return top_chunks
