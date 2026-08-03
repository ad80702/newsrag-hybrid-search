import numpy as np


def normalize(scores):
    scores = np.array(scores)
    if scores.max() == scores.min():
        return np.ones_like(scores)  # avoid divide-by-zero if all scores are equal
    return (scores - scores.min()) / (scores.max() - scores.min())


def hybrid_search(query, chunks, faiss_index, bm25_index, embed_model, top_k=3, alpha=0.5):
    from vector_store import search_index
    from bm25_search import search_bm25

    # Get FAISS results (distances) for ALL chunks, not just top few
    query_vector = embed_model.encode(query)
    faiss_indices, faiss_distances = search_index(faiss_index, query_vector, top_k=len(chunks))

    # Get BM25 results (scores) for ALL chunks
    bm25_indices, bm25_scores_top = search_bm25(bm25_index, query, chunks, top_k=len(chunks))

    # Build full score arrays aligned by chunk position (0 to len(chunks)-1)
    faiss_full = np.zeros(len(chunks))
    for idx, dist in zip(faiss_indices, faiss_distances):
        faiss_full[idx] = dist

    bm25_full = np.zeros(len(chunks))
    for idx, score in zip(bm25_indices, bm25_scores_top):
        bm25_full[idx] = score

    # Normalize: for FAISS, smaller distance = better, so we flip it (1 - normalized)
    faiss_norm = 1 - normalize(faiss_full)
    bm25_norm = normalize(bm25_full)

    # Combine: alpha controls how much weight semantic vs keyword search gets
    combined_scores = alpha * faiss_norm + (1 - alpha) * bm25_norm

    ranked = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)
    top_indices = ranked[:top_k]

    return top_indices, [combined_scores[i] for i in top_indices]


if __name__ == "__main__":
    from fetch_articles import fetch_articles
    from chunking import chunk_text
    from embeddings import embed_chunks, model
    from vector_store import build_index
    from bm25_search import build_bm25_index

    articles = fetch_articles("artificial intelligence", page_size=1)
    chunks = chunk_text(articles[0]["text"])
    vectors = embed_chunks(chunks)

    faiss_index = build_index(vectors)
    bm25_index = build_bm25_index(chunks)

    query = "Tang Yew Tan"
    top_indices, scores = hybrid_search(query, chunks, faiss_index, bm25_index, model, top_k=3)

    print(f"Question: {query}\n")
    for rank, idx in enumerate(top_indices):
        print(f"--- Match {rank+1} (combined score: {scores[rank]:.4f}) ---")
        print(chunks[idx])
        print()