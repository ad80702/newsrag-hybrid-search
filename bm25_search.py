from rank_bm25 import BM25Okapi


def build_bm25_index(chunks):
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25


def search_bm25(bm25, query, chunks, top_k=3):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # pair each chunk with its score, then sort by score (highest first)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_indices = ranked[:top_k]

    return top_indices, [scores[i] for i in top_indices]


if __name__ == "__main__":
    from fetch_articles import fetch_articles
    from chunking import chunk_text

    articles = fetch_articles("artificial intelligence", page_size=1)
    chunks = chunk_text(articles[0]["text"])

    bm25 = build_bm25_index(chunks)

    query = "Tang Yew Tan"
    top_indices, scores = search_bm25(bm25, query, chunks, top_k=3)

    print(f"Question: {query}\n")
    for rank, idx in enumerate(top_indices):
        print(f"--- Match {rank+1} (score: {scores[rank]:.4f}) ---")
        print(chunks[idx])
        print()