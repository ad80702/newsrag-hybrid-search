import faiss
import numpy as np


def build_index(embeddings):
    dimension = embeddings.shape[1]        # 384, from our embedding model
    index = faiss.IndexFlatL2(dimension)   # creates an empty "filing cabinet" sized for 384-number vectors
    index.add(np.array(embeddings))        # loads all our vectors into it
    return index


def search_index(index, query_vector, top_k=3):
    query_vector = np.array([query_vector])  # FAISS expects a 2D array, even for one query
    distances, indices = index.search(query_vector, top_k)
    return indices[0], distances[0]  # which chunks matched, and how close they were


if __name__ == "__main__":
    from fetch_articles import fetch_articles
    from chunking import chunk_text
    from embeddings import embed_chunks, model

    articles = fetch_articles("artificial intelligence", page_size=1)
    chunks = chunk_text(articles[0]["text"])
    vectors = embed_chunks(chunks)

    index = build_index(vectors)

    query = "What did Apple accuse OpenAI of stealing?"
    query_vector = model.encode(query)

    matched_indices, distances = search_index(index, query_vector, top_k=3)

    print(f"Question: {query}\n")
    for rank, idx in enumerate(matched_indices):
        print(f"--- Match {rank+1} (distance: {distances[rank]:.4f}) ---")
        print(chunks[idx])
        print()