from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks):
    embeddings = model.encode(chunks)
    return embeddings


if __name__ == "__main__":
    from fetch_articles import fetch_articles
    from chunking import chunk_text

    articles = fetch_articles("artificial intelligence", page_size=1)
    chunks = chunk_text(articles[0]["text"])

    vectors = embed_chunks(chunks)

    print(f"Number of chunks: {len(chunks)}")
    print(f"Shape of embeddings: {vectors.shape}")
    print(f"First chunk's vector (first 10 numbers):")
    print(vectors[0][:10])