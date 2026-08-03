def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # move forward, but step back by the overlap amount

    return chunks


if __name__ == "__main__":
    from fetch_articles import fetch_articles

    articles = fetch_articles("artificial intelligence", page_size=2)  # just 2 articles for testing

    for article in articles:
        print(f"\n=== {article['title']} ===")
        chunks = chunk_text(article["text"])
        print(f"Split into {len(chunks)} chunks\n")
        for i, c in enumerate(chunks):
            print(f"--- Chunk {i+1} ---")
            print(c)
            print()