from sentence_transformers import util


def map_answer_to_sources(answer, retrieved_chunks, embed_model):
    # Split the answer into individual sentences (simple split on periods)
    answer_sentences = [s.strip() for s in answer.split(".") if s.strip()]

    # Embed the answer sentences AND the source chunks
    sentence_embeddings = embed_model.encode(answer_sentences)
    chunk_embeddings = embed_model.encode(retrieved_chunks)

    results = []
    for i, sentence in enumerate(answer_sentences):
        # Compare this one answer-sentence against ALL retrieved chunks
        similarities = util.cos_sim(sentence_embeddings[i], chunk_embeddings)[0]
        best_match_idx = similarities.argmax().item()
        best_score = similarities[best_match_idx].item()

        results.append({
            "sentence": sentence,
            "source_chunk_index": best_match_idx,
            "similarity": best_score,
        })

    return results


if __name__ == "__main__":
    from fetch_articles import fetch_articles
    from chunking import chunk_text
    from embeddings import embed_chunks, model
    from vector_store import build_index, search_index
    from summarize import generate_answer

    articles = fetch_articles("artificial intelligence", page_size=1)
    chunks = chunk_text(articles[0]["text"])
    vectors = embed_chunks(chunks)
    index = build_index(vectors)

    query = "What did Apple accuse OpenAI of stealing?"
    query_vector = model.encode(query)
    matched_indices, distances = search_index(index, query_vector, top_k=3)
    retrieved_chunks = [chunks[i] for i in matched_indices]

    answer = generate_answer(query, retrieved_chunks)

    mapping = map_answer_to_sources(answer, retrieved_chunks, model)

    print(f"Answer:\n{answer}\n")
    print("--- Source mapping ---")
    for item in mapping:
        print(f"\nSentence: {item['sentence']}")
        print(f"Best-matching source chunk (similarity: {item['similarity']:.4f}):")
        print(retrieved_chunks[item['source_chunk_index']][:200], "...")