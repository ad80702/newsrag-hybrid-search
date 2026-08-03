def check_query_validity(question):
    """Basic input validation before running the pipeline at all."""
    if not question or not question.strip():
        return False, "Question is empty."
    if len(question.strip()) < 3:
        return False, "Question is too short to be meaningful."
    return True, None


def check_retrieval_confidence(scores, threshold=0.3):
    """Check if the hybrid search scores are actually good enough to trust."""
    if not scores:
        return False
    best_score = max(scores)
    return best_score >= threshold


def safe_rag_pipeline(question, chunks, faiss_index, bm25_index, embed_model, generate_fn, threshold=0.3):
    from hybrid_search import hybrid_search

    # Guardrail 1: input validation
    is_valid, reason = check_query_validity(question)
    if not is_valid:
        return {"answer": f"Sorry, I can't process this question: {reason}", "sources": []}

    # Run retrieval as normal
    top_indices, scores = hybrid_search(question, chunks, faiss_index, bm25_index, embed_model, top_k=3)

    # Guardrail 2: confidence check
    if not check_retrieval_confidence(scores, threshold):
        return {
            "answer": "I don't have enough relevant information in my knowledge base to answer this question confidently.",
            "sources": [],
        }

    # If we passed both guardrails, proceed normally
    retrieved_chunks = [chunks[i] for i in top_indices]
    answer = generate_fn(question, retrieved_chunks)

    return {"answer": answer, "sources": retrieved_chunks}


if __name__ == "__main__":
    from fetch_articles import fetch_articles
    from chunking import chunk_text
    from embeddings import embed_chunks, model
    from vector_store import build_index
    from bm25_search import build_bm25_index
    from summarize import generate_answer

    articles = fetch_articles("artificial intelligence", page_size=2)
    all_chunks = []
    for article in articles:
        all_chunks.extend(chunk_text(article["text"]))

    vectors = embed_chunks(all_chunks)
    faiss_index = build_index(vectors)
    bm25_index = build_bm25_index(all_chunks)

    test_questions = [
        "What did Apple accuse OpenAI of stealing?",   # should work normally
        "",                                              # empty input
        "hi",                                            # too short
        "What's the best recipe for chocolate cake?",   # unrelated topic
    ]

    for q in test_questions:
        print(f"--- Question: '{q}' ---")
        result = safe_rag_pipeline(q, all_chunks, faiss_index, bm25_index, model, generate_answer)
        print(f"Answer: {result['answer']}")
        print(f"Number of sources used: {len(result['sources'])}\n")