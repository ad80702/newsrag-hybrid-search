from fetch_articles import fetch_articles
from chunking import chunk_text
from embeddings import embed_chunks, model
from vector_store import build_index, search_index
from bm25_search import build_bm25_index
from hybrid_search import hybrid_search
from summarize import generate_answer


test_set = [
    {
        "question": "What did Apple accuse OpenAI of stealing?",
        "correct_chunk_keyword": "trade secrets",
    },
    {
        "question": "Who is Tang Yew Tan?",
        "correct_chunk_keyword": "chief hardware officer",
    },
    {
        "question": "What did Chang Liu take when he left Apple?",
        "correct_chunk_keyword": "laptop",
    },
    {
        "question": "What company did OpenAI acquire to enter the hardware business?",
        "correct_chunk_keyword": "io Products",
    },
    {
        "question": "Which AI model did Apple use for Siri instead of ChatGPT?",
        "correct_chunk_keyword": "Gemini",
    },
    {
        "question": "How much did OpenAI spend to acquire Jony Ive's startup?",
        "correct_chunk_keyword": "$6.4bn",
    },
    {
        "question": "What company got hacked by OpenAI's AI agents?",
        "correct_chunk_keyword": "Hugging Face",
    },
    {
        "question": "What task were the OpenAI models given when they went rogue?",
        "correct_chunk_keyword": "hacking challenge",
    },
    {
        "question": "Who popularized the paperclip maximizer thought experiment?",
        "correct_chunk_keyword": "Nick Bostrom",
    },
    {
        "question": "What is the nightmare scenario researchers worry about, where a model copies itself?",
        "correct_chunk_keyword": "exfiltrating",
    },
    {
        "question": "Was any sensitive data actually stolen in the Hugging Face incident?",
        "correct_chunk_keyword": "no particularly sensitive data",
    },
    {
        "question": "Who wrote the article about OpenAI's rogue agents?",
        "correct_chunk_keyword": "Shakeel Hashim",
    },
    {
        "question": "What did Liu use to breach Apple's internal network?",
        "correct_chunk_keyword": "authentication bug",
    },
]


def run_evaluation():
    articles = fetch_articles("artificial intelligence", page_size=2)

    # Combine chunks from BOTH articles into one shared list
    all_chunks = []
    for article in articles:
        all_chunks.extend(chunk_text(article["text"]))

    vectors = embed_chunks(all_chunks)
    faiss_index = build_index(vectors)
    bm25_index = build_bm25_index(all_chunks)

    retrieval_correct = 0
    answer_correct = 0
    total = len(test_set)

    for i, test_case in enumerate(test_set):
        question = test_case["question"]
        keyword = test_case["correct_chunk_keyword"]

        top_indices, scores = hybrid_search(
            question, all_chunks, faiss_index, bm25_index, model, top_k=3
        )
        retrieved_chunks = [all_chunks[idx] for idx in top_indices]

        retrieval_hit = any(keyword.lower() in chunk.lower() for chunk in retrieved_chunks)
        if retrieval_hit:
            retrieval_correct += 1

        answer = generate_answer(question, retrieved_chunks)
        answer_hit = keyword.lower() in answer.lower()
        if answer_hit:
            answer_correct += 1

        print(f"--- Test {i+1}: {question} ---")
        print(f"Retrieval correct: {retrieval_hit}")
        print(f"Answer: {answer}")
        print(f"Answer correct: {answer_hit}\n")

    print("=== FINAL RESULTS ===")
    print(f"Retrieval accuracy: {retrieval_correct}/{total} ({retrieval_correct/total*100:.1f}%)")
    print(f"Answer accuracy: {answer_correct}/{total} ({answer_correct/total*100:.1f}%)")


if __name__ == "__main__":
    run_evaluation()