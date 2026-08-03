import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(question, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""Answer the question using ONLY the information in the context below.
If the context doesn't contain enough information to answer, say so — do not make anything up.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    from fetch_articles import fetch_articles
    from chunking import chunk_text
    from embeddings import embed_chunks, model
    from vector_store import build_index, search_index

    articles = fetch_articles("artificial intelligence", page_size=1)
    chunks = chunk_text(articles[0]["text"])
    vectors = embed_chunks(chunks)
    index = build_index(vectors)

    query = "What did Apple accuse OpenAI of stealing?"
    query_vector = model.encode(query)
    matched_indices, distances = search_index(index, query_vector, top_k=3)

    retrieved_chunks = [chunks[i] for i in matched_indices]

    answer = generate_answer(query, retrieved_chunks)

    print(f"Question: {query}\n")
    print(f"Answer:\n{answer}")