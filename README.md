# NewsRAG - Hybrid RAG News Q&A System

A question-answering system built on top of real, live news articles. Instead of relying on an LLM's fixed training knowledge, this project retrieves relevant article content and generates answers grounded in that retrieved text.

The system fetches current news from The Guardian API, splits articles into chunks, and retrieves relevant chunks using a hybrid of semantic search (FAISS) and keyword search (BM25). Retrieved chunks are passed to an LLM (Llama 3.1 via Groq) with instructions to answer only from the provided context. Each generated answer is also mapped back to its source text using cosine similarity, so answers are traceable rather than a black box. The system includes basic guardrails for invalid or out-of-scope questions, an evaluation pipeline to measure retrieval and answer accuracy, and a Streamlit interface for asking questions and viewing sources.

## How it works

```
Guardian API → Chunking → Embeddings → FAISS + BM25 (Hybrid Search)
     → Guardrails → LLM Answer Generation → Source Mapping → Streamlit UI
```

1. A question is checked for basic validity (not empty, not too short)
2. Hybrid search retrieves the most relevant article chunks - FAISS handles semantic/meaning-based matches, BM25 handles exact keyword matches, and the two are combined into one ranked result
3. The retrieved chunks are passed to Llama 3.1 with explicit instructions to answer only using that context, and to say so if the context isn't sufficient
4. Each sentence in the answer is compared against the retrieved chunks (cosine similarity) to identify which chunk it's most likely based on
5. The question, answer, and expandable source chunks are shown in a Streamlit web app

## Tech stack

- News source: The Guardian API
- Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
- Dense search: FAISS
- Sparse search: BM25 (rank_bm25)
- Generation: Groq (Llama 3.1 8B)
- Interface: Streamlit
- Secrets handling: .env + python-dotenv

## Project structure

```
newsrag_project/
├── fetch_articles.py       # Pulls live articles from The Guardian API
├── chunking.py             # Splits articles into overlapping text chunks
├── embeddings.py           # Converts chunks into vector embeddings
├── vector_store.py         # Builds and searches the FAISS semantic index
├── bm25_search.py          # Builds and searches the BM25 keyword index
├── hybrid_search.py        # Combines FAISS + BM25 into one ranked result
├── summarize.py            # Generates grounded answers via Groq Llama 3
├── explainability.py       # Maps answer sentences back to source chunks
├── guardrails.py           # Input validation and retrieval confidence checks
├── evaluate.py             # Evaluation suite (retrieval + answer accuracy)
├── app.py                  # Streamlit web interface
└── .env                    # API keys (not committed)
```

## Evaluation

A test set was built from real fetched articles, with independently verified answers used to check correctness rather than trusting the system's own output.

- Retrieval accuracy: 92.3%
- Answer accuracy: 84.6%

## Running it locally

Clone the repository:
```
git clone https://github.com/ad80702/newsrag-hybrid-search.git
cd newsrag-hybrid-search
```

Install dependencies:
```
pip install requests python-dotenv sentence-transformers faiss-cpu rank-bm25 groq streamlit
```

Create a `.env` file in the project root:
```
GUARDIAN_API_KEY=your_guardian_api_key
GROQ_API_KEY=your_groq_api_key
```
- Guardian API key (free): https://open-platform.theguardian.com/access/
- Groq API key (free): https://console.groq.com

Run the app:
```
streamlit run app.py
```

## Future improvements

- Sentence or paragraph-aware chunking instead of fixed character length
- Absolute (rather than relative) confidence thresholds for the retrieval guardrail
- A larger evaluation set covering more articles and question types
- Support for additional news categories beyond AI

## License

MIT License

## Author

Aditi Manivannan

Github: @ad80702
