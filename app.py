import streamlit as st
from fetch_articles import fetch_articles
from chunking import chunk_text
from embeddings import embed_chunks, model
from vector_store import build_index
from bm25_search import build_bm25_index
from summarize import generate_answer
from guardrails import safe_rag_pipeline

st.set_page_config(page_title="NewsRAG", page_icon="📰")
st.title("📰 NewsRAG — AI News Q&A")
st.write("Ask a question about recent AI-related news, and get a grounded, cited answer.")


# Cache this so we don't re-fetch/re-embed articles every single time someone asks a question
@st.cache_resource
def load_pipeline():
    articles = fetch_articles("artificial intelligence", page_size=2)
    all_chunks = []
    for article in articles:
        all_chunks.extend(chunk_text(article["text"]))

    vectors = embed_chunks(all_chunks)
    faiss_index = build_index(vectors)
    bm25_index = build_bm25_index(all_chunks)

    return all_chunks, faiss_index, bm25_index


with st.spinner("Loading articles and building search index..."):
    all_chunks, faiss_index, bm25_index = load_pipeline()

question = st.text_input("Your question:")

if st.button("Get Answer") and question:
    with st.spinner("Searching and generating answer..."):
        result = safe_rag_pipeline(
            question, all_chunks, faiss_index, bm25_index, model, generate_answer
        )

    st.subheader("Answer")
    st.write(result["answer"])

    if result["sources"]:
        st.subheader("Sources used")
        for i, source in enumerate(result["sources"]):
            with st.expander(f"Source {i+1}"):
                st.write(source)