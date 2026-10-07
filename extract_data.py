import os
import tempfile
import base64
from pathlib import Path

import streamlit as st

from extract_data import extract_chunks, get_images_base64, categorize_chunks
from summarize import summarize_text, summarize_tables, summarize_images
from vector_store import setup_retriever, load_into_retriever
from rag_pipeline import create_rag_chain, answer_with_citations
from config import UPLOAD_PATH

st.set_page_config(page_title="Multimodal Document Intelligence", layout="wide")
st.title("📄 Multimodal Document Intelligence")
st.markdown("A multimodal PDF Q&A app inspired by the same PDF retrieval pattern used in AdarshP2's Multi-Modal-RAG project.")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload PDF")
    file = st.file_uploader("Choose a PDF", type="pdf")
    if file is not None:
        b64 = base64.b64encode(file.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="700" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

with col2:
    st.header("Ask a Question")
    openai_key = st.text_input("OpenAI API Key", type="password")
    groq_key = st.text_input("Groq API Key", type="password")

    if st.button("Process PDF"):
        if file is None:
            st.error("Please upload a PDF first.")
        elif not openai_key:
            st.error("OpenAI API key is required.")
        else:
            os.environ["OPENAI_API_KEY"] = openai_key
            if groq_key:
                os.environ["GROQ_API_KEY"] = groq_key

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                pdf_path = tmp.name

            with st.spinner("Processing PDF..."):
                chunks = extract_chunks(pdf_path)
                categorized = categorize_chunks(chunks)
                texts = [str(chunk) for chunk in categorized["text"]]
                tables = [str(chunk) for chunk in categorized["tables"]]
                img_data = get_images_base64(chunks)

                text_summaries = summarize_text(texts) if texts else []
                table_summaries = summarize_tables(tables) if tables else []
                image_summaries = summarize_images(img_data) if img_data else []

                retriever = setup_retriever()
                load_into_retriever(retriever, texts, text_summaries, "text")
                load_into_retriever(retriever, tables, table_summaries, "table")
                load_into_retriever(retriever, img_data, image_summaries, "image")
                st.session_state.rag_chain = create_rag_chain(retriever)

            st.success("PDF processed successfully.")
            Path(pdf_path).unlink(missing_ok=True)

    question = st.text_area("Question", height=120)
    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif st.session_state.rag_chain is None:
            st.warning("Please process a PDF first.")
        else:
            result = answer_with_citations(st.session_state.rag_chain, question)
            st.write(result["answer"])
            if result.get("sources"):
                st.markdown("### Sources")
                for s in result["sources"]:
                    st.write(s)
