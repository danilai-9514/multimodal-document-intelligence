import base64
import os
import tempfile
from pathlib import Path

import streamlit as st

from config import OPENAI_API_KEY
from extract_data import extract_chunks, categorize_chunks, get_images_base64
from summarize import summarize_text, summarize_tables, summarize_images
from vector_store import setup_retriever, load_into_retriever
from rag_pipeline import create_rag_chain, answer_with_citations

st.set_page_config(page_title="Multimodal Document Intelligence", page_icon="📄", layout="wide")
st.title("📄 Multimodal Document Intelligence")
st.caption("Professional PDF QA system with multimodal retrieval and citation-aware answers.")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded_file:
        b64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="700" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

with col2:
    st.header("Query the document")
    openai_key = st.text_input("OpenAI API Key", type="password")
    groq_key = st.text_input("Groq API Key (Optional)", type="password")

    if st.button("Process PDF"):
        if uploaded_file is None:
            st.error("Please upload a PDF first.")
        elif not openai_key and not OPENAI_API_KEY:
            st.error("Please provide an OpenAI API key.")
        else:
            key = openai_key or OPENAI_API_KEY
            os.environ["OPENAI_API_KEY"] = key
            if groq_key:
                os.environ["GROQ_API_KEY"] = groq_key

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                pdf_path = tmp.name

            with st.spinner("Processing PDF and building index..."):
                try:
                    chunks = extract_chunks(pdf_path)
                    categorized = categorize_chunks(chunks)
                    texts = [str(chunk) for chunk in categorized["text"]]
                    tables = [str(chunk) for chunk in categorized["tables"]]
                    images = get_images_base64(chunks)

                    text_summaries = summarize_text(texts) if texts else []
                    table_summaries = summarize_tables(tables) if tables else []
                    image_summaries = summarize_images(images) if images else []

                    retriever = setup_retriever()
                    if retriever is not None:
                        load_into_retriever(retriever, texts, text_summaries, "text")
                        load_into_retriever(retriever, tables, table_summaries, "table")
                        load_into_retriever(retriever, images, image_summaries, "image")
                        st.session_state.rag_chain = create_rag_chain(retriever)
                    else:
                        st.error("Retriever initialization failed. Check your API keys and dependencies.")
                        st.stop()

                    st.success("PDF processed successfully.")
                finally:
                    Path(pdf_path).unlink(missing_ok=True)

    question = st.text_area("Question", height=120, placeholder="e.g. What are the key findings in this report?")
    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif st.session_state.rag_chain is None:
            st.warning("Please process a PDF before asking a question.")
        else:
            with st.spinner("Generating answer..."):
                result = answer_with_citations(st.session_state.rag_chain, question)
                if result.get("success"):
                    st.markdown("### Answer")
                    st.write(result["answer"])
                    if result.get("sources"):
                        st.markdown("### Sources")
                        for source in result["sources"]:
                            st.write(source)
                else:
                    st.error(result.get("answer", "Unable to generate answer."))
