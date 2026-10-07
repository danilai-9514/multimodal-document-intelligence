import sys
from pathlib import Path

from config import OPENAI_API_KEY
from extract_data import extract_chunks, categorize_chunks, get_images_base64
from summarize import summarize_text, summarize_tables, summarize_images
from vector_store import setup_retriever, load_into_retriever
from rag_pipeline import create_rag_chain, answer_with_citations


def main():
    pdf_file = "sample.pdf"
    if not Path(pdf_file).exists():
        print("Please place a PDF named sample.pdf in the project root.")
        sys.exit(1)

    if not OPENAI_API_KEY:
        print("Set OPENAI_API_KEY in your .env file.")
        sys.exit(1)

    chunks = extract_chunks(pdf_file)
    categorized = categorize_chunks(chunks)
    texts = [str(chunk) for chunk in categorized["text"]]
    tables = [str(chunk) for chunk in categorized["tables"]]
    images = get_images_base64(chunks)

    retriever = setup_retriever()
    load_into_retriever(retriever, texts, summarize_text(texts), "text")
    load_into_retriever(retriever, tables, summarize_tables(tables), "table")
    load_into_retriever(retriever, images, summarize_images(images), "image")

    rag_chain = create_rag_chain(retriever)
    while True:
        question = input("Ask a question (or type 'quit'): ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        result = answer_with_citations(rag_chain, question)
        print("\nAnswer:")
        print(result["answer"])


if __name__ == "__main__":
    main()
