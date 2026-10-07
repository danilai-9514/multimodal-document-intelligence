# Multimodal Document Intelligence

A multimodal document Q&A system inspired by the same structure as AdarshP2's Multi-Modal-RAG project, but adapted for your repository and simplified to be easier to run and extend.

## What it does
- Upload a PDF
- Extract text, tables, and images
- Summarize extracted content
- Index chunks in a vector store
- Answer questions using retrieval-augmented generation
- Return answers with source citations

## Project structure
- `app/` — backend logic for ingestion, retrieval, and answer generation
- `app.py` — Streamlit UI for uploading PDFs and asking questions
- `main.py` — CLI entry point
- `requirements.txt` — Python dependencies
- `.env.example` — environment variables

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

For the API backend:

```bash
uvicorn app.main:app --reload
```

## Example API calls

```bash
curl -X POST "http://localhost:8000/upload" -F "file=@sample.pdf"
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"question":"What are the key findings in this document?"}'
```

## Environment variables
Create a `.env` file using `.env.example` and add your API keys.

## Notes
This version is built to be a practical starter for:
- PDF-based multimodal QA
- OCR + table extraction
- chart/image understanding
- citation-grounded answers
- future extension with VLMs and dashboard UI
