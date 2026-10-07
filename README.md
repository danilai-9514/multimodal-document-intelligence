# Multimodal Document Intelligence

This repository contains a starter implementation for a system that reads mixed-document inputs and answers questions with citations.

## Features
- PDF ingestion scaffold
- FastAPI backend
- Question answering API with citation format
- Extensible architecture for OCR, tables, charts, and multimodal retrieval

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## API endpoints

- `GET /` - health check
- `POST /ask` - ask a question
- `POST /upload` - upload a PDF

## Notes
This is a starter project designed to be extended with:
- OCR for scanned pages
- Table extraction
- Chart reasoning
- Vision-language models
- Hybrid retrieval + RAG with citations
