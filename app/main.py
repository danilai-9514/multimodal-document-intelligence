from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.config import UPLOAD_PATH
from app.ingest import save_uploaded_file, build_document_records, VectorIndex
from app.answer import answer_question

app = FastAPI(title="Multimodal Document Intelligence")


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def health():
    return {"message": "Multimodal Document Intelligence API is running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = save_uploaded_file(file)
    records = build_document_records(file_path)

    if not records:
        raise HTTPException(status_code=400, detail="No pages were extracted from the PDF.")

    chunks = []
    meta = []
    for rec in records:
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        chunks.append(text)
        meta.append({
            "doc_id": rec["doc_id"],
            "page_no": rec["page_no"],
            "section": "body",
            "source": rec.get("source", "pdf")
        })

    index = VectorIndex()
    if chunks:
        index.add_texts(chunks, meta)
        index.save()

    return {
        "status": "ok",
        "document_id": records[0]["doc_id"],
        "pages": len(records),
        "message": "Document indexed successfully."
    }


@app.post("/ask")
async def ask_question(payload: QuestionRequest):
    q = payload.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    index = VectorIndex()
    index.load()
    results = index.search(q, k=5)

    evidence = []
    for item in results:
        meta = item["metadata"]
        evidence.append(
            f"Document: {meta['doc_id']} | Page: {meta['page_no']} | Section: {meta['section']} | Source: {meta['source']}"
        )

    return answer_question(q, evidence)
