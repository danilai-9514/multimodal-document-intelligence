import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader
import pdfplumber
from PIL import Image
import pytesseract

from app.config import UPLOAD_PATH


def save_uploaded_file(file_obj) -> str:
    filename = file_obj.filename or f"document_{uuid.uuid4().hex}.pdf"
    save_path = Path(UPLOAD_PATH) / filename
    with open(save_path, "wb") as f:
        f.write(file_obj.file.read())
    return str(save_path)


def extract_pdf_text(pdf_path: str) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    try:
        reader = PdfReader(pdf_path)
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page_no": idx, "text": text, "source": "pdf"})
    except Exception as exc:
        print(f"PDF extraction failed: {exc}")

    if not pages or all((p.get("text") or "").strip() == "" for p in pages):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for idx, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    pages.append({"page_no": idx, "text": text, "source": "pdfplumber"})
        except Exception as exc:
            print(f"Fallback extraction failed: {exc}")

    return pages


def ocr_page_image(image_path: str) -> str:
    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image)
    except Exception as exc:
        print(f"OCR failed for {image_path}: {exc}")
        return ""


def build_document_records(pdf_path: str) -> List[Dict[str, Any]]:
    doc_id = str(uuid.uuid4())
    records: List[Dict[str, Any]] = []
    for page in extract_pdf_text(pdf_path):
        text = (page.get("text") or "").strip()
        records.append({
            "doc_id": doc_id,
            "page_no": page["page_no"],
            "text": text,
            "source": page.get("source", "pdf"),
            "tables": [],
            "charts": [],
            "metadata": {}
        })
    return records
