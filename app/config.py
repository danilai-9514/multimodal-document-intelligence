import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", str(BASE_DIR / "content")))
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", str(BASE_DIR / "vector_store")))
UPLOAD_PATH = Path(os.getenv("UPLOAD_PATH", str(BASE_DIR / "uploads")))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")

for path in [OUTPUT_PATH, VECTOR_STORE_PATH, UPLOAD_PATH]:
    path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("LANGCHAIN_TRACING_V2", LANGCHAIN_TRACING_V2)
