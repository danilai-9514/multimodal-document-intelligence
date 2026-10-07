from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from typing import Any, List
import logging

from config import VECTOR_STORE_PATH, OPENAI_API_KEY

logger = logging.getLogger(__name__)


def setup_retriever():
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Chroma(
            collection_name="multimodal-documents",
            embedding_function=embeddings,
            persist_directory=str(VECTOR_STORE_PATH)
        )
        return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        logger.error(f"Retriever setup failed: {e}")
        return None


def load_into_retriever(retriever: Any, contents: List[Any], summaries: List[str], content_type: str = "text"):
    try:
        if retriever is None:
            return False
        docs = []
        for i, (content, summary) in enumerate(zip(contents, summaries)):
            docs.append(Document(
                page_content=str(summary),
                metadata={
                    "source": content_type,
                    "chunk_index": i,
                    "original_content": str(content)[:2000]
                }
            ))
        if hasattr(retriever, "vectorstore") and hasattr(retriever.vectorstore, "add_documents"):
            retriever.vectorstore.add_documents(docs)
            return True
        return False
    except Exception as e:
        logger.error(f"Load into retriever failed: {e}")
        return False
