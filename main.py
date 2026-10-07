from typing import Any, Dict
import logging

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)


def create_rag_chain(retriever: Any):
    try:
        llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.1)
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the context to answer the question. If the answer is not in the context, say that you cannot find it.
            Always include citations in the format [SOURCE: {source}] when possible.

            Context:
            {context}

            Question: {question}
            Answer:"""
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    except Exception as e:
        logger.error(f"RAG chain creation failed: {e}")
        raise


def answer_with_citations(chain: Any, question: str) -> Dict[str, Any]:
    try:
        result = chain.invoke({"query": question})
        source_docs = result.get("source_documents", [])
        sources = []
        for doc in source_docs:
            meta = doc.metadata
            sources.append({
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "content": doc.page_content[:400]
            })
        return {
            "answer": result.get("result", ""),
            "sources": sources,
            "success": True
        }
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": [],
            "success": False
        }
