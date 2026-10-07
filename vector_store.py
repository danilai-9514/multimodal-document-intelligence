from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def summarize_text(texts: List[str]) -> List[str]:
    if not texts:
        return []
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.1)
    summaries = []
    for text in texts:
        try:
            prompt = PromptTemplate(
                input_variables=["text"],
                template="Summarize this text in 2-3 sentences:\n{text}"
            )
            summary = llm.invoke(prompt.format(text=text)).content
            summaries.append(summary)
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            summaries.append(str(text)[:500])
    return summaries


def summarize_tables(tables: List[str]) -> List[str]:
    if not tables:
        return []
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.1)
    summaries = []
    for table in tables:
        try:
            prompt = PromptTemplate(
                input_variables=["table"],
                template="Summarize the main findings in this table:\n{table}"
            )
            summaries.append(llm.invoke(prompt.format(table=table)).content)
        except Exception as e:
            logger.warning(f"Table summary failed: {e}")
            summaries.append(str(table)[:500])
    return summaries


def summarize_images(images: List[Dict[str, str]]) -> List[str]:
    if not images:
        return []
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.1)
    summaries = []
    for image in images:
        try:
            base64_data = image.get("image_base64", "")
            if not base64_data:
                summaries.append("Image not available")
                continue
            summary = llm.invoke([{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the key information in this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_data}"}}
                ]
            }]).content
            summaries.append(summary)
        except Exception as e:
            logger.warning(f"Image summary failed: {e}")
            summaries.append("Image summary unavailable")
    return summaries


def summarize_elements(elements: List[str]) -> List[str]:
    return [str(e)[:500] for e in elements]
