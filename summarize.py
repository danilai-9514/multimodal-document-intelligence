from unstructured.partition.pdf import partition_pdf
from typing import List, Any, Dict
import logging

logger = logging.getLogger(__name__)


def extract_chunks(file_path: str):
    try:
        return partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000,
        )
    except Exception as e:
        logger.error(f"Error extracting chunks: {e}")
        return []


def get_images_base64(chunks):
    images = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            for elem in getattr(chunk.metadata, "orig_elements", []):
                if "Image" in str(type(elem)):
                    images.append({
                        "image_base64": getattr(elem.metadata, "image_base64", ""),
                        "source": "image"
                    })
    return images


def categorize_chunks(chunks: List[Any]) -> Dict[str, List[Any]]:
    result = {"text": [], "tables": [], "images": []}
    for chunk in chunks:
        t = str(type(chunk))
        if "Table" in t:
            result["tables"].append(chunk)
        elif "Image" in t:
            result["images"].append(chunk)
        else:
            result["text"].append(chunk)
    return result
