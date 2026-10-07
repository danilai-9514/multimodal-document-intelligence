from openai import OpenAI

from app.config import OPENAI_API_KEY, LLM_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def answer_question(question: str, evidence: list[str]) -> dict:
    if not OPENAI_API_KEY:
        return {
            "answer": "OpenAI API key is missing. Add OPENAI_API_KEY in your .env file.",
            "citations": evidence,
            "status": "error"
        }

    context = "\n\n---\n\n".join(evidence)
    prompt = f"""
You are a careful document intelligence assistant.

Use only the provided evidence.
Every answer must include citations in this format: [Document: <doc_id>, Page: <page_no>, Section: <section>]
If the answer cannot be supported by the evidence, say so clearly.

Question:
{question}

Evidence:
{context}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a careful multimodal document Q&A assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content or "No answer generated."
    return {"answer": answer, "citations": evidence, "status": "ok"}
