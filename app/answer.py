from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def answer_question(question: str, evidence: list[str]) -> dict:
    context = "\n\n---\n\n".join(evidence)
    prompt = f"""
You are a multimodal document QA assistant.

Use only the provided evidence. If information is missing, say so clearly.
Every answer must include citations in this format: [Document: <doc_id>, Page: <page_no>, Section: <section>]

Question:
{question}

Evidence:
{context}
"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a careful document intelligence assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content or "No answer generated."
    return {"answer": answer, "citations": evidence}
