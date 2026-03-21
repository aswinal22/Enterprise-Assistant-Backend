import requests
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL
import traceback

def _send_chat_completion(messages: list[dict[str, str]]) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")
    if not OPENROUTER_BASE_URL:
        raise RuntimeError("OPENROUTER_BASE_URL is not configured.")
    if not LLM_MODEL:
        raise RuntimeError("LLM_MODEL is not configured.")

    url = f"{OPENROUTER_BASE_URL}/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.2
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        details = ""
        if getattr(exc, "response", None) is not None:
            details = f" Response body: {exc.response.text[:500]}"
        raise RuntimeError(f"LLM request failed: {exc}.{details}") from exc

    result = response.json()
    choices = result.get("choices")
    if not choices:
        raise RuntimeError(f"LLM response did not include choices. Response: {str(result)[:500]}")

    message = choices[0].get("message", {})
    content = message.get("content")
    if not content:
        raise RuntimeError(f"LLM response did not include message content. Response: {str(result)[:500]}")

    return content


def generate_answer(question: str, context: str) -> str:
    system_prompt = """
You are a precise enterprise document assistant.

Answer the user's query using ONLY the provided context. 

Guidelines:
- CLEAN & CONCISE: Do not repeat information. If the context contains overlapping text from multiple chunks, merge them into a single coherent explanation.
- PARAPHRASE: Avoid long, raw passages. Translate legal or technical jargon into clear, professional language.
- GROUNDED: If the answer is not in the context, explicitly state: "I cannot find this information in the provided document.
"""

    user_prompt = f"""
Question:
{question}

Context:
{context}
"""

    return _send_chat_completion(
        [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]
    )


def generate_general_answer(question: str) -> str:
    system_prompt = """
You are a helpful enterprise assistant.

The available documents do not contain a strong enough match for the user's question.
Answer the question generally using your own knowledge.
Do not claim that the answer came from the documents.
If the question is domain-specific and uncertain, be honest about uncertainty.

Response format:
1. Start with a direct answer in 2-4 sentences.
2. Then add a short section titled "Key points:" with concise bullet points.
3. Add a short "Notes:" section if you are making a general answer instead of a document-grounded one.
"""

    user_prompt = f"""
Question:
{question}
"""

    return _send_chat_completion(
        [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]
    )


def generate_summary(text: str) -> str:
    system_prompt = """
You are an expert document summarizer.

Summarize the provided document text in a concise, professional manner.
Focus on key points, main ideas, and important details.
Keep the summary to 200-300 words.
Structure it with a brief overview followed by bullet points for main sections or topics.
"""

    user_prompt = f"""
Document text:
{text[:4000]}  # Limit to first 4000 chars to avoid token limits
"""

    return _send_chat_completion(
        [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]
    )
