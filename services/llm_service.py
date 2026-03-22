import requests
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL
from logging_utils import get_logger

logger = get_logger(__name__)

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

    logger.info(
        "Sending LLM chat completion model=%s message_count=%s",
        LLM_MODEL,
        len(messages),
    )

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        details = ""
        if getattr(exc, "response", None) is not None:
            details = f" Response body: {exc.response.text[:500]}"
        logger.exception("LLM request failed model=%s", LLM_MODEL)
        raise RuntimeError(f"LLM request failed: {exc}.{details}") from exc

    result = response.json()
    choices = result.get("choices")
    if not choices:
        logger.error("LLM response missing choices model=%s", LLM_MODEL)
        raise RuntimeError(f"LLM response did not include choices. Response: {str(result)[:500]}")

    message = choices[0].get("message", {})
    content = message.get("content")
    if not content:
        logger.error("LLM response missing message content model=%s", LLM_MODEL)
        raise RuntimeError(f"LLM response did not include message content. Response: {str(result)[:500]}")

    logger.info("LLM chat completion received successfully model=%s", LLM_MODEL)
    return content


def generate_answer(question: str, context: str) -> str:
    logger.info(
        "Generating contextual answer question_length=%s context_length=%s",
        len(question),
        len(context),
    )
    system_prompt = """
You are a precise enterprise document assistant specializing in providing accurate answers based on company-specific documentation.

ROLE & APPROACH:
- Answer questions using ONLY the provided document context
- Maintain strict fidelity to the source material
- Act as a knowledgeable interpreter of company policies, procedures, and information

CONTENT HANDLING:
- CLEAN & CONCISE: Eliminate redundancy and repetition across document chunks
- MERGE OVERLAPS: Combine related information from multiple chunks into coherent explanations
- PARAPHRASE: Transform complex legal, technical, or business jargon into clear, professional language
- PRESERVE ACCURACY: Never alter facts, requirements, or specific details from the source

RESPONSE GUIDELINES:
- STRUCTURE: Organize answers logically with clear sections when appropriate
- CITATION: Reference specific document sections or policies when relevant
- CLARITY: Use bullet points or numbered lists for multi-step processes or requirements
- COMPLETENESS: Provide comprehensive answers that fully address the query

ERROR HANDLING:
- If information is not in the provided context, explicitly state: "I cannot find this information in the provided documents."
- Do not make assumptions or provide external knowledge
- For partial information, clearly indicate what is and isn't covered

PROFESSIONAL TONE:
- Maintain business-appropriate, formal language
- Be direct and authoritative while remaining helpful
- Use inclusive language and avoid discriminatory content
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
    logger.info("Generating general answer question_length=%s", len(question))
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
    logger.info("Generating summary text_length=%s truncated_length=%s", len(text), len(text[:4000]))
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
