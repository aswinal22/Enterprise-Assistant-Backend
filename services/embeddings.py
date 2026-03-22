import requests
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, EMBEDDING_MODEL
from logging_utils import get_logger

logger = get_logger(__name__)


def generate_embedding(text: str):
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")
    if not OPENROUTER_BASE_URL:
        raise RuntimeError("OPENROUTER_BASE_URL is not configured.")
    if not EMBEDDING_MODEL:
        raise RuntimeError("EMBEDDING_MODEL is not configured.")

    url = f"{OPENROUTER_BASE_URL}/embeddings"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }

    logger.info(
        "Generating embedding model=%s input_length=%s",
        EMBEDDING_MODEL,
        len(text),
    )

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        details = ""
        if getattr(exc, "response", None) is not None:
            details = f" Response body: {exc.response.text[:500]}"
        logger.exception("Embedding request failed model=%s", EMBEDDING_MODEL)
        raise RuntimeError(f"Embedding request failed: {exc}.{details}") from exc

    result = response.json()

    data = result.get("data")
    if not data:
        logger.error("Embedding response missing data model=%s", EMBEDDING_MODEL)
        raise RuntimeError(f"Embedding response did not include data. Response: {str(result)[:500]}")

    embedding = data[0].get("embedding")
    if embedding is None:
        logger.error("Embedding response missing embedding vector model=%s", EMBEDDING_MODEL)
        raise RuntimeError(f"Embedding response did not include an embedding vector. Response: {str(result)[:500]}")

    logger.info("Embedding generated successfully vector_length=%s", len(embedding))
    return embedding
