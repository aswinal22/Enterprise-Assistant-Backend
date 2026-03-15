import requests
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, EMBEDDING_MODEL


def generate_embedding(text: str):

    url = f"{OPENROUTER_BASE_URL}/embeddings"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }

    response = requests.post(url, headers=headers, json=payload)

    result = response.json()

    return result["data"][0]["embedding"]