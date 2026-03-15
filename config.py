import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

LLM_MODEL = os.getenv("LLM_MODEL")