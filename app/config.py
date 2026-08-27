import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PORT: int = 8000
    HOST: str = "127.0.0.1"
    DEBUG: bool = True

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"

    # Cloud LLM
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # Gateway Logic
    SIMILARITY_THRESHOLD: float = 0.95
    COMPLEXITY_THRESHOLD_CHARS: int = 150
    CACHE_DB_PATH: str = "./semantic_cache.json"

    # Load from .env file
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
