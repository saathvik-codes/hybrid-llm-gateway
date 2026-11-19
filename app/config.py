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

    # Cache storage: "file" (JSON on disk, default) or "redis" (shared by every replica)
    CACHE_BACKEND: str = "file"
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_KEY_PREFIX: str = "gateway"
    CACHE_REFRESH_SECONDS: float = 5.0     # redis only: how often a replica re-reads entries written by the others

    # Cost policy: once cloud spend for the (UTC) day reaches this, complex prompts are answered locally and
    # force_cloud requests are refused. 0 turns the policy off.
    DAILY_BUDGET_USD: float = 0.0

    # gRPC front end next to the REST API (0 turns it off)
    GRPC_PORT: int = 50051

    # Load from .env file
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
