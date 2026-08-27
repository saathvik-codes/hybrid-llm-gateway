import httpx
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class OllamaService:
    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    async def is_healthy(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = await self.client.get("/")
            return response.status_code == 200
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        model: str = settings.OLLAMA_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Call Ollama text generation endpoint."""
        url = "/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    async def get_embedding(self, text: str, model: str = settings.OLLAMA_EMBED_MODEL) -> list[float]:
        """Fetch embedding vector for a prompt text using Ollama."""
        try:
            url = "/api/embeddings"
            payload = {
                "model": model,
                "prompt": text
            }
            response = await self.client.post(url, json=payload)
            data = response.json()
            
            # Check if embedding returned. If not, try /api/embed
            if response.status_code == 404 or "embedding" not in data:
                url = "/api/embed"
                payload = {
                    "model": model,
                    "input": text
                }
                response = await self.client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                embeddings = data.get("embeddings", [])
                if embeddings:
                    return embeddings[0]
                elif "embedding" in data:
                    return data["embedding"]
            else:
                response.raise_for_status()
                return data.get("embedding", [])
        except Exception as e:
            logger.error(f"Error fetching embedding from Ollama: {e}")
            raise e

    async def close(self):
        await self.client.aclose()
