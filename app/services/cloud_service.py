import httpx
import asyncio
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class CloudService:
    def __init__(self, api_key: str = settings.GEMINI_API_KEY):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    async def generate(
        self,
        prompt: str,
        model: str = settings.GEMINI_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        max_retries: int = 2
    ) -> tuple[str, int, float]:
        """
        Call Gemini API to generate content with transient error retries.
        Returns:
            Tuple of (response_text, token_count, cost_usd)
        """
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured in environment variables.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }

        retries = 0
        backoff_sec = 2.0

        while True:
            try:
                response = await self.client.post(url, json=payload)
                
                # Check for rate limits or server errors
                if response.status_code == 429:
                    logger.warning(f"Gemini API rate limited (429). Retry {retries + 1}/{max_retries}")
                    if retries >= max_retries:
                        response.raise_for_status()
                    await asyncio.sleep(backoff_sec)
                    retries += 1
                    backoff_sec *= 2
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                # Extract generated text
                candidates = data.get("candidates", [])
                if not candidates:
                    raise ValueError("No candidates returned from Gemini API response.")
                
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise ValueError("Empty response parts from Gemini API candidate.")
                
                response_text = parts[0].get("text", "")
                
                # Token usage metadata
                usage = data.get("usageMetadata", {})
                total_tokens = usage.get("totalTokenCount", 0)
                prompt_tokens = usage.get("promptTokenCount", 0)
                candidates_tokens = usage.get("candidatesTokenCount", 0)
                
                # Cost Calculation (Gemini 1.5 Flash standard pricing:
                # Input: $0.075 / 1M tokens ($0.000000075 / token)
                # Output: $0.30 / 1M tokens ($0.0000003 / token)
                cost_usd = (prompt_tokens * 0.000000075) + (candidates_tokens * 0.0000003)
                
                return response_text, total_tokens, cost_usd
                
            except (httpx.HTTPError, ValueError) as e:
                logger.error(f"Gemini API error during generation (attempt {retries + 1}/{max_retries + 1}): {e}")
                if retries >= max_retries:
                    raise e
                await asyncio.sleep(backoff_sec)
                retries += 1
                backoff_sec *= 2

    async def close(self):
        await self.client.aclose()
