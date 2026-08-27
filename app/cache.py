import json
import os
import time
import uuid
import numpy as np
import logging
from app.config import settings
from app.models import CacheEntry
from app.services.ollama_service import OllamaService

logger = logging.getLogger(__name__)

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

class SemanticCache:
    def __init__(self, db_path: str = settings.CACHE_DB_PATH, threshold: float = settings.SIMILARITY_THRESHOLD):
        self.db_path = db_path
        self.threshold = threshold
        self.entries: list[CacheEntry] = []
        self.load_cache()

    def load_cache(self):
        """Load cache entries from JSON file, with backward-compatibility for IDs."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.entries = []
                    for entry in data:
                        # Auto-assign UUID if not present in legacy entries
                        if "id" not in entry:
                            entry["id"] = str(uuid.uuid4())
                        self.entries.append(CacheEntry(**entry))
                logger.info(f"Loaded {len(self.entries)} entries from semantic cache DB.")
            except Exception as e:
                logger.error(f"Failed to load semantic cache DB: {e}. Starting fresh.")
                self.entries = []
        else:
            self.entries = []

    def save_cache(self):
        """Save cache entries to JSON file."""
        try:
            # Ensure parent directories exist
            dir_name = os.path.dirname(os.path.abspath(self.db_path))
            os.makedirs(dir_name, exist_ok=True)
            
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump([entry.model_dump() for entry in self.entries], f, indent=2, ensure_ascii=False)
            logger.info("Saved semantic cache DB.")
        except Exception as e:
            logger.error(f"Failed to save semantic cache DB: {e}")

    async def get(self, prompt: str, ollama_service: OllamaService) -> tuple[str | None, float]:
        """
        Check if prompt matches any cached entry above the similarity threshold.
        Returns:
            Tuple of (cached_response, similarity_score)
        """
        if not self.entries:
            return None, 0.0

        try:
            # Get embedding of incoming prompt
            query_emb = await ollama_service.get_embedding(prompt)
            if not query_emb:
                return None, 0.0
            
            best_match = None
            highest_similarity = 0.0

            # Find closest match
            for entry in self.entries:
                similarity = cosine_similarity(query_emb, entry.embedding)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = entry

            if highest_similarity >= self.threshold and best_match:
                logger.info(f"Semantic Cache Hit! Similarity: {highest_similarity:.4f}")
                return best_match.response, highest_similarity
            
            logger.info(f"Semantic Cache Miss. Closest similarity: {highest_similarity:.4f}")
            return None, highest_similarity
        except Exception as e:
            logger.error(f"Error checking semantic cache: {e}")
            return None, 0.0

    async def set(self, prompt: str, response: str, ollama_service: OllamaService):
        """Add prompt and response to semantic cache."""
        try:
            # Get embedding of prompt
            embedding = await ollama_service.get_embedding(prompt)
            if not embedding:
                logger.warning("Failed to generate embedding for cache entry.")
                return

            # Check if prompt already exists to avoid duplicates
            for entry in self.entries:
                if entry.prompt == prompt:
                    entry.response = response
                    entry.timestamp = time.time()
                    self.save_cache()
                    return

            new_entry = CacheEntry(
                id=str(uuid.uuid4()),
                prompt=prompt,
                embedding=embedding,
                response=response,
                timestamp=time.time()
            )
            self.entries.append(new_entry)
            self.save_cache()
        except Exception as e:
            logger.error(f"Error writing to semantic cache: {e}")

    def delete(self, entry_id: str) -> bool:
        """Deletes a cache entry by its ID. Returns True if deleted, False otherwise."""
        for idx, entry in enumerate(self.entries):
            if entry.id == entry_id:
                self.entries.pop(idx)
                self.save_cache()
                logger.info(f"Deleted cache entry: {entry_id}")
                return True
        logger.warning(f"Cache entry not found for deletion: {entry_id}")
        return False

    def clear(self):
        """Clears all entries from the cache."""
        self.entries = []
        self.save_cache()
        logger.info("Cleared semantic cache.")
