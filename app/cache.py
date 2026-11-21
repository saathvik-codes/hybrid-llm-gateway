import asyncio
import json
import os
import tempfile
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


def _unit(vec: list[float]) -> np.ndarray | None:
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else None


class SemanticCache:
    """
    Prompt -> answer cache that matches by meaning. Embeddings are kept as one normalised matrix, so a lookup is a
    single matrix-vector product instead of a Python loop. Writes are serialised by a lock and the JSON file is
    replaced atomically, so concurrent requests can't interleave or truncate it.
    """

    def __init__(self, db_path: str = settings.CACHE_DB_PATH, threshold: float = settings.SIMILARITY_THRESHOLD):
        self.db_path = db_path
        self.threshold = threshold
        self.entries: list[CacheEntry] = []
        self._matrix: np.ndarray | None = None     # rows = unit embeddings, aligned with self.entries
        self._rows: list[int] = []                 # index into self.entries for each matrix row
        self._lock = asyncio.Lock()
        self.load_cache()

    # ── index ────────────────────────────────────────────────────────────────
    def _reindex(self) -> None:
        rows, vecs, dim = [], [], None
        for i, e in enumerate(self.entries):
            u = _unit(e.embedding)
            if u is None or (dim is not None and len(u) != dim):
                continue                              # empty / mismatched embeddings can never match
            dim = len(u)
            rows.append(i)
            vecs.append(u)
        self._rows = rows
        self._matrix = np.vstack(vecs) if vecs else None

    # ── storage (the Redis subclass replaces these) ─────────────────────────
    def load_cache(self):
        """Load cache entries from the JSON file, with backward-compatibility for IDs."""
        self.entries = []
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    for entry in json.load(f):
                        if "id" not in entry:          # legacy entries had no id
                            entry["id"] = str(uuid.uuid4())
                        self.entries.append(CacheEntry(**entry))
                logger.info(f"Loaded {len(self.entries)} entries from semantic cache DB.")
            except Exception as e:
                logger.error(f"Failed to load semantic cache DB: {e}. Starting fresh.")
                self.entries = []
        self._reindex()

    def save_cache(self):
        """Write the whole cache to a temp file, then swap it in, so readers never see a half-written file."""
        try:
            dir_name = os.path.dirname(os.path.abspath(self.db_path))
            os.makedirs(dir_name, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=dir_name, prefix=".cache-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump([entry.model_dump() for entry in self.entries], f, indent=2, ensure_ascii=False)
                os.replace(tmp, self.db_path)
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
            logger.info("Saved semantic cache DB.")
        except Exception as e:
            logger.error(f"Failed to save semantic cache DB: {e}")

    def _persist_put(self, entry: CacheEntry) -> None:
        self.save_cache()

    def _persist_delete(self, entry_id: str) -> None:
        self.save_cache()

    def _persist_clear(self) -> None:
        self.save_cache()

    def _needs_refresh(self) -> bool:
        return False

    def _refresh(self) -> None:
        """Hook for shared backends: pick up entries other replicas have written."""

    # ── API ──────────────────────────────────────────────────────────────────
    async def get(self, prompt: str, ollama_service: OllamaService) -> tuple[str | None, float]:
        """
        Check if prompt matches any cached entry above the similarity threshold.
        Returns:
            Tuple of (cached_response, similarity_score)
        """
        if self._needs_refresh():
            await asyncio.to_thread(self._refresh)
        if not self.entries or self._matrix is None:
            return None, 0.0
        try:
            query_emb = await ollama_service.get_embedding(prompt)
            q = _unit(query_emb) if query_emb else None
            if q is None or q.shape[0] != self._matrix.shape[1]:
                return None, 0.0
            sims = self._matrix @ q
            best = int(np.argmax(sims))
            similarity = float(sims[best])
            if similarity >= self.threshold:
                logger.info(f"Semantic Cache Hit! Similarity: {similarity:.4f}")
                return self.entries[self._rows[best]].response, similarity
            logger.info(f"Semantic Cache Miss. Closest similarity: {similarity:.4f}")
            return None, similarity
        except Exception as e:
            logger.error(f"Error checking semantic cache: {e}")
            return None, 0.0

    async def set(self, prompt: str, response: str, ollama_service: OllamaService):
        """Add prompt and response to semantic cache."""
        try:
            embedding = await ollama_service.get_embedding(prompt)
            if not embedding:
                logger.warning("Failed to generate embedding for cache entry.")
                return
            async with self._lock:
                entry = next((e for e in self.entries if e.prompt == prompt), None)
                if entry:                              # same prompt again: refresh the answer
                    entry.response, entry.timestamp = response, time.time()
                else:
                    entry = CacheEntry(id=str(uuid.uuid4()), prompt=prompt, embedding=embedding,
                                       response=response, timestamp=time.time())
                    self.entries.append(entry)
                    self._reindex()
                await asyncio.to_thread(self._persist_put, entry)     # disk / network I/O off the event loop
        except Exception as e:
            logger.error(f"Error writing to semantic cache: {e}")

    def delete(self, entry_id: str) -> bool:
        """Deletes a cache entry by its ID. Returns True if deleted, False otherwise."""
        for idx, entry in enumerate(self.entries):
            if entry.id == entry_id:
                self.entries.pop(idx)
                self._reindex()
                self._persist_delete(entry_id)
                logger.info(f"Deleted cache entry: {entry_id}")
                return True
        logger.warning(f"Cache entry not found for deletion: {entry_id}")
        return False

    def clear(self):
        """Clears all entries from the cache."""
        self.entries = []
        self._reindex()
        self._persist_clear()
        logger.info("Cleared semantic cache.")


class RedisSemanticCache(SemanticCache):
    """
    Same cache, with entries kept in a Redis hash so every gateway replica sees them. Each replica searches its own
    in-memory index and re-reads the hash every CACHE_REFRESH_SECONDS, so an answer cached by one pod is served by
    the others within a few seconds.
    """

    def __init__(self, client=None, prefix: str = settings.REDIS_KEY_PREFIX, threshold: float = settings.SIMILARITY_THRESHOLD,
                 refresh_seconds: float = settings.CACHE_REFRESH_SECONDS):
        if client is None:
            import redis
            client = redis.Redis.from_url(settings.REDIS_URL, socket_timeout=2, socket_connect_timeout=2)
        self.redis, self.key, self.refresh_seconds = client, f"{prefix}:cache", refresh_seconds
        self._last_refresh = 0.0
        super().__init__(db_path="", threshold=threshold)

    def load_cache(self):
        self.entries = []
        try:
            for raw in self.redis.hvals(self.key):
                self.entries.append(CacheEntry(**json.loads(raw)))
            logger.info(f"Loaded {len(self.entries)} entries from Redis.")
        except Exception as e:
            logger.error(f"Failed to load the cache from Redis: {e}")
        self._last_refresh = time.monotonic()
        self._reindex()

    def _needs_refresh(self) -> bool:
        return time.monotonic() - self._last_refresh >= self.refresh_seconds

    def _refresh(self) -> None:
        self.load_cache()

    def save_cache(self):
        """Nothing to do: every change is written to Redis as it happens."""

    def _persist_put(self, entry: CacheEntry) -> None:
        try:
            self.redis.hset(self.key, entry.id, json.dumps(entry.model_dump(), ensure_ascii=False))
        except Exception as e:
            logger.error(f"Failed to write the cache entry to Redis: {e}")

    def _persist_delete(self, entry_id: str) -> None:
        try:
            self.redis.hdel(self.key, entry_id)
        except Exception as e:
            logger.error(f"Failed to delete the cache entry in Redis: {e}")

    def _persist_clear(self) -> None:
        try:
            self.redis.delete(self.key)
        except Exception as e:
            logger.error(f"Failed to clear the cache in Redis: {e}")


class DynamoDBSemanticCache(SemanticCache):
    """
    Same cache, with entries in an Amazon DynamoDB table (partition key `id`), for the serverless deployment where
    every Lambda instance is a separate process. Like the Redis backend, each instance searches its own in-memory
    index and re-reads the table every CACHE_REFRESH_SECONDS; an AWS error degrades to a cache miss, never a failure.
    """

    def __init__(self, table=None, threshold: float = settings.SIMILARITY_THRESHOLD,
                 refresh_seconds: float = settings.CACHE_REFRESH_SECONDS):
        if table is None:
            import boto3
            table = boto3.resource("dynamodb", region_name=settings.AWS_REGION).Table(settings.DYNAMODB_TABLE)
        self.table, self.refresh_seconds = table, refresh_seconds
        self._last_refresh = 0.0
        super().__init__(db_path="", threshold=threshold)

    def load_cache(self):
        self.entries = []
        try:
            kwargs = {}
            while True:                                  # Scan is paginated (1 MB per page)
                page = self.table.scan(**kwargs)
                for item in page.get("Items", []):
                    self.entries.append(CacheEntry(**json.loads(item["entry"])))
                if "LastEvaluatedKey" not in page:
                    break
                kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
            logger.info(f"Loaded {len(self.entries)} entries from DynamoDB.")
        except Exception as e:
            logger.error(f"Failed to load the cache from DynamoDB: {e}")
        self._last_refresh = time.monotonic()
        self._reindex()

    def _needs_refresh(self) -> bool:
        return time.monotonic() - self._last_refresh >= self.refresh_seconds

    def _refresh(self) -> None:
        self.load_cache()

    def save_cache(self):
        """Nothing to do: every change is written to DynamoDB as it happens."""

    def _persist_put(self, entry: CacheEntry) -> None:
        try:
            # the entry is stored as one JSON string: DynamoDB numbers are Decimals, and a 768-float embedding as a
            # string is smaller and faster to read back than a list attribute
            self.table.put_item(Item={"id": entry.id, "entry": json.dumps(entry.model_dump())})
        except Exception as e:
            logger.error(f"Failed to write the cache entry to DynamoDB: {e}")

    def _persist_delete(self, entry_id: str) -> None:
        try:
            self.table.delete_item(Key={"id": entry_id})
        except Exception as e:
            logger.error(f"Failed to delete the cache entry in DynamoDB: {e}")

    def _persist_clear(self) -> None:
        try:
            kwargs = {"ProjectionExpression": "id"}
            with self.table.batch_writer() as batch:
                while True:
                    page = self.table.scan(**kwargs)
                    for item in page.get("Items", []):
                        batch.delete_item(Key={"id": item["id"]})
                    if "LastEvaluatedKey" not in page:
                        break
                    kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
        except Exception as e:
            logger.error(f"Failed to clear the cache in DynamoDB: {e}")


def build_cache() -> SemanticCache:
    """The cache chosen by CACHE_BACKEND ("file", "redis" or "dynamodb")."""
    backend = settings.CACHE_BACKEND.lower()
    if backend == "redis":
        return RedisSemanticCache()
    if backend == "dynamodb":
        return DynamoDBSemanticCache()
    return SemanticCache()
