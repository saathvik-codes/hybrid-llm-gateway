import asyncio
import hashlib
import json
import unittest.mock as mock

import numpy as np
import pytest

from app.cache import SemanticCache
from app.models import CacheEntry


def unit(rng, dim=64):
    v = rng.normal(size=dim)
    return (v / np.linalg.norm(v)).tolist()


@pytest.mark.asyncio
async def test_vector_search_finds_the_nearest_entry_among_many(tmp_path):
    rng = np.random.default_rng(7)
    cache = SemanticCache(db_path=str(tmp_path / "c.json"), threshold=0.9)
    vecs = [unit(rng) for _ in range(300)]
    cache.entries = [CacheEntry(id=str(i), prompt=f"p{i}", embedding=v, response=f"r{i}", timestamp=0.0) for i, v in enumerate(vecs)]
    cache._reindex()
    ollama = mock.AsyncMock()
    noisy = (np.array(vecs[123]) + rng.normal(scale=0.01, size=64)).tolist()
    ollama.get_embedding.return_value = noisy
    response, sim = await cache.get("anything", ollama)
    assert response == "r123" and sim > 0.99


@pytest.mark.asyncio
async def test_unrelated_query_is_a_miss(tmp_path):
    rng = np.random.default_rng(1)
    cache = SemanticCache(db_path=str(tmp_path / "c.json"), threshold=0.95)
    cache.entries = [CacheEntry(id="a", prompt="p", embedding=unit(rng), response="r", timestamp=0.0)]
    cache._reindex()
    ollama = mock.AsyncMock()
    ollama.get_embedding.return_value = unit(rng)
    assert (await cache.get("x", ollama))[0] is None


@pytest.mark.asyncio
async def test_query_with_different_dimension_is_a_miss_not_a_crash(tmp_path):
    rng = np.random.default_rng(2)
    cache = SemanticCache(db_path=str(tmp_path / "c.json"), threshold=0.5)
    cache.entries = [CacheEntry(id="a", prompt="p", embedding=unit(rng, 64), response="r", timestamp=0.0)]
    cache._reindex()
    ollama = mock.AsyncMock()
    ollama.get_embedding.return_value = unit(rng, 32)
    assert await cache.get("x", ollama) == (None, 0.0)


@pytest.mark.asyncio
async def test_concurrent_writes_leave_a_complete_valid_file(tmp_path):
    path = tmp_path / "c.json"
    cache = SemanticCache(db_path=str(path), threshold=0.95)
    ollama = mock.AsyncMock()

    def embed(text):
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        return unit(np.random.default_rng(seed))

    ollama.get_embedding.side_effect = embed
    await asyncio.gather(*(cache.set(f"prompt {i}", f"answer {i}", ollama) for i in range(40)))
    data = json.loads(path.read_text(encoding="utf-8"))          # would raise if two writers had interleaved
    assert len(data) == 40 and len(SemanticCache(db_path=str(path)).entries) == 40
    assert not list(tmp_path.glob(".cache-*.tmp"))               # no temp files left behind


@pytest.mark.asyncio
async def test_same_prompt_updates_instead_of_duplicating(tmp_path):
    cache = SemanticCache(db_path=str(tmp_path / "c.json"))
    ollama = mock.AsyncMock()
    ollama.get_embedding.return_value = [1.0, 0.0]
    await cache.set("same", "first", ollama)
    await cache.set("same", "second", ollama)
    assert len(cache.entries) == 1 and cache.entries[0].response == "second"


def test_delete_and_clear_keep_the_index_in_step(tmp_path):
    cache = SemanticCache(db_path=str(tmp_path / "c.json"))
    cache.entries = [CacheEntry(id=str(i), prompt=f"p{i}", embedding=[1.0, float(i)], response="r", timestamp=0.0) for i in range(3)]
    cache._reindex()
    assert cache._matrix.shape == (3, 2)
    assert cache.delete("1") and cache._matrix.shape == (2, 2)
    cache.clear()
    assert cache._matrix is None and cache.entries == []
