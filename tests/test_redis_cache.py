import unittest.mock as mock

import fakeredis
import pytest

from app.cache import RedisSemanticCache


def ollama_with(vec):
    o = mock.AsyncMock()
    o.get_embedding.return_value = vec
    return o


def replica(server, **kw):
    return RedisSemanticCache(client=fakeredis.FakeRedis(server=server), threshold=0.9, **kw)


@pytest.mark.asyncio
async def test_an_answer_cached_by_one_replica_is_served_by_another():
    server = fakeredis.FakeServer()
    a, b = replica(server, refresh_seconds=0), replica(server, refresh_seconds=0)
    await a.set("what is raft", "a consensus algorithm", ollama_with([1.0, 0.0]))
    response, sim = await b.get("explain raft", ollama_with([0.99, 0.05]))
    assert response == "a consensus algorithm" and sim > 0.99


@pytest.mark.asyncio
async def test_new_replica_loads_existing_entries_at_startup():
    server = fakeredis.FakeServer()
    await replica(server).set("p", "r", ollama_with([0.0, 1.0]))
    assert len(replica(server).entries) == 1


@pytest.mark.asyncio
async def test_delete_and_clear_reach_the_other_replica():
    server = fakeredis.FakeServer()
    a, b = replica(server, refresh_seconds=0), replica(server, refresh_seconds=0)
    await a.set("one", "r1", ollama_with([1.0, 0.0]))
    await a.set("two", "r2", ollama_with([0.0, 1.0]))
    await b.get("x", ollama_with([1.0, 0.0]))                # b refreshes and now sees both
    assert len(b.entries) == 2
    assert a.delete(a.entries[0].id)
    await b.get("x", ollama_with([1.0, 0.0]))
    assert len(b.entries) == 1
    a.clear()
    await b.get("x", ollama_with([1.0, 0.0]))
    assert b.entries == []


@pytest.mark.asyncio
async def test_redis_outage_degrades_to_a_cache_miss_not_an_error():
    broken = mock.Mock()
    broken.hvals.side_effect = ConnectionError("down")
    broken.hset.side_effect = ConnectionError("down")
    cache = RedisSemanticCache(client=broken, refresh_seconds=3600)
    await cache.set("p", "r", ollama_with([1.0, 0.0]))          # must not raise
    assert (await cache.get("p", ollama_with([1.0, 0.0])))[0] == "r"      # still served from this replica's memory
