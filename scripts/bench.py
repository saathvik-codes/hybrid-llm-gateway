"""
Two measurements, run on the developer's machine with stubbed model providers (so the numbers are about the gateway's own
code, not about Ollama or Gemini):

  1. semantic-cache lookup: the original per-entry Python loop vs the matrix search used now
  2. concurrency: N requests in flight against stubbed providers that each take 20 ms; if the event loop were blocked
     the total would be N x 20 ms, if requests overlap it is close to 20 ms x (N / concurrency)

    python scripts/bench.py
"""
import asyncio
import statistics
import sys
import time
import unittest.mock as mock
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cache import SemanticCache, cosine_similarity    # noqa: E402
from app.core import GatewayCore                           # noqa: E402
from app.budget import BudgetPolicy                        # noqa: E402
from app.models import CacheEntry, QueryRequest            # noqa: E402


def bench_cache(entries=5000, dim=768, lookups=50):
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(entries, dim)).astype(np.float32)
    cache = SemanticCache(db_path="./bench_cache.json", threshold=0.99)
    cache.entries = [CacheEntry(id=str(i), prompt=f"p{i}", embedding=v.tolist(), response="r", timestamp=0.0) for i, v in enumerate(vecs)]
    cache._reindex()
    queries = rng.normal(size=(lookups, dim)).tolist()

    def loop_search(q):                      # the original implementation
        best = 0.0
        for e in cache.entries:
            best = max(best, cosine_similarity(q, e.embedding))
        return best

    t = time.perf_counter()
    for q in queries[:10]:
        loop_search(q)
    loop_ms = (time.perf_counter() - t) / 10 * 1000

    ollama = mock.AsyncMock()

    async def run():
        t0 = time.perf_counter()
        for q in queries:
            ollama.get_embedding.return_value = q
            await cache.get("x", ollama)
        return (time.perf_counter() - t0) / lookups * 1000

    matrix_ms = asyncio.run(run())
    print(f"cache lookup over {entries} x {dim}-dim entries: loop {loop_ms:.1f} ms  ->  matrix {matrix_ms:.2f} ms  ({loop_ms / matrix_ms:.0f}x)")


def bench_concurrency(total=400, concurrency=100, delay=0.02):
    ollama, cloud, cache, router = mock.AsyncMock(), mock.AsyncMock(), mock.AsyncMock(), mock.Mock()

    async def slow_generate(**kw):
        await asyncio.sleep(delay)
        return "answer"

    cache.get.return_value = (None, 0.0)
    ollama.is_healthy.return_value = True
    ollama.generate.side_effect = slow_generate
    router.is_complex.return_value = False
    core = GatewayCore(ollama, cloud, cache, router, BudgetPolicy(0))
    gate = asyncio.Semaphore(concurrency)
    latencies = []

    async def one(i):
        async with gate:
            t = time.perf_counter()
            await core.handle(QueryRequest(prompt=f"hello {i}"), lambda fn, *a: None)
            latencies.append((time.perf_counter() - t) * 1000)

    async def run():
        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(total)))
        return time.perf_counter() - t0

    seconds = asyncio.run(run())
    lat = sorted(latencies)
    print(f"{total} requests, {concurrency} in flight, {delay * 1000:.0f} ms stub model: {seconds:.2f} s total "
          f"({total / seconds:.0f} req/s; serial would be {total * delay:.1f} s), "
          f"p50 {statistics.median(lat):.0f} ms, p95 {lat[int(len(lat) * 0.95)]:.0f} ms")


if __name__ == "__main__":
    bench_cache()
    bench_concurrency()
