"""
The gateway's brain, shared by every front end (REST, gRPC, MCP): look in the semantic cache, pick local or cloud,
fall back the other way when a provider fails, apply the daily cost policy, and record metrics.
"""
import logging
import time
from typing import Callable

from app import metrics
from app.budget import BudgetPolicy
from app.cache import SemanticCache, build_cache
from app.config import settings
from app.models import QueryRequest, QueryResponse
from app.router import PromptRouter
from app.services.cloud_service import CloudService
from app.services.ollama_service import OllamaService

logger = logging.getLogger("gateway.core")

# One policy for the process; its limit comes from DAILY_BUDGET_USD
budget = BudgetPolicy(settings.DAILY_BUDGET_USD)

# schedule(fn, *args): run a coroutine function later without making the caller wait (FastAPI BackgroundTasks.add_task,
# or asyncio.create_task for the other front ends)
Scheduler = Callable[..., None]


class GatewayError(Exception):
    """A request that can't be answered. status_code follows HTTP (429 over budget, 502 providers failed)."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code, self.detail = status_code, detail


class GatewayCore:
    def __init__(self, ollama: OllamaService, cloud: CloudService, cache: SemanticCache, router: PromptRouter,
                 policy: BudgetPolicy | None = None):
        self.ollama, self.cloud, self.cache, self.router = ollama, cloud, cache, router
        self.policy = policy or budget

    async def handle(self, req: QueryRequest, schedule: Scheduler) -> QueryResponse:
        start = time.perf_counter()
        prompt = req.prompt

        # 1. semantic cache (skipped when the caller insists on the cloud model)
        if not req.force_cloud:
            hit, similarity = await self.cache.get(prompt, self.ollama)
            if hit is not None:
                metrics.CACHE_LOOKUPS.labels("hit").inc()
                return self._done(start, hit, f"cache (similarity: {similarity:.4f})")
            metrics.CACHE_LOOKUPS.labels("miss").inc()

        # 2. route: complex prompts go to the cloud model, the rest to the local one, unless the budget says otherwise
        wants_cloud = req.force_cloud or self.router.is_complex(prompt)
        if wants_cloud and self.policy.exceeded():
            if req.force_cloud:
                metrics.BUDGET_REJECTIONS.labels("rejected").inc()
                raise GatewayError(429, f"Daily cloud budget of ${self.policy.daily_limit_usd:.2f} is used up.")
            metrics.BUDGET_REJECTIONS.labels("degraded_to_local").inc()
            logger.warning("Daily cloud budget used up: answering a complex prompt locally.")
            text = await self._local(req, "local_slm_budget")
            return self._finish(start, text, "local_slm_budget", schedule, prompt)

        if not wants_cloud:
            if await self.ollama.is_healthy():
                try:
                    text = await self.ollama.generate(prompt=prompt, temperature=req.temperature, max_tokens=req.max_tokens)
                    return self._finish(start, text, "local_slm", schedule, prompt)
                except Exception as e:
                    metrics.PROVIDER_ERRORS.labels("ollama").inc()
                    logger.error("Local SLM failed (%s): falling back to the cloud model.", type(e).__name__)
            else:
                logger.warning("Local Ollama is unreachable: falling back to the cloud model.")
            cloud_source = "cloud_llm_fallback"
        else:
            cloud_source = "cloud_llm_forced" if req.force_cloud else "cloud_llm"

        # 3. cloud model, with a local fallback when it fails (unless the caller forced the cloud)
        try:
            text, tokens, cost = await self.cloud.generate(prompt=prompt, temperature=req.temperature, max_tokens=req.max_tokens)
        except Exception as e:
            metrics.PROVIDER_ERRORS.labels("gemini").inc()
            logger.error("Cloud model failed (%s).", type(e).__name__)
            if req.force_cloud or not await self.ollama.is_healthy():
                raise GatewayError(502, f"Cloud model failed and no fallback was possible: {type(e).__name__}") from e
            try:
                text = await self.ollama.generate(prompt=prompt, temperature=req.temperature, max_tokens=req.max_tokens)
            except Exception as local_err:
                metrics.PROVIDER_ERRORS.labels("ollama").inc()
                raise GatewayError(502, f"Both cloud and local models failed: {type(local_err).__name__}") from local_err
            return self._finish(start, text, "local_slm_fallback", schedule, prompt)

        self.policy.record(cost)
        metrics.TOKENS.inc(tokens)
        metrics.COST.inc(cost)
        metrics.BUDGET_LEFT.set(self.policy.remaining())
        return self._finish(start, text, cloud_source, schedule, prompt, tokens, cost)

    async def _local(self, req: QueryRequest, source: str) -> str:
        try:
            return await self.ollama.generate(prompt=req.prompt, temperature=req.temperature, max_tokens=req.max_tokens)
        except Exception as e:
            metrics.PROVIDER_ERRORS.labels("ollama").inc()
            raise GatewayError(502, f"Cloud budget is used up and the local model failed: {type(e).__name__}") from e

    def _finish(self, start, text, source, schedule, prompt, tokens=0, cost=0.0) -> QueryResponse:
        if text:
            schedule(self.cache.set, prompt, text, self.ollama)     # store the answer without making the caller wait
        return self._done(start, text, source, tokens, cost)

    @staticmethod
    def _done(start, text, source, tokens=0, cost=0.0) -> QueryResponse:
        seconds = time.perf_counter() - start
        label = "cache" if source.startswith("cache") else source
        metrics.REQUESTS.labels(label).inc()
        metrics.LATENCY.labels(label).observe(seconds)
        return QueryResponse(response=text, source=source, latency_ms=round(seconds * 1000.0, 2), tokens_used=tokens, cost_usd=cost)

    async def health(self) -> dict:
        local = await self.ollama.is_healthy()
        cloud = bool(settings.GEMINI_API_KEY)
        return {"status": "healthy" if local or cloud else "degraded", "local_ollama_connected": local,
                "cloud_gemini_configured": cloud, "debug_mode": settings.DEBUG}

    async def close(self) -> None:
        await self.ollama.close()
        await self.cloud.close()


def build_core() -> GatewayCore:
    """Real services from settings (used by the server and by the MCP tool)."""
    return GatewayCore(OllamaService(), CloudService(), build_cache(), PromptRouter())
