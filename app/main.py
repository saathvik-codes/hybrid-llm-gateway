import os
import logging
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Response
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

from app import metrics
from app.config import settings
from app.core import GatewayCore, GatewayError, budget, build_core
from app.models import QueryRequest, QueryResponse
from app.services.ollama_service import OllamaService
from app.services.cloud_service import CloudService
from app.cache import SemanticCache
from app.router import PromptRouter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("gateway")

# Services & Cache Lifespan Manager
core: GatewayCore | None = None
ollama_service = None
cloud_service = None
semantic_cache = None
prompt_router = None
grpc_server = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global core, ollama_service, cloud_service, semantic_cache, prompt_router, grpc_server
    logger.info("Initializing services...")
    core = build_core()
    ollama_service, cloud_service, semantic_cache, prompt_router = core.ollama, core.cloud, core.cache, core.router
    metrics.BUDGET_LEFT.set(budget.remaining())

    if settings.GRPC_PORT:
        try:
            from app.grpc_server import start_grpc
            grpc_server = await start_grpc(core, settings.GRPC_PORT)
            logger.info("gRPC server listening on :%s", settings.GRPC_PORT)
        except ImportError:
            logger.warning("gRPC stubs not generated (run: python scripts/gen_proto.py); REST only.")

    yield

    logger.info("Closing connections...")
    if grpc_server:
        await grpc_server.stop(grace=5)
    await core.close()

app = FastAPI(
    title="Hybrid LLM Router & Cache Gateway",
    description="Production-grade API Gateway with Semantic Caching and Local-to-Cloud Routing",
    version="1.1.0",
    lifespan=lifespan
)

# Dependency Injection helpers
def get_ollama_service() -> OllamaService:
    return ollama_service

def get_cloud_service() -> CloudService:
    return cloud_service

def get_cache() -> SemanticCache:
    return semantic_cache

def get_router() -> PromptRouter:
    return prompt_router

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the Tailwind UI dashboard on the root path."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Dashboard template not found.")

    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/query", response_model=QueryResponse)
async def query_gateway(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    ollama: OllamaService = Depends(get_ollama_service),
    cloud: CloudService = Depends(get_cloud_service),
    cache: SemanticCache = Depends(get_cache),
    router: PromptRouter = Depends(get_router)
):
    try:
        return await GatewayCore(ollama, cloud, cache, router).handle(request, background_tasks.add_task)
    except GatewayError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

# Admin Cache Endpoints
@app.get("/api/cache")
async def get_cache_entries(cache: SemanticCache = Depends(get_cache)):
    """Returns all entries in the cache database (without heavy embedding vectors to optimize payload size)."""
    return [
        {
            "id": entry.id,
            "prompt": entry.prompt,
            "response": entry.response,
            "timestamp": entry.timestamp
        }
        for entry in cache.entries
    ]

@app.delete("/api/cache/{entry_id}")
async def delete_cache_entry(entry_id: str, cache: SemanticCache = Depends(get_cache)):
    """Deletes a specific cache entry by ID."""
    success = cache.delete(entry_id)
    if not success:
        raise HTTPException(status_code=404, detail="Cache entry not found")
    return {"message": "Cache entry deleted successfully"}

@app.post("/cache/clear")
async def clear_cache(cache: SemanticCache = Depends(get_cache)):
    cache.clear()
    return {"message": "Semantic cache cleared successfully"}

@app.get("/cache/stats")
async def cache_stats(cache: SemanticCache = Depends(get_cache)):
    return {
        "cache_entries_count": len(cache.entries),
        "cache_backend": settings.CACHE_BACKEND,
        "cache_db_path": settings.CACHE_DB_PATH,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD
    }

@app.get("/budget")
async def budget_status():
    """Today's cloud spend against the daily cap (remaining is -1 when no cap is set)."""
    return {"daily_limit_usd": budget.daily_limit_usd, "spent_today_usd": round(budget.spent_today, 6),
            "remaining_usd": budget.remaining(), "exceeded": budget.exceeded()}

@app.get("/metrics")
async def prometheus_metrics():
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)

@app.get("/health")
async def health_check(ollama: OllamaService = Depends(get_ollama_service)):
    ollama_healthy = await ollama.is_healthy()
    gemini_configured = bool(settings.GEMINI_API_KEY)

    return {
        "status": "healthy" if ollama_healthy or gemini_configured else "degraded",
        "local_ollama_connected": ollama_healthy,
        "cloud_gemini_configured": gemini_configured,
        "debug_mode": settings.DEBUG
    }
