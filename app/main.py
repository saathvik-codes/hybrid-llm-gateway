import time
import os
import logging
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

from app.config import settings
from app.models import QueryRequest, QueryResponse
from app.services.ollama_service import OllamaService
from app.services.cloud_service import CloudService
from app.cache import SemanticCache
from app.router import PromptRouter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("gateway")

# Services & Cache Lifespan Manager
ollama_service = None
cloud_service = None
semantic_cache = None
prompt_router = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ollama_service, cloud_service, semantic_cache, prompt_router
    logger.info("Initializing services...")
    ollama_service = OllamaService()
    cloud_service = CloudService()
    semantic_cache = SemanticCache()
    prompt_router = PromptRouter()
    
    yield
    
    logger.info("Closing connections...")
    await ollama_service.close()
    await cloud_service.close()

app = FastAPI(
    title="Hybrid LLM Router & Cache Gateway",
    description="Production-grade API Gateway with Semantic Caching and Local-to-Cloud Routing",
    version="1.0.0",
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
    start_time = time.time()
    prompt = request.prompt
    
    # 1. Try Semantic Cache first (unless forced to bypass)
    if not request.force_cloud:
        cached_response, similarity = await cache.get(prompt, ollama)
        if cached_response is not None:
            latency_ms = (time.time() - start_time) * 1000.0
            return QueryResponse(
                response=cached_response,
                source=f"cache (similarity: {similarity:.4f})",
                latency_ms=round(latency_ms, 2),
                tokens_used=0,
                cost_usd=0.0
            )

    # 2. Determine routing logic (Local SLM vs. Cloud LLM)
    is_cloud_route = request.force_cloud or router.is_complex(prompt)
    
    response_text = ""
    source = ""
    tokens = 0
    cost = 0.0

    if not is_cloud_route:
        # Route to Local SLM (Ollama)
        logger.info("Routing query to Local SLM (Ollama)...")
        if await ollama.is_healthy():
            try:
                response_text = await ollama.generate(
                    prompt=prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
                source = "local_slm"
            except Exception as e:
                logger.error(f"Local SLM generation failed: {e}. Falling back to Cloud LLM.")
                is_cloud_route = True
        else:
            logger.warning("Local Ollama server is unhealthy. Falling back to Cloud LLM.")
            is_cloud_route = True

    if is_cloud_route:
        # Route to Cloud LLM (Gemini)
        logger.info("Routing query to Cloud LLM (Gemini)...")
        try:
            response_text, tokens, cost = await cloud.generate(
                prompt=prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            source = "cloud_llm" if not request.force_cloud and router.is_complex(prompt) else "cloud_llm_forced"
            if source == "cloud_llm" and not router.is_complex(prompt):
                source = "cloud_llm_fallback"
        except Exception as e:
            logger.error(f"Cloud LLM generation failed: {e}")
            if not request.force_cloud and await ollama.is_healthy():
                logger.info("Attempting local SLM fallback...")
                try:
                    response_text = await ollama.generate(
                        prompt=prompt,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens
                    )
                    source = "local_slm_fallback"
                except Exception as local_err:
                    logger.error(f"Local fallback also failed: {local_err}")
                    raise HTTPException(status_code=502, detail=f"Both cloud and local models failed. Cloud error: {e}")
            else:
                raise HTTPException(status_code=502, detail=f"Cloud model execution failed and no fallback possible. Error: {e}")

    # 3. Store in semantic cache in the BACKGROUND (non-blocking)
    if response_text:
        background_tasks.add_task(cache.set, prompt, response_text, ollama)

    latency_ms = (time.time() - start_time) * 1000.0
    return QueryResponse(
        response=response_text,
        source=source,
        latency_ms=round(latency_ms, 2),
        tokens_used=tokens,
        cost_usd=cost
    )

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
        "cache_db_path": settings.CACHE_DB_PATH,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD
    }

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
