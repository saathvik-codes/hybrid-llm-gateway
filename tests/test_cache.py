import pytest
from app.cache import cosine_similarity, SemanticCache
from app.models import CacheEntry
import unittest.mock as mock

def test_cosine_similarity():
    # Exact match
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    # Orthogonal
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    # Zero vector
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

@pytest.mark.asyncio
async def test_cache_hit_and_miss():
    cache = SemanticCache(db_path="./test_cache.json", threshold=0.90)
    cache.clear()
    
    # Mock OllamaService
    mock_ollama = mock.AsyncMock()
    mock_ollama.get_embedding.side_effect = [
        [1.0, 0.0], # Embedding for set prompt
        [0.99, 0.1], # Embedding for test get prompt (close match)
        [0.0, 1.0]  # Embedding for test get prompt (orthogonal)
    ]
    
    await cache.set("test prompt", "cached output", mock_ollama)
    
    # Check close match -> should hit
    response, similarity = await cache.get("test prompt close", mock_ollama)
    assert response == "cached output"
    assert similarity >= 0.90
    
    # Check orthogonal match -> should miss
    response, similarity = await cache.get("completely different", mock_ollama)
    assert response is None
    assert similarity < 0.90
    
    cache.clear()


@pytest.mark.asyncio
async def test_real_ollama_cache_flow():
    # Only run this if a local Ollama server is running and healthy
    from app.services.ollama_service import OllamaService
    
    ollama = OllamaService()
    if not await ollama.is_healthy():
        pytest.skip("Local Ollama server is not running or unhealthy. Skipping real integration test.")
    
    try:
        cache = SemanticCache(db_path="./test_real_cache.json", threshold=0.85)
        cache.clear()
        
        # Insert a real entry with Ollama embeddings
        await cache.set("What is the capital of France?", "The capital of France is Paris.", ollama)
        
        # Test a semantically identical query
        res, similarity = await cache.get("Tell me the capital of France.", ollama)
        assert res == "The capital of France is Paris."
        assert similarity >= 0.85
        
        # Test an unrelated query
        res_miss, similarity_miss = await cache.get("How do I cook pasta?", ollama)
        assert res_miss is None
        assert similarity_miss < 0.85
        
        cache.clear()
    finally:
        await ollama.close()

