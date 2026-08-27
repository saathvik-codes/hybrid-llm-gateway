import pytest
from fastapi.testclient import TestClient
from app.main import app, get_ollama_service, get_cloud_service, get_cache, get_router
from app.models import CacheEntry
import unittest.mock as mock

@pytest.fixture
def mock_services():
    mock_ollama = mock.AsyncMock()
    mock_cloud = mock.AsyncMock()
    mock_cache = mock.AsyncMock()
    mock_cache.delete = mock.Mock()
    mock_cache.clear = mock.Mock()
    mock_router = mock.Mock()
    
    app.dependency_overrides[get_ollama_service] = lambda: mock_ollama
    app.dependency_overrides[get_cloud_service] = lambda: mock_cloud
    app.dependency_overrides[get_cache] = lambda: mock_cache
    app.dependency_overrides[get_router] = lambda: mock_router
    
    yield mock_ollama, mock_cloud, mock_cache, mock_router
    
    app.dependency_overrides.clear()

def test_serve_dashboard():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "LLM Gateway - Admin Console" in response.text

def test_gateway_cache_hit(mock_services):
    mock_ollama, mock_cloud, mock_cache, mock_router = mock_services
    
    # Setup cache hit response
    mock_cache.get.return_value = ("cached response text", 0.98)
    
    client = TestClient(app)
    response = client.post("/query", json={"prompt": "hello there"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "cached response text"
    assert "cache" in data["source"]
    
    # Verify that LLMs were NOT called
    mock_ollama.generate.assert_not_called()
    mock_cloud.generate.assert_not_called()

def test_gateway_route_local(mock_services):
    mock_ollama, mock_cloud, mock_cache, mock_router = mock_services
    
    # Cache miss
    mock_cache.get.return_value = (None, 0.0)
    # Simple route
    mock_router.is_complex.return_value = False
    # Local health check OK
    mock_ollama.is_healthy.return_value = True
    # Local output
    mock_ollama.generate.return_value = "local slm response"
    
    client = TestClient(app)
    response = client.post("/query", json={"prompt": "hello there"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "local slm response"
    assert data["source"] == "local_slm"
    
    mock_ollama.generate.assert_called_once()
    mock_cloud.generate.assert_not_called()

def test_gateway_route_cloud(mock_services):
    mock_ollama, mock_cloud, mock_cache, mock_router = mock_services
    
    # Cache miss
    mock_cache.get.return_value = (None, 0.0)
    # Complex route
    mock_router.is_complex.return_value = True
    # Cloud output
    mock_cloud.generate.return_value = ("cloud llm response", 100, 0.00015)
    
    client = TestClient(app)
    response = client.post("/query", json={"prompt": "solve 2+2"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "cloud llm response"
    assert data["source"] == "cloud_llm"
    
    mock_cloud.generate.assert_called_once()
    mock_ollama.generate.assert_not_called()

def test_get_cache_entries(mock_services):
    mock_ollama, mock_cloud, mock_cache, mock_router = mock_services
    
    # Mock data list
    mock_cache.entries = [
        CacheEntry(id="123", prompt="test prompt 1", response="response 1", embedding=[], timestamp=0.0)
    ]
    
    client = TestClient(app)
    response = client.get("/api/cache")
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == "123"
    assert data[0]["prompt"] == "test prompt 1"

def test_delete_cache_entry_success(mock_services):
    mock_ollama, mock_cloud, mock_cache, mock_router = mock_services
    
    mock_cache.delete.return_value = True
    
    client = TestClient(app)
    response = client.delete("/api/cache/123")
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Cache entry deleted successfully"
    mock_cache.delete.assert_called_once_with("123")

def test_delete_cache_entry_not_found(mock_services):
    mock_ollama, mock_cloud, mock_cache, mock_router = mock_services
    
    mock_cache.delete.return_value = False
    
    client = TestClient(app)
    response = client.delete("/api/cache/123")
    
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()
