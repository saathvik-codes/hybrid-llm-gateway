# Hybrid LLM Router & Semantic Cache Gateway

A production-grade, high-performance API Gateway (written in Python/FastAPI) that intercepts LLM requests. It is designed to optimize response latency and minimize cloud API costs using dynamic query routing and semantic vector caching.

## 🚀 Key Features

1. **Semantic Vector Cache**:
   * Saves past prompts and their model responses.
   * Computes prompt embeddings locally using Ollama.
   * Compares incoming prompts using **cosine similarity** (threshold $\ge 0.95$).
   * Serves cache hits in **$< 15\text{ms}$** with \$0 cost.
2. **Dynamic Complexity Router**:
   * Uses regex keyword matching and character length filters to categorize queries as "simple" or "complex" in microseconds.
   * **Simple Queries**: Routed to a local Small Language Model (SLM) running via Ollama.
   * **Complex Queries**: Forwarded to a premium cloud provider (Google Gemini).
3. **Resilience & Fallback**:
   * Cascading fallbacks: If the local Ollama instance is down, simple queries fail over to Gemini.
   * Reverse fallbacks: If Gemini returns a rate-limit error (HTTP 429) or times out, the query automatically fails over to the local model to ensure high availability.
4. **Token & Cost Auditing**:
   * Tracks latency, token consumption, and estimates actual USD cost for cloud requests.

---

## 🛠️ Tech Stack

* **Framework**: FastAPI (Asynchronous API endpoints)
* **Local Inference**: Ollama (`llama3` for text generation, `nomic-embed-text` for vector embeddings)
* **Cloud Inference**: Google Gemini API (`gemini-1.5-flash`)
* **Vector Math**: NumPy (Cosine similarity calculation)
* **Testing**: Pytest (Mock-based integration tests)

---

## 📁 Repository Structure

```text
hybrid_llm_gateway/
├── app/
│   ├── services/
│   │   ├── ollama_service.py # Local model handler
│   │   └── cloud_service.py  # Gemini API handler with retries
│   ├── cache.py              # Semantic vector cache logic
│   ├── config.py             # Pydantic Settings configuration
│   ├── main.py               # Main FastAPI server & endpoints
│   ├── models.py             # Request/Response schemas
│   └── router.py             # Dynamic complexity router
├── tests/
│   ├── test_cache.py         # Semantic cache unit tests
│   ├── test_gateway.py       # Mock-based endpoint tests
│   └── test_router.py        # Routing heuristic tests
├── .env.example              # Example configurations
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Pre-requisites
Download and run **Ollama** locally on your machine:
* [Download Ollama](https://ollama.com/)
* Pull the required models in your terminal:
  ```bash
  ollama pull llama3
  ollama pull nomic-embed-text
  ```

### 2. Install Python Dependencies
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Variables
Copy `.env.example` to `.env` and add your **Gemini API Key**:
```bash
cp .env.example .env
```
Update your `.env` file:
```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

---

## 🧪 Running & Testing

### Run Automated Tests
Execute the pytest suite to verify cache, routing, and fallback logic:
```bash
python -m pytest
```

### Run the API Gateway Server
Start the server locally:
```bash
uvicorn app.main:app --reload --port 8000
```
Access the interactive OpenAPI docs at `http://127.0.0.1:8000/docs`.

---

## 📡 API Endpoints

### 1. Post Query (`POST /query`)
Send a prompt to the gateway.
* **Request**:
  ```json
  {
    "prompt": "what is 2 + 2?",
    "temperature": 0.7
  }
  ```
* **Response**:
  ```json
  {
    "response": "2 + 2 is 4.",
    "source": "local_slm",
    "latency_ms": 284.50,
    "tokens_used": 0,
    "cost_usd": 0.0
  }
  ```

*Note: The second time you run this query, the source will change to `cache (similarity: 1.0000)` and latency will drop to $< 10\text{ms}$.*

### 2. Get Cache Stats (`GET /cache/stats`)
See active entries and database location.

### 3. Clear Cache (`POST /cache/clear`)
Purges all entries from the semantic cache.
