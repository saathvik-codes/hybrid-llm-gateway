import uuid
from pydantic import BaseModel, Field
from typing import Optional, List

class QueryRequest(BaseModel):
    prompt: str = Field(..., description="The input text prompt for the LLM")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(1000, ge=1, description="Maximum number of tokens to generate")
    force_cloud: bool = Field(False, description="Bypass local SLM and cache, force cloud execution")

class QueryResponse(BaseModel):
    response: str = Field(..., description="The generated text response")
    source: str = Field(..., description="Source of the response: cache, local_slm, cloud_llm, or fallback")
    latency_ms: float = Field(..., description="Time taken to process the query in milliseconds")
    tokens_used: int = Field(0, description="Total tokens used in request and response")
    cost_usd: float = Field(0.0, description="Calculated API cost in USD")

class CacheEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    embedding: List[float]
    response: str
    timestamp: float
