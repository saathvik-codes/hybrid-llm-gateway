"""Prometheus metrics, served at /metrics. A private registry keeps them out of the way of tests and reloads."""
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

REGISTRY = CollectorRegistry()

REQUESTS = Counter("gateway_requests_total", "Requests answered, by where the answer came from.", ["source"], registry=REGISTRY)
LATENCY = Histogram("gateway_request_latency_seconds", "End-to-end request latency.", ["source"], registry=REGISTRY,
                    buckets=(0.005, 0.015, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30))
CACHE_LOOKUPS = Counter("gateway_cache_lookups_total", "Semantic cache lookups.", ["result"], registry=REGISTRY)
TOKENS = Counter("gateway_tokens_total", "Tokens used by cloud calls.", registry=REGISTRY)
COST = Counter("gateway_cost_usd_total", "Estimated cloud spend in USD.", registry=REGISTRY)
PROVIDER_ERRORS = Counter("gateway_provider_errors_total", "Failed model calls.", ["provider"], registry=REGISTRY)
BUDGET_REJECTIONS = Counter("gateway_budget_events_total", "Cost-policy decisions.", ["action"], registry=REGISTRY)
BUDGET_LEFT = Gauge("gateway_budget_remaining_usd", "Cloud budget left today (-1 when no budget is set).", registry=REGISTRY)


def render() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
