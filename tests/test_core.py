import pytest
from fastapi.testclient import TestClient

from app import metrics
from app.core import GatewayError
from app.main import app
from app.models import QueryRequest


def sample(name, **labels):
    return metrics.REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.mark.asyncio
async def test_cache_hit_skips_models_and_counts(make_core):
    core, ollama, cloud, cache, schedule = make_core(cache_hit=("cached", 0.97))
    before = sample("gateway_requests_total", source="cache")
    r = await core.handle(QueryRequest(prompt="hello there"), schedule)
    assert r.response == "cached" and r.source.startswith("cache")
    ollama.generate.assert_not_called()
    cloud.generate.assert_not_called()
    assert sample("gateway_requests_total", source="cache") == before + 1


@pytest.mark.asyncio
async def test_simple_prompt_goes_local_and_is_cached_afterwards(make_core):
    core, ollama, cloud, cache, schedule = make_core()
    scheduled = []
    r = await core.handle(QueryRequest(prompt="hello there"), lambda fn, *a: scheduled.append((fn, a)))
    assert (r.source, r.response) == ("local_slm", "local answer")
    assert scheduled and scheduled[0][0] == cache.set      # the answer is stored without making the caller wait


@pytest.mark.asyncio
async def test_complex_prompt_goes_to_cloud_and_spend_is_tracked(make_core):
    core, ollama, cloud, cache, schedule = make_core(complex_prompt=True)
    r = await core.handle(QueryRequest(prompt="design a system"), schedule)
    assert (r.source, r.tokens_used, r.cost_usd) == ("cloud_llm", 120, 0.002)
    assert core.policy.spent_today == pytest.approx(0.002)


@pytest.mark.asyncio
async def test_local_down_falls_back_to_cloud(make_core):
    core, ollama, cloud, cache, schedule = make_core(local_healthy=False)
    r = await core.handle(QueryRequest(prompt="hello there"), schedule)
    assert r.source == "cloud_llm_fallback"


@pytest.mark.asyncio
async def test_cloud_failure_falls_back_to_local(make_core):
    core, ollama, cloud, cache, schedule = make_core(complex_prompt=True)
    cloud.generate.side_effect = RuntimeError("429")
    before = sample("gateway_provider_errors_total", provider="gemini")
    r = await core.handle(QueryRequest(prompt="design a system"), schedule)
    assert r.source == "local_slm_fallback"
    assert sample("gateway_provider_errors_total", provider="gemini") == before + 1


@pytest.mark.asyncio
async def test_forced_cloud_failure_is_a_502_not_a_silent_downgrade(make_core):
    core, ollama, cloud, cache, schedule = make_core()
    cloud.generate.side_effect = RuntimeError("down")
    with pytest.raises(GatewayError) as e:
        await core.handle(QueryRequest(prompt="hello there", force_cloud=True), schedule)
    assert e.value.status_code == 502
    ollama.generate.assert_not_called()


@pytest.mark.asyncio
async def test_both_providers_down_is_a_502(make_core):
    core, ollama, cloud, cache, schedule = make_core(complex_prompt=True)
    cloud.generate.side_effect = RuntimeError("down")
    ollama.generate.side_effect = RuntimeError("down")
    with pytest.raises(GatewayError) as e:
        await core.handle(QueryRequest(prompt="design a system"), schedule)
    assert e.value.status_code == 502


@pytest.mark.asyncio
async def test_budget_exhausted_degrades_complex_prompts_to_local(make_core):
    core, ollama, cloud, cache, schedule = make_core(complex_prompt=True, limit=0.001)
    core.policy.record(0.001)
    r = await core.handle(QueryRequest(prompt="design a system"), schedule)
    assert r.source == "local_slm_budget"
    cloud.generate.assert_not_called()


@pytest.mark.asyncio
async def test_budget_exhausted_rejects_forced_cloud(make_core):
    core, ollama, cloud, cache, schedule = make_core(limit=0.001)
    core.policy.record(0.5)
    with pytest.raises(GatewayError) as e:
        await core.handle(QueryRequest(prompt="hello there", force_cloud=True), schedule)
    assert e.value.status_code == 429


@pytest.mark.asyncio
async def test_budget_runs_out_mid_stream(make_core):
    core, ollama, cloud, cache, schedule = make_core(complex_prompt=True, limit=0.003)
    sources = [(await core.handle(QueryRequest(prompt=f"design {i}"), schedule)).source for i in range(3)]
    assert sources == ["cloud_llm", "cloud_llm", "local_slm_budget"]      # 0.002 each: the third call is over the cap


@pytest.mark.asyncio
async def test_own_key_skips_shared_budget_and_cache_writes(make_core):
    core, ollama, cloud, cache, _ = make_core(complex_prompt=True, limit=0.001)
    core.policy.record(0.5)                                    # shared budget is used up...
    scheduled = []
    r = await core.handle(QueryRequest(prompt="design a system"), lambda fn, *a: scheduled.append(fn), own_key=True)
    assert r.source == "cloud_llm"                             # ...but the visitor pays with their own key
    assert core.policy.spent_today == pytest.approx(0.5)       # and their spend isn't added to it
    assert scheduled == []                                     # their prompt doesn't land in the public cache


@pytest.mark.asyncio
async def test_rejected_own_key_says_so(make_core):
    import httpx
    core, ollama, cloud, cache, schedule = make_core(complex_prompt=True)
    req = httpx.Request("POST", "https://example.invalid")
    cloud.generate.side_effect = httpx.HTTPStatusError("bad key", request=req, response=httpx.Response(400, request=req))
    with pytest.raises(GatewayError) as e:
        await core.handle(QueryRequest(prompt="design a system"), schedule, own_key=True)
    assert "rejected your key" in e.value.detail


def test_query_uses_header_key_per_request(monkeypatch):
    seen = {}

    async def fake_generate(self, prompt, **kw):
        seen["key"] = self.api_key
        return "byok answer", 10, 0.0001
    monkeypatch.setattr("app.services.cloud_service.CloudService.generate", fake_generate)
    r = TestClient(app).post("/query", json={"prompt": "hi", "force_cloud": True}, headers={"X-Gemini-Key": "AIza-test"})
    assert r.status_code == 200 and r.json()["response"] == "byok answer"
    assert seen["key"] == "AIza-test"


def test_query_rejects_garbage_key():
    r = TestClient(app).post("/query", json={"prompt": "hi"}, headers={"X-Gemini-Key": "x" * 500})
    assert r.status_code == 400


def test_metrics_endpoint_exposes_gateway_series():
    r = TestClient(app).get("/metrics")
    assert r.status_code == 200
    assert "gateway_requests_total" in r.text and "gateway_cost_usd_total" in r.text


def test_budget_endpoint_reports_policy():
    r = TestClient(app).get("/budget")
    assert r.status_code == 200
    assert {"daily_limit_usd", "spent_today_usd", "remaining_usd", "exceeded"} <= r.json().keys()
