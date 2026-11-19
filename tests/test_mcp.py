import pytest

from app import mcp_server


@pytest.mark.asyncio
async def test_ask_tool_returns_the_answer_and_its_source(make_core, monkeypatch):
    core, *_ = make_core()
    monkeypatch.setattr(mcp_server, "_core", core)
    out = await mcp_server.ask("hello there")
    assert out.startswith("local answer") and "source: local_slm" in out


@pytest.mark.asyncio
async def test_cache_stats_tool(make_core, monkeypatch):
    core, _, _, cache, _ = make_core()
    cache.entries, cache.threshold = [object(), object()], 0.95
    monkeypatch.setattr(mcp_server, "_core", core)
    assert (await mcp_server.cache_stats())["entries"] == 2


@pytest.mark.asyncio
async def test_both_tools_are_registered_with_the_server():
    tools = await mcp_server.server.list_tools()
    names = {t.name for t in (tools if isinstance(tools, list) else tools.tools)}
    assert names == {"ask", "cache_stats"}
