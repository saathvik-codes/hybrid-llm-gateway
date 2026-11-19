"""
The gateway as an MCP tool server, so an agent (Claude Desktop, an IDE, any MCP client) can route a prompt through the
cache and the local/cloud router like any other tool:

    python -m app.mcp_server            # stdio transport

Tools: ask(prompt, force_cloud) and cache_stats(). It uses the same GatewayCore as the REST and gRPC front ends.
"""
import asyncio

try:                                                # mcp 2.x renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                                 # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from app.config import settings
from app.core import GatewayCore, build_core
from app.models import QueryRequest

server = _Server("hybrid-llm-gateway")
_core: GatewayCore | None = None
_pending: set[asyncio.Task] = set()


def _get_core() -> GatewayCore:
    global _core
    if _core is None:
        _core = build_core()
    return _core


def _schedule(fn, *args) -> None:
    task = asyncio.ensure_future(fn(*args))
    _pending.add(task)
    task.add_done_callback(_pending.discard)


@server.tool()
async def ask(prompt: str, force_cloud: bool = False) -> str:
    """Answer a prompt through the gateway: served from the semantic cache when a similar prompt was answered before,
    otherwise by the local model (simple prompts) or the cloud model (complex ones). Set force_cloud to skip the
    cache and local model. The reply ends with a line saying where it came from and what it cost."""
    r = await _get_core().handle(QueryRequest(prompt=prompt, force_cloud=force_cloud), _schedule)
    return f"{r.response}\n\n[source: {r.source} | {r.latency_ms} ms | ${r.cost_usd:.6f}]"


@server.tool()
async def cache_stats() -> dict:
    """How many answers the semantic cache holds, and its settings."""
    cache = _get_core().cache
    return {"entries": len(cache.entries), "backend": settings.CACHE_BACKEND, "similarity_threshold": cache.threshold}


if __name__ == "__main__":
    server.run()
