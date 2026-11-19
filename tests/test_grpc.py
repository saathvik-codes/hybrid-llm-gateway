import grpc
import pytest

from app import gateway_pb2, gateway_pb2_grpc
from app.grpc_server import start_grpc


async def serve(core):
    server = await start_grpc(core, 0)
    channel = grpc.aio.insecure_channel(f"localhost:{server.bound_port}")
    return server, channel, gateway_pb2_grpc.GatewayStub(channel)


async def stop(server, channel):
    await channel.close()
    await server.stop(None)


@pytest.mark.asyncio
async def test_query_over_grpc_matches_rest_behaviour(make_core):
    core, *_ = make_core()
    server, channel, stub = await serve(core)
    try:
        r = await stub.Query(gateway_pb2.QueryRequest(prompt="hello there"))
        assert (r.response, r.source) == ("local answer", "local_slm")
    finally:
        await stop(server, channel)


@pytest.mark.asyncio
async def test_defaults_are_applied_for_unset_fields(make_core):
    core, ollama, *_ = make_core()
    server, channel, stub = await serve(core)
    try:
        await stub.Query(gateway_pb2.QueryRequest(prompt="hello there"))
        kwargs = ollama.generate.call_args.kwargs
        assert kwargs["max_tokens"] == 1000 and kwargs["temperature"] == pytest.approx(0.7)
    finally:
        await stop(server, channel)


@pytest.mark.asyncio
async def test_empty_prompt_is_invalid_argument(make_core):
    core, *_ = make_core()
    server, channel, stub = await serve(core)
    try:
        with pytest.raises(grpc.aio.AioRpcError) as e:
            await stub.Query(gateway_pb2.QueryRequest(prompt="   "))
        assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        await stop(server, channel)


@pytest.mark.asyncio
async def test_budget_exhaustion_maps_to_resource_exhausted(make_core):
    core, *_ = make_core(limit=0.001)
    core.policy.record(1.0)
    server, channel, stub = await serve(core)
    try:
        with pytest.raises(grpc.aio.AioRpcError) as e:
            await stub.Query(gateway_pb2.QueryRequest(prompt="hello there", force_cloud=True))
        assert e.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
    finally:
        await stop(server, channel)


@pytest.mark.asyncio
async def test_provider_failure_maps_to_unavailable(make_core):
    core, ollama, cloud, *_ = make_core()
    cloud.generate.side_effect = RuntimeError("down")
    server, channel, stub = await serve(core)
    try:
        with pytest.raises(grpc.aio.AioRpcError) as e:
            await stub.Query(gateway_pb2.QueryRequest(prompt="hello there", force_cloud=True))
        assert e.value.code() == grpc.StatusCode.UNAVAILABLE
    finally:
        await stop(server, channel)


@pytest.mark.asyncio
async def test_health_over_grpc(make_core):
    core, *_ = make_core()
    server, channel, stub = await serve(core)
    try:
        h = await stub.Health(gateway_pb2.HealthRequest())
        assert h.local_connected is True and h.status == "healthy"
    finally:
        await stop(server, channel)
