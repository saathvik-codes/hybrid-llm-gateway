import subprocess
import sys
import unittest.mock as mock
from pathlib import Path

import pytest

from app.budget import BudgetPolicy
from app.core import GatewayCore

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "app" / "gateway_pb2.py").exists():          # gRPC stubs are build output, not committed
    subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_proto.py")], check=True)


@pytest.fixture
def make_core():
    """GatewayCore over mocked providers. Returns (core, ollama, cloud, cache, schedule) so tests can steer and inspect."""
    def build(complex_prompt=False, cache_hit=None, local_healthy=True, limit=0.0, clock=None):
        ollama, cloud, cache = mock.AsyncMock(), mock.AsyncMock(), mock.AsyncMock()
        router = mock.Mock()
        router.is_complex.return_value = complex_prompt
        cache.get.return_value = cache_hit if cache_hit else (None, 0.0)
        ollama.is_healthy.return_value = local_healthy
        ollama.generate.return_value = "local answer"
        cloud.generate.return_value = ("cloud answer", 120, 0.002)
        scheduled = []
        policy = BudgetPolicy(limit, clock=clock) if clock else BudgetPolicy(limit)
        core = GatewayCore(ollama, cloud, cache, router, policy)
        return core, ollama, cloud, cache, (lambda fn, *a: scheduled.append((fn, a)))
    return build
