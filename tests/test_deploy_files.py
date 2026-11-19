"""The deployment files are code too: keep them consistent with the app and with each other."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def docs(name):
    return [d for d in yaml.safe_load_all((ROOT / name).read_text(encoding="utf-8")) if d]


def by_kind(items, kind):
    return [d for d in items if d["kind"] == kind]


def test_deployment_has_probes_limits_and_runs_unprivileged():
    deployment = by_kind(docs("k8s/gateway.yaml"), "Deployment")[0]
    spec = deployment["spec"]["template"]["spec"]
    c = spec["containers"][0]
    assert c["readinessProbe"]["httpGet"]["path"] == "/health" and c["livenessProbe"]["httpGet"]["path"] == "/health"
    assert "limits" in c["resources"] and "requests" in c["resources"]
    assert spec["securityContext"]["runAsNonRoot"] is True
    assert c["securityContext"]["allowPrivilegeEscalation"] is False


def test_service_ports_match_the_container_ports():
    items = docs("k8s/gateway.yaml")
    container_ports = {p["name"]: p["containerPort"] for p in by_kind(items, "Deployment")[0]["spec"]["template"]["spec"]["containers"][0]["ports"]}
    service_targets = {p["targetPort"] for p in by_kind(items, "Service")[0]["spec"]["ports"]}
    assert service_targets <= set(container_ports) and container_ports == {"http": 8000, "grpc": 50051}


def test_autoscaler_targets_the_deployment():
    items = docs("k8s/gateway.yaml")
    hpa = by_kind(items, "HorizontalPodAutoscaler")[0]["spec"]
    assert hpa["scaleTargetRef"]["name"] == by_kind(items, "Deployment")[0]["metadata"]["name"]
    assert hpa["minReplicas"] >= 2          # a single replica would make the rolling update an outage


def test_config_uses_redis_so_replicas_share_the_cache():
    config = by_kind(docs("k8s/gateway.yaml"), "ConfigMap")[0]["data"]
    assert config["CACHE_BACKEND"] == "redis" and config["REDIS_URL"].startswith("redis://redis:")
    redis_service = by_kind(docs("k8s/redis.yaml"), "Service")[0]["metadata"]["name"]
    assert redis_service == "redis"


def test_every_config_key_is_a_real_setting():
    from app.config import Settings
    known = set(Settings.model_fields)
    assert set(by_kind(docs("k8s/gateway.yaml"), "ConfigMap")[0]["data"]) <= known
    compose_env = set(docs("docker-compose.yml")[0]["services"]["gateway"]["environment"])
    assert compose_env <= known


def test_compose_gateway_waits_for_a_healthy_redis():
    services = docs("docker-compose.yml")[0]["services"]
    assert services["gateway"]["depends_on"]["redis"]["condition"] == "service_healthy"
    assert "healthcheck" in services["redis"]


def test_ci_runs_tests_and_builds_the_image():
    wf = docs(".github/workflows/ci.yml")[0]
    assert {"test", "docker", "manifests"} <= set(wf["jobs"])
    steps = " ".join(str(s.get("run", "")) for s in wf["jobs"]["test"]["steps"])
    assert "gen_proto.py" in steps and "pytest" in steps
