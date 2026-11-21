"""AWS deployment: the DynamoDB cache against moto's in-memory AWS, the Lambda handler, and the SAM template."""
import json
import unittest.mock as mock
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from app.cache import DynamoDBSemanticCache

ROOT = Path(__file__).resolve().parents[1]


def ollama_with(vec):
    o = mock.AsyncMock()
    o.get_embedding.return_value = vec
    return o


@pytest.fixture
def table(monkeypatch):
    for k, v in {"AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test", "AWS_DEFAULT_REGION": "ap-south-1"}.items():
        monkeypatch.setenv(k, v)                    # moto never talks to AWS; these just satisfy boto3
    with mock_aws():
        ddb = boto3.resource("dynamodb", region_name="ap-south-1")
        t = ddb.create_table(TableName="cache", BillingMode="PAY_PER_REQUEST",
                             AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
                             KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}])
        yield t


@pytest.mark.asyncio
async def test_an_answer_cached_by_one_lambda_instance_is_served_by_another(table):
    a = DynamoDBSemanticCache(table=table, refresh_seconds=0, threshold=0.9)
    b = DynamoDBSemanticCache(table=table, refresh_seconds=0, threshold=0.9)
    await a.set("what is raft", "a consensus algorithm", ollama_with([1.0, 0.0]))
    response, sim = await b.get("explain raft", ollama_with([0.99, 0.05]))
    assert response == "a consensus algorithm" and sim > 0.99


@pytest.mark.asyncio
async def test_entries_survive_a_cold_start(table):
    await DynamoDBSemanticCache(table=table).set("p", "r", ollama_with([0.0, 1.0]))
    assert len(DynamoDBSemanticCache(table=table).entries) == 1


@pytest.mark.asyncio
async def test_delete_and_clear_reach_the_table(table):
    cache = DynamoDBSemanticCache(table=table, refresh_seconds=0, threshold=0.9)
    for i in range(30):
        await cache.set(f"prompt {i}", f"r{i}", ollama_with([1.0, float(i)]))
    assert table.scan(Select="COUNT")["Count"] == 30
    assert cache.delete(cache.entries[0].id)
    assert table.scan(Select="COUNT")["Count"] == 29
    cache.clear()
    assert table.scan(Select="COUNT")["Count"] == 0


@pytest.mark.asyncio
async def test_an_aws_outage_degrades_to_a_cache_miss_not_an_error():
    broken = mock.Mock()
    broken.scan.side_effect = ConnectionError("throttled")
    broken.put_item.side_effect = ConnectionError("throttled")
    cache = DynamoDBSemanticCache(table=broken, refresh_seconds=3600)
    await cache.set("p", "r", ollama_with([1.0, 0.0]))            # must not raise
    assert (await cache.get("p", ollama_with([1.0, 0.0])))[0] == "r"


def http_api_event(method: str, path: str) -> dict:
    """The event API Gateway (HTTP API, payload v2) sends to Lambda."""
    return {"version": "2.0", "routeKey": "$default", "rawPath": path, "rawQueryString": "",
            "headers": {"host": "api.example.com"}, "isBase64Encoded": False,
            "requestContext": {"http": {"method": method, "path": path, "protocol": "HTTP/1.1", "sourceIp": "1.2.3.4",
                                        "userAgent": "test"}, "stage": "$default", "requestId": "r1",
                               "routeKey": "$default", "accountId": "1", "apiId": "a", "domainName": "api.example.com",
                               "domainPrefix": "api", "time": "01/Jan/2026:00:00:00 +0000", "timeEpoch": 0}}


def test_lambda_handler_serves_the_api(monkeypatch):
    monkeypatch.setattr("app.config.settings.GRPC_PORT", 0)
    from app.lambda_handler import handler
    r = handler(http_api_event("GET", "/budget"), {})
    assert r["statusCode"] == 200 and "daily_limit_usd" in json.loads(r["body"])


def test_sam_template_is_valid_cloudformation():
    from cfnlint.api import lint_all
    problems = [m for m in lint_all((ROOT / "template.yaml").read_text(encoding="utf-8")) if m.rule.severity == "error"]
    assert problems == [], problems


def test_template_gives_the_function_its_table_and_no_wider_access():
    import yaml

    class Loader(yaml.SafeLoader):
        pass
    for tag in ("!Ref", "!Sub", "!GetAtt"):
        Loader.add_constructor(tag, lambda loader, node: loader.construct_scalar(node))
    t = yaml.load((ROOT / "template.yaml").read_text(encoding="utf-8"), Loader=Loader)
    fn = t["Resources"]["GatewayFunction"]["Properties"]
    assert fn["Environment"]["Variables"]["CACHE_BACKEND"] == "dynamodb"
    assert fn["Environment"]["Variables"]["DYNAMODB_TABLE"] == "CacheTable"
    assert fn["Policies"] == [{"DynamoDBCrudPolicy": {"TableName": "CacheTable"}}]
    assert t["Parameters"]["GeminiApiKey"]["NoEcho"] is True
