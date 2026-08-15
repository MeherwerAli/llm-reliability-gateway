import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal

import httpx
import pytest
from pydantic import ValidationError

from llm_gateway.api import create_app
from llm_gateway.config import Settings
from llm_gateway.models import ChatCompletionRequest, ResponseFormat
from llm_gateway.pricing import PricingCatalog
from llm_gateway.providers import ProviderResponse
from llm_gateway.rate_limit import InMemoryTokenBucketLimiter
from llm_gateway.service import GatewayService
from llm_gateway.store import InMemoryGatewayStore

HEADERS = {"Authorization": "Bearer test-key", "Idempotency-Key": "request-1"}


def settings(**overrides: object) -> Settings:
    return Settings(
        env="test",
        api_key="test-key",
        monthly_budget_usd=Decimal("1.00"),
        requests_per_minute=10,
        tokens_per_minute=10_000,
        **overrides,
    )


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as value:
        yield value


@pytest.mark.asyncio
async def test_estimate_and_completion_are_accounted_and_idempotent(
    client: httpx.AsyncClient,
) -> None:
    payload = {
        "model": "portfolio-model",
        "messages": [{"role": "user", "content": "Explain bounded retries."}],
        "max_tokens": 100,
    }
    estimate = await client.post("/v1/estimates", json=payload, headers=HEADERS)
    assert estimate.status_code == 200
    assert estimate.json()["budget_status"] == "allowed"
    first = await client.post("/v1/chat/completions", json=payload, headers=HEADERS)
    assert first.status_code == 200
    assert first.headers["X-Idempotent-Replay"] == "false"
    replay = await client.post("/v1/chat/completions", json=payload, headers=HEADERS)
    assert replay.status_code == 200
    assert replay.headers["X-Idempotent-Replay"] == "true"
    usage = await client.get("/v1/usage", headers=HEADERS)
    assert usage.json()["request_count"] == 1
    assert usage.json()["groups"][0]["endpoint"] == "/v1/chat/completions"
    assert usage.json()["groups"][0]["model"] == "portfolio-model"


@pytest.mark.asyncio
async def test_idempotency_key_rejects_changed_payload(client: httpx.AsyncClient) -> None:
    first = {"model": "portfolio-model", "messages": [{"role": "user", "content": "one"}]}
    second = {"model": "portfolio-model", "messages": [{"role": "user", "content": "two"}]}
    assert (
        await client.post("/v1/chat/completions", json=first, headers=HEADERS)
    ).status_code == 200
    conflict = await client.post("/v1/chat/completions", json=second, headers=HEADERS)
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "idempotency_conflict"


class SequenceProvider:
    def __init__(self, values: list[str]) -> None:
        self.values = iter(values)

    async def generate(self, request: ChatCompletionRequest) -> ProviderResponse:
        return ProviderResponse(next(self.values))

    async def stream(self, request: ChatCompletionRequest):  # type: ignore[no-untyped-def]
        yield "unused"


class CapturingProvider:
    def __init__(self) -> None:
        self.request: ChatCompletionRequest | None = None

    async def generate(self, request: ChatCompletionRequest) -> ProviderResponse:
        self.request = request
        return ProviderResponse("accepted")

    async def stream(self, request: ChatCompletionRequest):  # type: ignore[no-untyped-def]
        self.request = request
        yield "accepted"


@pytest.mark.asyncio
async def test_structured_output_repairs_twice_without_persisting_raw_output() -> None:
    provider = SequenceProvider(["not json", '{"answer": 7}', '{"answer": "valid"}'])
    app = create_app(settings(), provider=provider)
    payload = {
        "model": "portfolio-model",
        "messages": [{"role": "user", "content": "return JSON"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as test_client:
        response = await test_client.post("/v1/chat/completions", json=payload, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["validation_attempts"] == 3


@pytest.mark.asyncio
async def test_concurrent_budget_reservations_cannot_oversubscribe() -> None:
    store = InMemoryGatewayStore(Decimal("0.000100"))
    from llm_gateway.store import BudgetExceededError, Reservation

    results = await asyncio.gather(
        store.reserve(Reservation("one", "p", Decimal("0.000075"))),
        store.reserve(Reservation("two", "p", Decimal("0.000075"))),
        return_exceptions=True,
    )
    assert sum(isinstance(item, BudgetExceededError) for item in results) == 1


@pytest.mark.asyncio
async def test_token_bucket_is_atomic_under_concurrency() -> None:
    limiter = InMemoryTokenBucketLimiter(requests_per_minute=3, tokens_per_minute=30)
    results = await asyncio.gather(*(limiter.consume("project", 10, now=100.0) for _ in range(5)))
    assert sum(item.allowed for item in results) == 3


def test_json_schema_format_requires_a_schema() -> None:
    with pytest.raises(ValidationError, match="json_schema is required"):
        ResponseFormat.model_validate({"type": "json_schema"})


class TimeoutProvider:
    async def generate(self, request: ChatCompletionRequest) -> ProviderResponse:
        raise httpx.ReadTimeout("provider stalled")

    async def stream(self, request: ChatCompletionRequest):  # type: ignore[no-untyped-def]
        raise httpx.ReadTimeout("provider stalled")
        yield "unreachable"


@pytest.mark.asyncio
async def test_provider_timeout_is_explicit_and_releases_budget() -> None:
    store = InMemoryGatewayStore(Decimal("1.00"))
    app = create_app(settings(), provider=TimeoutProvider(), store=store)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as test_client:
        response = await test_client.post(
            "/v1/chat/completions",
            json={"model": "portfolio-model", "messages": [{"role": "user", "content": "hi"}]},
            headers=HEADERS,
        )
    assert response.status_code == 504
    assert response.json()["code"] == "provider_timeout"
    assert await store.budget_snapshot("portfolio-demo") == (Decimal("0"), Decimal("1.00"))


class DisconnectProvider:
    async def generate(self, request: ChatCompletionRequest) -> ProviderResponse:
        return ProviderResponse("unused")

    async def stream(self, request: ChatCompletionRequest):  # type: ignore[no-untyped-def]
        yield "first"
        yield "second"


@pytest.mark.asyncio
async def test_stream_disconnect_releases_reserved_budget() -> None:
    store = InMemoryGatewayStore(Decimal("1.00"))
    service = GatewayService(
        DisconnectProvider(),
        store,
        InMemoryTokenBucketLimiter(requests_per_minute=10, tokens_per_minute=10_000),
        PricingCatalog(),
    )
    request = ChatCompletionRequest(
        model="portfolio-model",
        messages=[{"role": "user", "content": "stream"}],
        stream=True,
        max_tokens=100,
    )
    stream = service.stream("portfolio", request, "trace")
    assert "first" in await anext(stream)
    await stream.aclose()
    assert await store.budget_snapshot("portfolio") == (Decimal("0"), Decimal("1.00"))


@pytest.mark.asyncio
async def test_pii_is_redacted_before_provider_and_audit_is_hash_only() -> None:
    provider = CapturingProvider()
    app = create_app(settings(), provider=provider)
    raw_pii = "person@example.com"
    payload = {
        "model": "portfolio-model",
        "messages": [{"role": "user", "content": f"Summarize for {raw_pii}"}],
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as test_client:
        response = await test_client.post(
            "/v1/chat/completions", json=payload, headers={**HEADERS, "Idempotency-Key": "pii"}
        )
        audit = await test_client.get("/v1/audit-events", headers=HEADERS)
    assert response.status_code == 200
    assert provider.request is not None
    assert raw_pii not in provider.request.messages[0].content
    assert "[REDACTED_EMAIL]" in provider.request.messages[0].content
    assert raw_pii not in audit.text
    assert [event["event_type"] for event in audit.json()] == ["policy.input", "policy.output"]
