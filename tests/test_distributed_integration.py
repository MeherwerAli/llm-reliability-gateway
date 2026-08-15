import asyncio
import os
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from redis.asyncio import Redis

from llm_gateway.models import PolicyAuditEvent, UsageRecord
from llm_gateway.postgres_store import PostgresGatewayStore
from llm_gateway.rate_limit import RedisTokenBucketLimiter
from llm_gateway.store import BudgetExceededError, Reservation

DATABASE_URL = os.getenv("GATEWAY_INTEGRATION_DATABASE_URL")
REDIS_URL = os.getenv("GATEWAY_INTEGRATION_REDIS_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL or not REDIS_URL,
    reason="set disposable PostgreSQL and Redis integration URLs",
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgres_transactions_prevent_budget_oversubscription() -> None:
    assert DATABASE_URL is not None
    project = f"integration-{uuid.uuid4().hex}"
    store = PostgresGatewayStore(DATABASE_URL, Decimal("0.000100"))
    await store.initialize()
    reservations = [
        Reservation(f"res-{uuid.uuid4().hex}", project, Decimal("0.000075")) for _ in range(2)
    ]
    results = await asyncio.gather(
        *(store.reserve(reservation) for reservation in reservations),
        return_exceptions=True,
    )
    assert sum(isinstance(result, BudgetExceededError) for result in results) == 1
    for reservation in reservations:
        await store.cancel_reservation(reservation.id)

    usage_reservation = Reservation(f"res-{uuid.uuid4().hex}", project, Decimal("0.000050"))
    await store.reserve(usage_reservation)
    usage = UsageRecord(
        request_id=f"req-{uuid.uuid4().hex}",
        project_id=project,
        endpoint="/v1/chat/completions",
        model="portfolio-model",
        prompt_tokens=10,
        completion_tokens=5,
        actual_cost_usd=Decimal("0.000040"),
        trace_id="trace",
        created_at=datetime.now(UTC),
    )
    await store.reconcile(usage_reservation.id, usage.actual_cost_usd, usage)
    summary = await store.usage_summary(project)
    assert summary.request_count == 1
    assert summary.actual_cost_usd == Decimal("0.000040")

    event = PolicyAuditEvent(
        id=f"aud-{uuid.uuid4().hex}",
        project_id=project,
        event_type="policy.input",
        action="flag",
        reasons=["email"],
        content_sha256="a" * 64,
        trace_id="trace",
        created_at=datetime.now(UTC),
    )
    await store.add_audit_event(event)
    assert (await store.audit_events(project))[0].content_sha256 == "a" * 64
    await store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_lua_bucket_is_atomic_across_concurrent_requests() -> None:
    assert REDIS_URL is not None
    project = f"integration-{uuid.uuid4().hex}"
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    limiter = RedisTokenBucketLimiter(redis, requests_per_minute=3, tokens_per_minute=30)
    results = await asyncio.gather(*(limiter.consume(project, 10, now=1_000.0) for _ in range(5)))
    assert sum(result.allowed for result in results) == 3
    await redis.delete(
        f"gateway:rate:{project}:requests",
        f"gateway:rate:{project}:tokens",
    )
    await redis.aclose()
