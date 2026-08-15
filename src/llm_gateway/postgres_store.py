import json
from decimal import Decimal
from typing import Any, cast

import asyncpg

from .models import (
    PolicyAuditEvent,
    UsageRecord,
    UsageSummary,
    WebhookDelivery,
    WebhookRecord,
)
from .store import (
    BudgetExceededError,
    IdempotencyConflictError,
    Reservation,
    summarize_usage,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS budget_spend (
  project_id TEXT PRIMARY KEY,
  spent_usd NUMERIC(20, 10) NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS budget_reservations (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  reserved_usd NUMERIC(20, 10) NOT NULL CHECK (reserved_usd >= 0)
);
CREATE INDEX IF NOT EXISTS budget_reservations_project_idx
  ON budget_reservations(project_id);
CREATE TABLE IF NOT EXISTS usage_records (
  request_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  model TEXT NOT NULL,
  prompt_tokens INTEGER NOT NULL,
  completion_tokens INTEGER NOT NULL,
  actual_cost_usd NUMERIC(20, 10) NOT NULL,
  trace_id TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_records_project_idx ON usage_records(project_id, created_at);
CREATE TABLE IF NOT EXISTS idempotency_entries (
  project_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  request_hash TEXT NOT NULL,
  response_json JSONB NOT NULL,
  PRIMARY KEY(project_id, idempotency_key)
);
CREATE TABLE IF NOT EXISTS policy_audit_events (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  action TEXT NOT NULL,
  reasons_json JSONB NOT NULL,
  content_sha256 TEXT NOT NULL,
  trace_id TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS policy_audit_project_idx
  ON policy_audit_events(project_id, created_at);
CREATE TABLE IF NOT EXISTS webhooks (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  url TEXT NOT NULL,
  events_json JSONB NOT NULL,
  secret TEXT NOT NULL,
  secret_masked TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS webhook_deliveries (
  id TEXT PRIMARY KEY,
  webhook_id TEXT NOT NULL REFERENCES webhooks(id),
  event_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  attempt INTEGER NOT NULL,
  state TEXT NOT NULL,
  response_status INTEGER,
  error_code TEXT,
  created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS webhook_delivery_webhook_idx
  ON webhook_deliveries(webhook_id, created_at);
CREATE TABLE IF NOT EXISTS budget_alerts (
  project_id TEXT NOT NULL,
  threshold NUMERIC(4, 3) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY(project_id, threshold)
);
"""


class PostgresGatewayStore:
    """Transactional state adapter used by the distributed Docker profile."""

    def __init__(self, database_url: str, monthly_budget_usd: Decimal) -> None:
        self.database_url = database_url
        self.monthly_budget_usd = monthly_budget_usd
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        self._pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=5)
        async with self._pool.acquire() as connection:
            await connection.execute(SCHEMA)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()

    def _ready(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgreSQL store is not initialized")
        return self._pool

    @staticmethod
    async def _project_lock(connection: asyncpg.Connection, project_id: str) -> None:
        await connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", project_id
        )

    async def budget_snapshot(self, project_id: str) -> tuple[Decimal, Decimal]:
        async with self._ready().acquire() as connection:
            spent = Decimal(
                str(
                    await connection.fetchval(
                        "SELECT COALESCE(spent_usd, 0) FROM budget_spend WHERE project_id = $1",
                        project_id,
                    )
                    or 0
                )
            )
            reserved = Decimal(
                str(
                    await connection.fetchval(
                        "SELECT COALESCE(SUM(reserved_usd), 0) FROM budget_reservations WHERE project_id = $1",
                        project_id,
                    )
                    or 0
                )
            )
        return spent, self.monthly_budget_usd - spent - reserved

    async def reserve(self, reservation: Reservation) -> None:
        async with self._ready().acquire() as connection, connection.transaction():
            await self._project_lock(connection, reservation.project_id)
            spent = Decimal(
                str(
                    await connection.fetchval(
                        "SELECT COALESCE(spent_usd, 0) FROM budget_spend WHERE project_id = $1",
                        reservation.project_id,
                    )
                    or 0
                )
            )
            reserved = Decimal(
                str(
                    await connection.fetchval(
                        "SELECT COALESCE(SUM(reserved_usd), 0) FROM budget_reservations WHERE project_id = $1",
                        reservation.project_id,
                    )
                    or 0
                )
            )
            if spent + reserved + reservation.reserved_usd > self.monthly_budget_usd:
                raise BudgetExceededError("monthly budget would be exceeded")
            await connection.execute(
                "INSERT INTO budget_reservations(id, project_id, reserved_usd) VALUES ($1, $2, $3)",
                reservation.id,
                reservation.project_id,
                reservation.reserved_usd,
            )

    async def reconcile(self, reservation_id: str, actual_usd: Decimal, usage: UsageRecord) -> None:
        async with self._ready().acquire() as connection, connection.transaction():
            reservation = await connection.fetchrow(
                "SELECT project_id FROM budget_reservations WHERE id = $1 FOR UPDATE",
                reservation_id,
            )
            if reservation is None:
                raise KeyError(reservation_id)
            project_id = str(reservation["project_id"])
            await self._project_lock(connection, project_id)
            spent = Decimal(
                str(
                    await connection.fetchval(
                        "SELECT COALESCE(spent_usd, 0) FROM budget_spend WHERE project_id = $1",
                        project_id,
                    )
                    or 0
                )
            )
            if spent + actual_usd > self.monthly_budget_usd:
                raise BudgetExceededError("actual usage exceeds monthly budget")
            await connection.execute(
                """
                INSERT INTO budget_spend(project_id, spent_usd) VALUES ($1, $2)
                ON CONFLICT(project_id) DO UPDATE SET spent_usd = EXCLUDED.spent_usd
                """,
                project_id,
                spent + actual_usd,
            )
            await connection.execute(
                "DELETE FROM budget_reservations WHERE id = $1", reservation_id
            )
            await connection.execute(
                """
                INSERT INTO usage_records(
                  request_id, project_id, endpoint, model, prompt_tokens,
                  completion_tokens, actual_cost_usd, trace_id, created_at
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                """,
                usage.request_id,
                usage.project_id,
                usage.endpoint,
                usage.model,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.actual_cost_usd,
                usage.trace_id,
                usage.created_at,
            )

    async def cancel_reservation(self, reservation_id: str) -> None:
        async with self._ready().acquire() as connection:
            await connection.execute(
                "DELETE FROM budget_reservations WHERE id = $1", reservation_id
            )

    async def claim_budget_alert(self, project_id: str, threshold: Decimal) -> bool:
        async with self._ready().acquire() as connection, connection.transaction():
            await self._project_lock(connection, project_id)
            spent = Decimal(
                str(
                    await connection.fetchval(
                        "SELECT COALESCE(spent_usd, 0) FROM budget_spend WHERE project_id = $1",
                        project_id,
                    )
                    or 0
                )
            )
            if self.monthly_budget_usd <= 0 or spent / self.monthly_budget_usd < threshold:
                return False
            row = await connection.fetchrow(
                """
                INSERT INTO budget_alerts(project_id, threshold) VALUES ($1, $2)
                ON CONFLICT DO NOTHING RETURNING project_id
                """,
                project_id,
                threshold,
            )
            return row is not None

    async def get_idempotent(
        self, project_id: str, key: str, request_hash: str
    ) -> dict[str, Any] | None:
        async with self._ready().acquire() as connection:
            row = await connection.fetchrow(
                "SELECT request_hash, response_json FROM idempotency_entries WHERE project_id = $1 AND idempotency_key = $2",
                project_id,
                key,
            )
        if row is None:
            return None
        if row["request_hash"] != request_hash:
            raise IdempotencyConflictError("idempotency key was reused with another payload")
        response = row["response_json"]
        return cast(dict[str, Any], json.loads(response) if isinstance(response, str) else response)

    async def put_idempotent(
        self, project_id: str, key: str, request_hash: str, response: dict[str, Any]
    ) -> None:
        async with self._ready().acquire() as connection:
            result = await connection.execute(
                """
                INSERT INTO idempotency_entries(project_id, idempotency_key, request_hash, response_json)
                VALUES ($1, $2, $3, $4::jsonb) ON CONFLICT DO NOTHING
                """,
                project_id,
                key,
                request_hash,
                json.dumps(response),
            )
            if result == "INSERT 0 0":
                existing = await connection.fetchval(
                    "SELECT request_hash FROM idempotency_entries WHERE project_id = $1 AND idempotency_key = $2",
                    project_id,
                    key,
                )
                if existing != request_hash:
                    raise IdempotencyConflictError("idempotency key was concurrently reused")

    async def usage_summary(self, project_id: str) -> UsageSummary:
        async with self._ready().acquire() as connection:
            rows = await connection.fetch(
                "SELECT * FROM usage_records WHERE project_id = $1 ORDER BY created_at",
                project_id,
            )
        records = [
            UsageRecord(
                request_id=row["request_id"],
                project_id=row["project_id"],
                endpoint=row["endpoint"],
                model=row["model"],
                prompt_tokens=row["prompt_tokens"],
                completion_tokens=row["completion_tokens"],
                actual_cost_usd=row["actual_cost_usd"],
                trace_id=row["trace_id"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
        return summarize_usage(project_id, records)

    async def add_audit_event(self, event: PolicyAuditEvent) -> None:
        async with self._ready().acquire() as connection:
            await connection.execute(
                """
                INSERT INTO policy_audit_events(
                  id, project_id, event_type, action, reasons_json,
                  content_sha256, trace_id, created_at
                ) VALUES ($1,$2,$3,$4,$5::jsonb,$6,$7,$8)
                """,
                event.id,
                event.project_id,
                event.event_type,
                event.action,
                json.dumps(event.reasons),
                event.content_sha256,
                event.trace_id,
                event.created_at,
            )

    async def audit_events(self, project_id: str) -> list[PolicyAuditEvent]:
        async with self._ready().acquire() as connection:
            rows = await connection.fetch(
                "SELECT * FROM policy_audit_events WHERE project_id = $1 ORDER BY created_at",
                project_id,
            )
        return [
            PolicyAuditEvent(
                id=row["id"],
                project_id=row["project_id"],
                event_type=row["event_type"],
                action=row["action"],
                reasons=json.loads(row["reasons_json"])
                if isinstance(row["reasons_json"], str)
                else row["reasons_json"],
                content_sha256=row["content_sha256"],
                trace_id=row["trace_id"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def create_webhook(self, record: WebhookRecord, secret: str) -> WebhookRecord:
        async with self._ready().acquire() as connection:
            await connection.execute(
                """
                INSERT INTO webhooks(id, project_id, url, events_json, secret, secret_masked, created_at)
                VALUES ($1,$2,$3,$4::jsonb,$5,$6,$7)
                """,
                record.id,
                record.project_id,
                record.url,
                json.dumps(sorted(record.events)),
                secret,
                record.secret_masked,
                record.created_at,
            )
        return record

    @staticmethod
    def _webhook(row: asyncpg.Record) -> tuple[WebhookRecord, str]:
        events = row["events_json"]
        return (
            WebhookRecord(
                id=row["id"],
                project_id=row["project_id"],
                url=row["url"],
                events=set(json.loads(events) if isinstance(events, str) else events),
                secret_masked=row["secret_masked"],
                created_at=row["created_at"],
            ),
            str(row["secret"]),
        )

    async def list_webhooks(self, project_id: str) -> list[WebhookRecord]:
        async with self._ready().acquire() as connection:
            rows = await connection.fetch(
                "SELECT * FROM webhooks WHERE project_id = $1 ORDER BY created_at", project_id
            )
        return [self._webhook(row)[0] for row in rows]

    async def webhook_targets(
        self, project_id: str, event_type: str
    ) -> list[tuple[WebhookRecord, str]]:
        async with self._ready().acquire() as connection:
            rows = await connection.fetch(
                "SELECT * FROM webhooks WHERE project_id = $1 AND events_json ? $2",
                project_id,
                event_type,
            )
        return [self._webhook(row) for row in rows]

    async def add_delivery(self, delivery: WebhookDelivery) -> None:
        async with self._ready().acquire() as connection:
            await connection.execute(
                """
                INSERT INTO webhook_deliveries(
                  id, webhook_id, event_id, event_type, attempt, state,
                  response_status, error_code, created_at
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                """,
                delivery.id,
                delivery.webhook_id,
                delivery.event_id,
                delivery.event_type,
                delivery.attempt,
                delivery.state,
                delivery.response_status,
                delivery.error_code,
                delivery.created_at,
            )

    async def deliveries(self, project_id: str) -> list[WebhookDelivery]:
        async with self._ready().acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT delivery.* FROM webhook_deliveries delivery
                JOIN webhooks webhook ON webhook.id = delivery.webhook_id
                WHERE webhook.project_id = $1 ORDER BY delivery.created_at
                """,
                project_id,
            )
        return [
            WebhookDelivery(
                id=row["id"],
                webhook_id=row["webhook_id"],
                event_id=row["event_id"],
                event_type=row["event_type"],
                attempt=row["attempt"],
                state=row["state"],
                response_status=row["response_status"],
                error_code=row["error_code"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
