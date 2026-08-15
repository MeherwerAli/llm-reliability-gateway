import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from .models import (
    PolicyAuditEvent,
    UsageGroup,
    UsageRecord,
    UsageSummary,
    WebhookDelivery,
    WebhookRecord,
)


class BudgetExceededError(RuntimeError):
    pass


class IdempotencyConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class Reservation:
    id: str
    project_id: str
    reserved_usd: Decimal


@dataclass(frozen=True)
class IdempotencyEntry:
    request_hash: str
    response: dict[str, Any]


class GatewayStore(Protocol):
    monthly_budget_usd: Decimal

    async def budget_snapshot(self, project_id: str) -> tuple[Decimal, Decimal]: ...

    async def reserve(self, reservation: Reservation) -> None: ...

    async def reconcile(
        self, reservation_id: str, actual_usd: Decimal, usage: UsageRecord
    ) -> None: ...

    async def cancel_reservation(self, reservation_id: str) -> None: ...

    async def claim_budget_alert(self, project_id: str, threshold: Decimal) -> bool: ...

    async def get_idempotent(
        self, project_id: str, key: str, request_hash: str
    ) -> dict[str, Any] | None: ...

    async def put_idempotent(
        self, project_id: str, key: str, request_hash: str, response: dict[str, Any]
    ) -> None: ...

    async def usage_summary(self, project_id: str) -> UsageSummary: ...

    async def add_audit_event(self, event: PolicyAuditEvent) -> None: ...

    async def audit_events(self, project_id: str) -> list[PolicyAuditEvent]: ...

    async def create_webhook(self, record: WebhookRecord, secret: str) -> WebhookRecord: ...

    async def list_webhooks(self, project_id: str) -> list[WebhookRecord]: ...

    async def webhook_targets(
        self, project_id: str, event_type: str
    ) -> list[tuple[WebhookRecord, str]]: ...

    async def add_delivery(self, delivery: WebhookDelivery) -> None: ...

    async def deliveries(self, project_id: str) -> list[WebhookDelivery]: ...


def summarize_usage(project_id: str, records: list[UsageRecord]) -> UsageSummary:
    grouped: dict[tuple[str, str, str], list[UsageRecord]] = {}
    for record in records:
        key = (record.endpoint, record.model, record.created_at.strftime("%Y-%m"))
        grouped.setdefault(key, []).append(record)
    groups = [
        UsageGroup(
            project_id=project_id,
            endpoint=endpoint,
            model=model,
            period=period,
            request_count=len(items),
            prompt_tokens=sum(item.prompt_tokens for item in items),
            completion_tokens=sum(item.completion_tokens for item in items),
            actual_cost_usd=sum((item.actual_cost_usd for item in items), Decimal("0")),
        )
        for (endpoint, model, period), items in sorted(grouped.items())
    ]
    return UsageSummary(
        project_id=project_id,
        request_count=len(records),
        prompt_tokens=sum(item.prompt_tokens for item in records),
        completion_tokens=sum(item.completion_tokens for item in records),
        actual_cost_usd=sum((item.actual_cost_usd for item in records), Decimal("0")),
        records=records,
        groups=groups,
    )


class InMemoryGatewayStore:
    """Deterministic store used by tests and the zero-dependency sandbox profile."""

    def __init__(self, monthly_budget_usd: Decimal) -> None:
        self.monthly_budget_usd = monthly_budget_usd
        self._spent: dict[str, Decimal] = {}
        self._reserved: dict[str, Reservation] = {}
        self._usage: list[UsageRecord] = []
        self._audit_events: list[PolicyAuditEvent] = []
        self._idempotency: dict[tuple[str, str], IdempotencyEntry] = {}
        self._webhooks: dict[str, tuple[WebhookRecord, str]] = {}
        self._deliveries: list[WebhookDelivery] = []
        self._budget_alerted: set[str] = set()
        self._lock = asyncio.Lock()

    async def budget_snapshot(self, project_id: str) -> tuple[Decimal, Decimal]:
        async with self._lock:
            spent = self._spent.get(project_id, Decimal("0"))
            reserved = sum(
                (
                    item.reserved_usd
                    for item in self._reserved.values()
                    if item.project_id == project_id
                ),
                Decimal("0"),
            )
            return spent, self.monthly_budget_usd - spent - reserved

    async def reserve(self, reservation: Reservation) -> None:
        async with self._lock:
            spent = self._spent.get(reservation.project_id, Decimal("0"))
            reserved = sum(
                (
                    item.reserved_usd
                    for item in self._reserved.values()
                    if item.project_id == reservation.project_id
                ),
                Decimal("0"),
            )
            if spent + reserved + reservation.reserved_usd > self.monthly_budget_usd:
                raise BudgetExceededError("monthly budget would be exceeded")
            self._reserved[reservation.id] = reservation

    async def reconcile(self, reservation_id: str, actual_usd: Decimal, usage: UsageRecord) -> None:
        async with self._lock:
            reservation = self._reserved.pop(reservation_id)
            current = self._spent.get(reservation.project_id, Decimal("0"))
            if current + actual_usd > self.monthly_budget_usd:
                raise BudgetExceededError("actual usage exceeds monthly budget")
            self._spent[reservation.project_id] = current + actual_usd
            self._usage.append(usage)

    async def cancel_reservation(self, reservation_id: str) -> None:
        async with self._lock:
            self._reserved.pop(reservation_id, None)

    async def claim_budget_alert(self, project_id: str, threshold: Decimal) -> bool:
        async with self._lock:
            spent = self._spent.get(project_id, Decimal("0"))
            reached = self.monthly_budget_usd > 0 and spent / self.monthly_budget_usd >= threshold
            if not reached or project_id in self._budget_alerted:
                return False
            self._budget_alerted.add(project_id)
            return True

    async def get_idempotent(
        self, project_id: str, key: str, request_hash: str
    ) -> dict[str, Any] | None:
        async with self._lock:
            entry = self._idempotency.get((project_id, key))
            if entry is None:
                return None
            if entry.request_hash != request_hash:
                raise IdempotencyConflictError("idempotency key was reused with another payload")
            return entry.response

    async def put_idempotent(
        self, project_id: str, key: str, request_hash: str, response: dict[str, Any]
    ) -> None:
        async with self._lock:
            existing = self._idempotency.get((project_id, key))
            if existing is not None and existing.request_hash != request_hash:
                raise IdempotencyConflictError("idempotency key was concurrently reused")
            self._idempotency[(project_id, key)] = IdempotencyEntry(request_hash, response)

    async def usage_summary(self, project_id: str) -> UsageSummary:
        async with self._lock:
            records = [item for item in self._usage if item.project_id == project_id]
            return summarize_usage(project_id, records)

    async def add_audit_event(self, event: PolicyAuditEvent) -> None:
        async with self._lock:
            self._audit_events.append(event)

    async def audit_events(self, project_id: str) -> list[PolicyAuditEvent]:
        async with self._lock:
            return [event for event in self._audit_events if event.project_id == project_id]

    async def create_webhook(self, record: WebhookRecord, secret: str) -> WebhookRecord:
        async with self._lock:
            self._webhooks[record.id] = (record, secret)
            return record

    async def list_webhooks(self, project_id: str) -> list[WebhookRecord]:
        async with self._lock:
            return [
                record for record, _ in self._webhooks.values() if record.project_id == project_id
            ]

    async def webhook_targets(
        self, project_id: str, event_type: str
    ) -> list[tuple[WebhookRecord, str]]:
        async with self._lock:
            return [
                (record, secret)
                for record, secret in self._webhooks.values()
                if record.project_id == project_id and event_type in record.events
            ]

    async def add_delivery(self, delivery: WebhookDelivery) -> None:
        async with self._lock:
            self._deliveries.append(delivery)

    async def deliveries(self, project_id: str) -> list[WebhookDelivery]:
        webhook_ids = {item.id for item in await self.list_webhooks(project_id)}
        async with self._lock:
            return [item for item in self._deliveries if item.webhook_id in webhook_ids]

    @staticmethod
    def now() -> datetime:
        return datetime.now(UTC)
