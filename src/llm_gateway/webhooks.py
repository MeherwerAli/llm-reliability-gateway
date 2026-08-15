import asyncio
import hashlib
import hmac
import ipaddress
import json
import random
import socket
import time
import uuid
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

from .models import WebhookDelivery, WebhookRecord
from .store import GatewayStore


class UnsafeWebhookURLError(ValueError):
    pass


def validate_webhook_url(url: str, *, allow_http: bool = False, resolve_dns: bool = True) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ({"https", "http"} if allow_http else {"https"}):
        raise UnsafeWebhookURLError("webhook URL must use HTTPS")
    if not parsed.hostname or parsed.username or parsed.password:
        raise UnsafeWebhookURLError("webhook URL host is invalid")
    if parsed.port not in (None, 443) and not allow_http:
        raise UnsafeWebhookURLError("non-standard ports are disabled")
    addresses: set[str] = set()
    try:
        addresses.add(str(ipaddress.ip_address(parsed.hostname)))
    except ValueError:
        if resolve_dns:
            try:
                addresses.update(
                    str(item[4][0])
                    for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443)
                )
            except socket.gaierror as error:
                raise UnsafeWebhookURLError("webhook host did not resolve") from error
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise UnsafeWebhookURLError("webhook URL resolves to a non-public address")


def sign_webhook(secret: str, timestamp: int, body: bytes) -> str:
    signed = f"{timestamp}.".encode() + body
    return hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()


def verify_webhook_signature(
    secret: str, timestamp: int, body: bytes, signature: str, *, tolerance: int = 300
) -> bool:
    if abs(int(time.time()) - timestamp) > tolerance:
        return False
    return hmac.compare_digest(sign_webhook(secret, timestamp, body), signature)


class WebhookDispatcher:
    def __init__(
        self,
        store: GatewayStore,
        client: httpx.AsyncClient | None = None,
        base_delay_seconds: float = 0.05,
    ) -> None:
        self.store = store
        self.client = client or httpx.AsyncClient(timeout=5.0, follow_redirects=False)
        self.base_delay_seconds = base_delay_seconds

    async def dispatch(self, project_id: str, event_type: str, payload: dict[str, Any]) -> None:
        event_id = f"evt_{uuid.uuid4().hex}"
        envelope = {
            "id": event_id,
            "type": event_type,
            "created_at": datetime.now(UTC).isoformat(),
            "data": payload,
        }
        body = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
        for record, secret in await self.store.webhook_targets(project_id, event_type):
            await self._deliver(record, secret, event_id, event_type, body)

    async def _deliver(
        self, record: WebhookRecord, secret: str, event_id: str, event_type: str, body: bytes
    ) -> None:
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            timestamp = int(time.time())
            try:
                response = await self.client.post(
                    record.url,
                    content=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-Webhook-Id": event_id,
                        "X-Webhook-Timestamp": str(timestamp),
                        "X-Webhook-Signature": sign_webhook(secret, timestamp, body),
                    },
                )
                delivered = 200 <= response.status_code < 300
                state: Literal["delivered", "retrying", "dead_letter"] = (
                    "delivered"
                    if delivered
                    else ("retrying" if attempt < max_attempts else "dead_letter")
                )
                await self.store.add_delivery(
                    WebhookDelivery(
                        id=f"whd_{uuid.uuid4().hex}",
                        webhook_id=record.id,
                        event_id=event_id,
                        event_type=event_type,
                        attempt=attempt,
                        state=state,
                        response_status=response.status_code,
                        created_at=datetime.now(UTC),
                    )
                )
                if delivered:
                    return
            except httpx.HTTPError as error:
                await self.store.add_delivery(
                    WebhookDelivery(
                        id=f"whd_{uuid.uuid4().hex}",
                        webhook_id=record.id,
                        event_id=event_id,
                        event_type=event_type,
                        attempt=attempt,
                        state="retrying" if attempt < max_attempts else "dead_letter",
                        error_code=error.__class__.__name__,
                        created_at=datetime.now(UTC),
                    )
                )
            if attempt < max_attempts:
                delay = min(1.0, self.base_delay_seconds * (2 ** (attempt - 1)))
                await asyncio.sleep(delay * random.uniform(0.5, 1.5))
