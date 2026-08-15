import asyncio
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Protocol, cast

from redis.asyncio import Redis


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining_requests: int
    remaining_tokens: int
    retry_after_seconds: int


class RateLimiter(Protocol):
    async def consume(
        self, key: str, tokens: int, now: float | None = None
    ) -> RateLimitDecision: ...


class InMemoryTokenBucketLimiter:
    def __init__(self, requests_per_minute: int, tokens_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._state: dict[str, tuple[float, float, float]] = {}
        self._lock = asyncio.Lock()

    async def consume(self, key: str, tokens: int, now: float | None = None) -> RateLimitDecision:
        current = time.monotonic() if now is None else now
        async with self._lock:
            request_balance, token_balance, updated = self._state.get(
                key, (float(self.requests_per_minute), float(self.tokens_per_minute), current)
            )
            elapsed = max(0.0, current - updated)
            request_balance = min(
                self.requests_per_minute,
                request_balance + elapsed * self.requests_per_minute / 60,
            )
            token_balance = min(
                self.tokens_per_minute,
                token_balance + elapsed * self.tokens_per_minute / 60,
            )
            allowed = request_balance >= 1 and token_balance >= tokens
            if allowed:
                request_balance -= 1
                token_balance -= tokens
            self._state[key] = (request_balance, token_balance, current)
            request_wait = (
                0
                if request_balance >= 1
                else int((1 - request_balance) * 60 / self.requests_per_minute) + 1
            )
            token_wait = (
                0
                if token_balance >= tokens
                else int((tokens - token_balance) * 60 / self.tokens_per_minute) + 1
            )
            return RateLimitDecision(
                allowed=allowed,
                remaining_requests=max(0, int(request_balance)),
                remaining_tokens=max(0, int(token_balance)),
                retry_after_seconds=max(request_wait, token_wait),
            )


class RedisTokenBucketLimiter:
    _SCRIPT = """
local req_key = KEYS[1]
local tok_key = KEYS[2]
local now = tonumber(ARGV[1])
local req_cap = tonumber(ARGV[2])
local tok_cap = tonumber(ARGV[3])
local tokens = tonumber(ARGV[4])
local req = tonumber(redis.call('HGET', req_key, 'balance') or req_cap)
local req_at = tonumber(redis.call('HGET', req_key, 'updated') or now)
local tok = tonumber(redis.call('HGET', tok_key, 'balance') or tok_cap)
local tok_at = tonumber(redis.call('HGET', tok_key, 'updated') or now)
req = math.min(req_cap, req + math.max(0, now - req_at) * req_cap / 60)
tok = math.min(tok_cap, tok + math.max(0, now - tok_at) * tok_cap / 60)
local allowed = 0
if req >= 1 and tok >= tokens then
  allowed = 1
  req = req - 1
  tok = tok - tokens
end
redis.call('HSET', req_key, 'balance', req, 'updated', now)
redis.call('HSET', tok_key, 'balance', tok, 'updated', now)
redis.call('EXPIRE', req_key, 120)
redis.call('EXPIRE', tok_key, 120)
return {allowed, math.floor(req), math.floor(tok)}
"""

    def __init__(self, redis: Redis, requests_per_minute: int, tokens_per_minute: int) -> None:
        self.redis = redis
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute

    async def consume(self, key: str, tokens: int, now: float | None = None) -> RateLimitDecision:
        current = time.time() if now is None else now
        pending = self.redis.eval(
            self._SCRIPT,
            2,
            f"gateway:rate:{key}:requests",
            f"gateway:rate:{key}:tokens",
            str(current),
            str(self.requests_per_minute),
            str(self.tokens_per_minute),
            str(tokens),
        )
        raw = await cast(Awaitable[Any], pending)
        allowed, requests_left, tokens_left = (int(value) for value in raw)
        retry = 0 if allowed else max(1, int(tokens * 60 / self.tokens_per_minute))
        return RateLimitDecision(bool(allowed), requests_left, tokens_left, retry)
