import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

from jsonschema import ValidationError, validate

from .models import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatUsage,
    EstimateRequest,
    EstimateResponse,
    Message,
    PolicyAuditEvent,
    UsageRecord,
)
from .policy import assess_input, assess_output
from .pricing import PricingCatalog
from .providers import Provider
from .rate_limit import RateLimitDecision, RateLimiter
from .store import GatewayStore, Reservation
from .webhooks import WebhookDispatcher


class RateLimitExceededError(RuntimeError):
    def __init__(self, decision: RateLimitDecision) -> None:
        super().__init__("rate limit exceeded")
        self.decision = decision


class StructuredOutputError(RuntimeError):
    def __init__(self, attempts: list[dict[str, Any]]) -> None:
        super().__init__("provider did not return valid structured output")
        self.attempts = attempts


class OutputPolicyError(RuntimeError):
    pass


class GatewayService:
    def __init__(
        self,
        provider: Provider,
        store: GatewayStore,
        limiter: RateLimiter,
        pricing: PricingCatalog | None = None,
        webhooks: WebhookDispatcher | None = None,
    ) -> None:
        self.provider = provider
        self.store = store
        self.limiter = limiter
        self.pricing = pricing or PricingCatalog()
        self.webhooks = webhooks or WebhookDispatcher(store)

    async def _sanitize_request(
        self,
        project_id: str,
        request: ChatCompletionRequest,
        trace_id: str,
    ) -> ChatCompletionRequest:
        messages: list[Message] = []
        for message in request.messages:
            decision = assess_input(message.content)
            await self.store.add_audit_event(
                PolicyAuditEvent(
                    id=f"aud_{uuid.uuid4().hex}",
                    project_id=project_id,
                    event_type="policy.input",
                    action=decision.action,
                    reasons=list(decision.reasons),
                    content_sha256=hashlib.sha256(message.content.encode()).hexdigest(),
                    trace_id=trace_id,
                    created_at=datetime.now(UTC),
                )
            )
            messages.append(Message(role=message.role, content=decision.redacted_text))
        return request.model_copy(update={"messages": messages})

    async def _audit_output(
        self,
        project_id: str,
        text: str,
        trace_id: str,
    ) -> None:
        decision = assess_output(text)
        await self.store.add_audit_event(
            PolicyAuditEvent(
                id=f"aud_{uuid.uuid4().hex}",
                project_id=project_id,
                event_type="policy.output",
                action=decision.action,
                reasons=list(decision.reasons),
                content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                trace_id=trace_id,
                created_at=datetime.now(UTC),
            )
        )

    async def estimate(self, project_id: str, request: EstimateRequest) -> EstimateResponse:
        prompt_tokens = self.pricing.estimate_prompt_tokens(request.messages)
        estimated = self.pricing.cost(request.model, prompt_tokens, request.max_tokens)
        spent, remaining = await self.store.budget_snapshot(project_id)
        ratio = (
            (spent + estimated) / self.store.monthly_budget_usd
            if self.store.monthly_budget_usd
            else Decimal("1")
        )
        status: Literal["allowed", "warning", "exhausted"] = (
            "exhausted"
            if estimated > remaining
            else ("warning" if ratio >= Decimal("0.8") else "allowed")
        )
        return EstimateResponse(
            prompt_tokens=prompt_tokens,
            reserved_output_tokens=request.max_tokens,
            estimated_cost_usd=estimated,
            price_version=self.pricing.version,
            price_effective_at=self.pricing.effective_at,
            budget_status=status,
            budget_remaining_usd=max(Decimal("0"), remaining),
        )

    @staticmethod
    def request_hash(request: ChatCompletionRequest) -> str:
        payload = request.model_dump_json(by_alias=True, exclude_none=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    async def complete(
        self,
        project_id: str,
        request: ChatCompletionRequest,
        idempotency_key: str,
        trace_id: str,
    ) -> tuple[ChatCompletionResponse, RateLimitDecision, bool]:
        request_hash = self.request_hash(request)
        cached = await self.store.get_idempotent(project_id, idempotency_key, request_hash)
        if cached is not None:
            response = ChatCompletionResponse.model_validate(cached)
            return response, RateLimitDecision(True, -1, -1, 0), True

        estimate = await self.estimate(project_id, request)
        if estimate.budget_status == "exhausted":
            from .store import BudgetExceededError

            raise BudgetExceededError("estimated request exceeds remaining budget")
        limit = await self.limiter.consume(project_id, estimate.prompt_tokens + request.max_tokens)
        if not limit.allowed:
            raise RateLimitExceededError(limit)

        provider_request = await self._sanitize_request(project_id, request, trace_id)
        reservation_id = f"res_{uuid.uuid4().hex}"
        await self.store.reserve(
            Reservation(reservation_id, project_id, estimate.estimated_cost_usd)
        )
        attempts: list[dict[str, Any]] = []
        try:
            provider_response = None
            for attempt in range(1, 4):
                provider_response = await self.provider.generate(provider_request)
                if request.response_format.type != "json_schema":
                    break
                try:
                    parsed = json.loads(provider_response.text)
                    definition = request.response_format.json_schema
                    assert definition is not None
                    validate(parsed, definition.schema_)
                    break
                except (json.JSONDecodeError, ValidationError) as error:
                    attempts.append(
                        {
                            "attempt": attempt,
                            "error": error.__class__.__name__,
                            "raw_output_sha256": hashlib.sha256(
                                provider_response.text.encode()
                            ).hexdigest(),
                        }
                    )
                    if attempt == 3:
                        raise StructuredOutputError(attempts) from error
            assert provider_response is not None
            output_decision = assess_output(provider_response.text)
            await self._audit_output(project_id, provider_response.text, trace_id)
            if output_decision.action == "block":
                raise OutputPolicyError("provider output violated secret-exposure policy")
            prompt_tokens = provider_response.prompt_tokens or estimate.prompt_tokens
            completion_tokens = (
                provider_response.completion_tokens
                or self.pricing.estimate_text_tokens(provider_response.text)
            )
            actual_cost = self.pricing.cost(request.model, prompt_tokens, completion_tokens)
            request_id = f"chatcmpl_{uuid.uuid4().hex}"
            response = ChatCompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[ChatChoice(message=ChatMessage(content=output_decision.redacted_text))],
                usage=ChatUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    actual_cost_usd=actual_cost,
                ),
                trace_id=trace_id,
                validation_attempts=len(attempts) + 1,
            )
            usage = UsageRecord(
                request_id=request_id,
                project_id=project_id,
                endpoint="/v1/chat/completions",
                model=request.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                actual_cost_usd=actual_cost,
                trace_id=trace_id,
                created_at=datetime.now(UTC),
            )
            await self.store.reconcile(reservation_id, actual_cost, usage)
            dumped = response.model_dump(mode="json")
            await self.store.put_idempotent(project_id, idempotency_key, request_hash, dumped)
            if await self.store.claim_budget_alert(project_id, Decimal("0.8")):
                await self.webhooks.dispatch(
                    project_id,
                    "budget.threshold",
                    {"threshold": "0.8", "trace_id": trace_id},
                )
            await self.webhooks.dispatch(
                project_id, "generation.completed", {"request_id": request_id, "trace_id": trace_id}
            )
            return response, limit, False
        except (asyncio.CancelledError, GeneratorExit):
            await self.store.cancel_reservation(reservation_id)
            raise
        except Exception:
            await self.store.cancel_reservation(reservation_id)
            raise

    async def stream(
        self, project_id: str, request: ChatCompletionRequest, trace_id: str
    ) -> AsyncIterator[str]:
        estimate = await self.estimate(project_id, request)
        if request.response_format.type == "json_schema":
            raise StructuredOutputError(
                [{"attempt": 0, "error": "streaming_json_schema_not_supported"}]
            )
        limit = await self.limiter.consume(project_id, estimate.prompt_tokens + request.max_tokens)
        if not limit.allowed:
            raise RateLimitExceededError(limit)
        reservation_id = f"res_{uuid.uuid4().hex}"
        await self.store.reserve(
            Reservation(reservation_id, project_id, estimate.estimated_cost_usd)
        )
        content = ""
        provider_request = await self._sanitize_request(project_id, request, trace_id)
        try:
            async for token in self.provider.stream(provider_request):
                content += token
                output_decision = assess_output(content)
                if output_decision.action == "block":
                    await self._audit_output(project_id, content, trace_id)
                    raise OutputPolicyError("streaming output violated secret-exposure policy")
                yield json.dumps({"type": "token", "content": token, "trace_id": trace_id})
            await self._audit_output(project_id, content, trace_id)
            completion_tokens = self.pricing.estimate_text_tokens(content)
            actual = self.pricing.cost(request.model, estimate.prompt_tokens, completion_tokens)
            request_id = f"chatcmpl_{uuid.uuid4().hex}"
            await self.store.reconcile(
                reservation_id,
                actual,
                UsageRecord(
                    request_id=request_id,
                    project_id=project_id,
                    endpoint="/v1/chat/completions",
                    model=request.model,
                    prompt_tokens=estimate.prompt_tokens,
                    completion_tokens=completion_tokens,
                    actual_cost_usd=actual,
                    trace_id=trace_id,
                    created_at=datetime.now(UTC),
                ),
            )
            yield json.dumps({"type": "done", "request_id": request_id, "trace_id": trace_id})
        except (asyncio.CancelledError, GeneratorExit):
            await self.store.cancel_reservation(reservation_id)
            raise
        except Exception:
            await self.store.cancel_reservation(reservation_id)
            raise
