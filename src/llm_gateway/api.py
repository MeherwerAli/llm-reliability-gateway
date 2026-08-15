import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any, cast

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from redis.asyncio import Redis

from .auth import APIKeyRegistry, APIPrincipal
from .config import Settings
from .models import (
    ChatCompletionRequest,
    ErrorBody,
    EstimateRequest,
    EstimateResponse,
    PolicyAuditEvent,
    UsageSummary,
    WebhookCreate,
    WebhookDelivery,
    WebhookRecord,
)
from .postgres_store import PostgresGatewayStore
from .pricing import PricingCatalog
from .providers import DeterministicProvider, OpenAICompatibleProvider, Provider
from .rate_limit import InMemoryTokenBucketLimiter, RateLimiter, RedisTokenBucketLimiter
from .service import (
    GatewayService,
    OutputPolicyError,
    RateLimitExceededError,
    StructuredOutputError,
)
from .store import BudgetExceededError, GatewayStore, IdempotencyConflictError, InMemoryGatewayStore
from .webhooks import UnsafeWebhookURLError, validate_webhook_url

REQUESTS = Counter("gateway_requests_total", "Gateway requests", ["path", "status"])
LATENCY = Histogram("gateway_request_duration_seconds", "Gateway request duration", ["path"])


def create_app(
    settings: Settings | None = None,
    *,
    provider: Provider | None = None,
    store: GatewayStore | None = None,
    limiter: RateLimiter | None = None,
) -> FastAPI:
    settings = settings or Settings()
    provider = provider or (
        DeterministicProvider()
        if settings.provider == "deterministic"
        else OpenAICompatibleProvider(
            settings.provider_base_url,
            settings.provider_api_key,
            settings.provider_timeout_seconds,
        )
    )
    if store is None:
        store = (
            PostgresGatewayStore(settings.database_url, settings.monthly_budget_usd)
            if settings.state_backend == "distributed"
            else InMemoryGatewayStore(settings.monthly_budget_usd)
        )
    redis_client: Redis | None = None
    if limiter is None:
        if settings.state_backend == "distributed":
            redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
            limiter = RedisTokenBucketLimiter(
                redis_client,
                settings.requests_per_minute,
                settings.tokens_per_minute,
            )
        else:
            limiter = InMemoryTokenBucketLimiter(
                settings.requests_per_minute, settings.tokens_per_minute
            )
    service = GatewayService(provider, store, limiter, PricingCatalog())
    registry = APIKeyRegistry(settings.api_key)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if isinstance(store, PostgresGatewayStore):
            await store.initialize()
        if redis_client is not None:
            await redis_client.ping()
        yield
        if redis_client is not None:
            await redis_client.aclose()
        if isinstance(store, PostgresGatewayStore):
            await store.close()

    app = FastAPI(
        title="LLM Reliability Gateway",
        version="0.1.0",
        description="Failure-aware OpenAI-compatible gateway. Policy controls reduce risk; they are not a universal security guarantee.",
        lifespan=lifespan,
    )
    app.state.gateway_service = service
    app.state.gateway_store = store

    @app.middleware("http")
    async def trace_and_measure(request: Request, call_next: Any) -> Response:
        import time

        started = time.perf_counter()
        trace_id = request.headers.get("X-Trace-Id") or uuid.uuid4().hex
        request.state.trace_id = trace_id
        path = request.url.path
        try:
            response = cast(Response, await call_next(request))
        except Exception:
            REQUESTS.labels(path=path, status="500").inc()
            raise
        response.headers["X-Trace-Id"] = trace_id
        REQUESTS.labels(path=path, status=str(response.status_code)).inc()
        LATENCY.labels(path=path).observe(time.perf_counter() - started)
        return response

    async def principal(
        authorization: Annotated[str | None, Header()] = None,
    ) -> APIPrincipal:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer API key")
        authenticated = registry.authenticate(authorization[7:])
        if authenticated is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")
        return authenticated

    @app.exception_handler(BudgetExceededError)
    async def budget_error(request: Request, error: BudgetExceededError) -> JSONResponse:
        return JSONResponse(
            status_code=402,
            content=ErrorBody(
                code="budget_exceeded", message=str(error), trace_id=request.state.trace_id
            ).model_dump(),
        )

    @app.exception_handler(IdempotencyConflictError)
    async def idempotency_error(request: Request, error: IdempotencyConflictError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content=ErrorBody(
                code="idempotency_conflict", message=str(error), trace_id=request.state.trace_id
            ).model_dump(),
        )

    @app.exception_handler(StructuredOutputError)
    async def structured_error(request: Request, error: StructuredOutputError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                **ErrorBody(
                    code="structured_output_invalid",
                    message=str(error),
                    trace_id=request.state.trace_id,
                ).model_dump(),
                "attempts": error.attempts,
            },
        )

    @app.exception_handler(httpx.TimeoutException)
    async def provider_timeout(request: Request, error: httpx.TimeoutException) -> JSONResponse:
        return JSONResponse(
            status_code=504,
            content=ErrorBody(
                code="provider_timeout",
                message="upstream provider timed out",
                trace_id=request.state.trace_id,
            ).model_dump(),
        )

    @app.exception_handler(httpx.HTTPError)
    async def provider_error(request: Request, error: httpx.HTTPError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content=ErrorBody(
                code="provider_error",
                message="upstream provider request failed",
                trace_id=request.state.trace_id,
            ).model_dump(),
        )

    @app.exception_handler(OutputPolicyError)
    async def output_policy_error(request: Request, error: OutputPolicyError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorBody(
                code="output_policy_block",
                message=str(error),
                trace_id=request.state.trace_id,
            ).model_dump(),
        )

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/estimates", response_model=EstimateResponse)
    async def estimate(
        request: EstimateRequest, identity: Annotated[APIPrincipal, Depends(principal)]
    ) -> EstimateResponse:
        try:
            return await service.estimate(identity.project_id, request)
        except ValueError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error

    @app.post("/v1/chat/completions")
    async def complete(
        request: ChatCompletionRequest,
        raw_request: Request,
        identity: Annotated[APIPrincipal, Depends(principal)],
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> Response:
        trace_id = raw_request.state.trace_id
        if request.stream:

            async def events() -> AsyncIterator[str]:
                try:
                    async for event in service.stream(identity.project_id, request, trace_id):
                        yield f"data: {event}\n\n"
                    yield "data: [DONE]\n\n"
                except asyncio.CancelledError:
                    raise
                except httpx.TimeoutException:
                    yield (
                        'data: {"type":"error","code":"provider_timeout",'
                        f'"trace_id":"{trace_id}"}}\n\n'
                    )
                    yield "data: [DONE]\n\n"
                except OutputPolicyError:
                    yield (
                        'data: {"type":"error","code":"output_policy_block",'
                        f'"trace_id":"{trace_id}"}}\n\n'
                    )
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
            )
        if not idempotency_key or len(idempotency_key) > 128:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "a bounded Idempotency-Key header is required"
            )
        try:
            result, decision, replayed = await service.complete(
                identity.project_id, request, idempotency_key, trace_id
            )
        except RateLimitExceededError as error:
            return JSONResponse(
                status_code=429,
                content=ErrorBody(
                    code="rate_limited", message=str(error), trace_id=trace_id
                ).model_dump(),
                headers={
                    "Retry-After": str(error.decision.retry_after_seconds),
                    "RateLimit-Remaining-Requests": str(error.decision.remaining_requests),
                    "RateLimit-Remaining-Tokens": str(error.decision.remaining_tokens),
                },
            )
        response = JSONResponse(result.model_dump(mode="json"))
        response.headers["X-Idempotent-Replay"] = str(replayed).lower()
        if decision.remaining_requests >= 0:
            response.headers["RateLimit-Remaining-Requests"] = str(decision.remaining_requests)
            response.headers["RateLimit-Remaining-Tokens"] = str(decision.remaining_tokens)
        return response

    @app.get("/v1/usage", response_model=UsageSummary)
    async def usage(identity: Annotated[APIPrincipal, Depends(principal)]) -> UsageSummary:
        return await store.usage_summary(identity.project_id)

    @app.get("/v1/audit-events", response_model=list[PolicyAuditEvent])
    async def audit_events(
        identity: Annotated[APIPrincipal, Depends(principal)],
    ) -> list[PolicyAuditEvent]:
        return await store.audit_events(identity.project_id)

    @app.post("/v1/webhooks", response_model=WebhookRecord, status_code=201)
    async def create_webhook(
        request: WebhookCreate,
        identity: Annotated[APIPrincipal, Depends(principal)],
    ) -> WebhookRecord:
        try:
            validate_webhook_url(
                str(request.url),
                allow_http=settings.allow_http_webhooks,
                resolve_dns=settings.env != "test",
            )
        except UnsafeWebhookURLError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error
        record = WebhookRecord(
            id=f"wh_{uuid.uuid4().hex}",
            project_id=identity.project_id,
            url=str(request.url),
            events=set(request.events),
            secret_masked=f"{request.secret[:4]}...{request.secret[-4:]}",
            created_at=datetime.now(UTC),
        )
        return await store.create_webhook(record, request.secret)

    @app.get("/v1/webhooks", response_model=list[WebhookRecord])
    async def webhooks(
        identity: Annotated[APIPrincipal, Depends(principal)],
    ) -> list[WebhookRecord]:
        return await store.list_webhooks(identity.project_id)

    @app.get("/v1/webhook-deliveries", response_model=list[WebhookDelivery])
    async def deliveries(
        identity: Annotated[APIPrincipal, Depends(principal)],
    ) -> list[WebhookDelivery]:
        return await store.deliveries(identity.project_id)

    FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())
    return app
