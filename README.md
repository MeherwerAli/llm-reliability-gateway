# LLM Reliability Gateway

A failure-aware FastAPI control plane for model generation. It makes cost reservation, rate limiting, structured-output validation, signed webhooks, policy decisions, and trace correlation visible instead of hiding them behind a single SDK call.

This is a clean-room portfolio system with synthetic defaults. It contains no employer code, personal prompts, provider credentials, or production usage data.

## What it demonstrates

- OpenAI-compatible synchronous and SSE-streaming generation;
- versioned preflight cost estimates and transactional budget reservations;
- atomic request/token rate limiting with in-memory and Redis token-bucket implementations;
- Pydantic request contracts and JSON Schema response validation with two bounded repair attempts;
- idempotency replay and payload-conflict detection;
- HMAC-signed webhooks with replay-resistant timestamps and dead-letter state;
- PII redaction, input/output policy events, and explicit prompt-injection limitations;
- Prometheus metrics and OpenTelemetry/Tempo/Grafana deployment wiring.

## Run locally

```bash
uv sync --extra dev
uv run uvicorn llm_gateway.main:app --reload --port 8080
```

The default key is for local demonstration only:

```bash
curl -sS http://localhost:8080/v1/estimates \
  -H 'Authorization: Bearer portfolio-demo-key-change-me' \
  -H 'Content-Type: application/json' \
  -d '{"model":"portfolio-model","messages":[{"role":"user","content":"Explain idempotency."}],"max_tokens":128}'
```

For the distributed topology, copy `.env.example` to `.env`, change the API key, then run `docker compose up --build`. Compose selects the PostgreSQL state adapter and Redis Lua token bucket; a plain local process retains deterministic in-memory adapters unless `GATEWAY_STATE_BACKEND=distributed` is set.

## Public API

| Route | Behavior |
| --- | --- |
| `POST /v1/estimates` | Upper-bound cost and budget status from a dated catalog |
| `POST /v1/chat/completions` | Idempotent synchronous generation or resumable-friendly SSE events |
| `GET /v1/usage` | Usage and cost grouped by project, endpoint, model, and month |
| `GET /v1/audit-events` | Hash-only input/output policy decisions |
| `POST /v1/webhooks` | Validated, masked webhook registration |
| `GET /v1/webhook-deliveries` | Retry/delivery/dead-letter evidence |
| `GET /metrics` | Prometheus exposition |

## Verify

```bash
uv run ruff check .
uv run mypy
uv run pytest --cov=llm_gateway --cov-report=term-missing
docker compose config --quiet
docker build .
```

The core tests cover concurrent budget/rate-limit behavior, idempotency conflict, bounded schema repair, PII redaction before provider dispatch, timeout/disconnect cleanup, webhook tampering/replay/dead-letter behavior, and SSRF-shaped destinations. Opt-in integration tests exercise transactional PostgreSQL oversubscription protection, persisted usage/audit data, and the Redis Lua bucket. These tests do not establish Redis/PostgreSQL failover, external-provider compatibility, or production capacity.

## Architecture and limits

See [ADR 0001](docs/adr/0001-provider-neutral-control-plane.md) and the [threat model](docs/THREAT-MODEL.md). “Prompt-injection mitigation” is the intentional claim; universal prevention is not.
