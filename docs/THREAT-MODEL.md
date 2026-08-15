# Threat and failure model

## Protected assets

API keys, budget authority, usage records, webhook secrets, structured-output contracts, and trace correlation.

## Trust boundaries

- Untrusted callers cross bearer authentication, rate limits, and validation before reaching a provider.
- Provider output is untrusted and must pass schema and output-policy checks.
- Webhook destinations are untrusted network targets; HTTPS, DNS/IP checks, signatures, and no redirects reduce SSRF and spoofing risk.
- Redis coordinates distributed token buckets. PostgreSQL is the durable budget, usage, idempotency, webhook, and policy-audit authority in the Compose profile; the default single-process sandbox uses an in-memory store.

## Explicit limits

- Prompt-injection classification is heuristic and can be bypassed. It is one defense layer, not a security guarantee.
- The default runtime is a portfolio sandbox. It is not evidence of production capacity or availability.
- Deterministic CI exercises the in-memory adapters; explicit integration tests require disposable PostgreSQL and Redis services and do not prove failover behavior.
- Webhook signing secrets are stored in PostgreSQL for delivery. A real deployment should encrypt them with a managed key and rotate them; the local Compose profile does not claim KMS integration.
- Pricing entries are dated examples and must be revalidated before any financial decision.
