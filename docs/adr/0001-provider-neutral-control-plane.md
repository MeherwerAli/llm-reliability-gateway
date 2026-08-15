# ADR 0001: Keep the control plane provider-neutral

Status: accepted
Date: 2026-08-15

## Context

The gateway must demonstrate operational controls without tying budgets, validation, streaming, or audit behavior to one hosted model vendor.

## Decision

Expose an OpenAI-compatible HTTP surface and place model calls behind a small provider protocol. Ship a deterministic sandbox provider and an HTTP adapter that can target vLLM or another explicitly configured compatible server.

## Consequences

- Provider-specific pricing remains a versioned configuration concern.
- Deterministic tests do not prove compatibility with every provider.
- A release report must name the exact provider and model used for runtime verification.
