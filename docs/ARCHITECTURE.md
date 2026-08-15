# Architecture dossier

```mermaid
flowchart LR
    C["Bearer-authenticated client"] --> API["FastAPI OpenAI-compatible boundary"]
    API --> P["Input redaction and policy audit"]
    API --> R["Redis Lua request/token bucket"]
    API --> B["PostgreSQL budget reservation"]
    B --> V["Deterministic or OpenAI-compatible provider"]
    V --> S["Schema and output policy checks"]
    S --> U["Usage reconciliation and grouped cost"]
    U --> W["Signed webhook delivery"]
    API --> O["OpenTelemetry and Prometheus"]
    O --> T["Tempo and Grafana"]
```

The deterministic single-process profile substitutes in-memory state for PostgreSQL and Redis. It establishes contract behavior cheaply; the Compose profile establishes that the distributed adapters start and complete an authenticated request. Neither establishes failover or production capacity.
