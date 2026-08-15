import json
import os

import httpx


def main() -> None:
    base_url = os.getenv("GATEWAY_DEMO_URL", "http://127.0.0.1:8080")
    api_key = os.getenv("GATEWAY_DEMO_API_KEY", "portfolio-demo-key-change-me")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Idempotency-Key": "reliability-demo-v1",
        "X-Trace-Id": "demo-trace-v1",
    }
    payload = {
        "model": "portfolio-model",
        "messages": [
            {
                "role": "user",
                "content": "Explain bounded retry safety to demo.person@example.com",
            }
        ],
        "max_tokens": 128,
    }
    with httpx.Client(base_url=base_url, timeout=20) as client:
        estimate = client.post("/v1/estimates", json=payload, headers=headers)
        estimate.raise_for_status()
        completion = client.post("/v1/chat/completions", json=payload, headers=headers)
        completion.raise_for_status()
        replay = client.post("/v1/chat/completions", json=payload, headers=headers)
        replay.raise_for_status()
        usage = client.get("/v1/usage", headers=headers)
        usage.raise_for_status()
        audit = client.get("/v1/audit-events", headers=headers)
        audit.raise_for_status()
    print(
        json.dumps(
            {
                "estimate": estimate.json(),
                "completion": completion.json(),
                "replayed": replay.headers["X-Idempotent-Replay"],
                "usage_groups": usage.json()["groups"],
                "policy_audit": audit.json(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
