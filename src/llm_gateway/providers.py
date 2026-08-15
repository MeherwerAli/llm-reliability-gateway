import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .models import ChatCompletionRequest


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class Provider(Protocol):
    async def generate(self, request: ChatCompletionRequest) -> ProviderResponse: ...

    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]: ...


def _schema_example(schema: dict[str, Any]) -> Any:
    schema_type = schema.get("type")
    if schema_type == "object":
        return {
            name: _schema_example(child)
            for name, child in schema.get("properties", {}).items()
            if name in schema.get("required", schema.get("properties", {}).keys())
        }
    if schema_type == "array":
        return [_schema_example(schema.get("items", {}))]
    if schema_type == "integer":
        return 1
    if schema_type == "number":
        return 1.0
    if schema_type == "boolean":
        return True
    return schema.get("default", "validated response")


class DeterministicProvider:
    async def generate(self, request: ChatCompletionRequest) -> ProviderResponse:
        if request.response_format.type == "json_schema" and request.response_format.json_schema:
            content = json.dumps(_schema_example(request.response_format.json_schema.schema_))
        else:
            user = next(
                (
                    message.content
                    for message in reversed(request.messages)
                    if message.role == "user"
                ),
                "",
            )
            content = f"Sandbox response: {user[:500]}"
        return ProviderResponse(text=content)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        response = await self.generate(request)
        words = response.text.split(" ")
        for index, word in enumerate(words):
            yield word + (" " if index < len(words) - 1 else "")


class OpenAICompatibleProvider:
    def __init__(self, base_url: str, api_key: str | None, timeout_seconds: float = 30.0) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_seconds
        )

    async def generate(self, request: ChatCompletionRequest) -> ProviderResponse:
        payload = request.model_dump(by_alias=True, exclude_none=True)
        payload["stream"] = False
        response = await self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        usage = body.get("usage", {})
        return ProviderResponse(
            text=body["choices"][0]["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        payload = request.model_dump(by_alias=True, exclude_none=True)
        payload["stream"] = True
        async with self._client.stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                body = json.loads(line[6:])
                content = body.get("choices", [{}])[0].get("delta", {}).get("content")
                if content:
                    yield content
