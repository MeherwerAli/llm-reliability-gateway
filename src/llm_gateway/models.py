from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class EstimateRequest(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    messages: list[Message] = Field(min_length=1, max_length=100)
    max_tokens: int = Field(default=256, ge=1, le=8192)


class EstimateResponse(BaseModel):
    prompt_tokens: int
    reserved_output_tokens: int
    estimated_cost_usd: Decimal
    price_version: str
    price_effective_at: str
    budget_status: Literal["allowed", "warning", "exhausted"]
    budget_remaining_usd: Decimal


class JsonSchemaDefinition(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool = True

    model_config = ConfigDict(populate_by_name=True)


class ResponseFormat(BaseModel):
    type: Literal["text", "json_schema"] = "text"
    json_schema: JsonSchemaDefinition | None = None

    @model_validator(mode="after")
    def schema_required_for_json(self) -> "ResponseFormat":
        if self.type == "json_schema" and self.json_schema is None:
            raise ValueError("json_schema is required when type is json_schema")
        return self


class ChatCompletionRequest(EstimateRequest):
    stream: bool = False
    temperature: float = Field(default=0.0, ge=0, le=2)
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    actual_cost_usd: Decimal


class ChatMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "content_filter"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage
    trace_id: str
    validation_attempts: int = 1


class UsageRecord(BaseModel):
    request_id: str
    project_id: str
    endpoint: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    actual_cost_usd: Decimal
    trace_id: str
    created_at: datetime


class UsageGroup(BaseModel):
    project_id: str
    endpoint: str
    model: str
    period: str
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    actual_cost_usd: Decimal


class UsageSummary(BaseModel):
    project_id: str
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    actual_cost_usd: Decimal
    records: list[UsageRecord]
    groups: list[UsageGroup]


class PolicyAuditEvent(BaseModel):
    id: str
    project_id: str
    event_type: Literal["policy.input", "policy.output"]
    action: Literal["allow", "flag", "block"]
    reasons: list[str]
    content_sha256: str
    trace_id: str
    created_at: datetime


class WebhookCreate(BaseModel):
    url: HttpUrl
    events: set[Literal["generation.completed", "generation.failed", "budget.threshold"]]
    secret: str = Field(min_length=24, max_length=256)


class WebhookRecord(BaseModel):
    id: str
    project_id: str
    url: str
    events: set[str]
    secret_masked: str
    created_at: datetime


class WebhookDelivery(BaseModel):
    id: str
    webhook_id: str
    event_id: str
    event_type: str
    attempt: int
    state: Literal["pending", "delivered", "retrying", "dead_letter"]
    response_status: int | None = None
    error_code: str | None = None
    created_at: datetime


class ErrorBody(BaseModel):
    code: str
    message: str
    trace_id: str | None = None
