import re
from dataclasses import dataclass
from decimal import ROUND_UP, Decimal

from .models import Message


@dataclass(frozen=True)
class ModelPrice:
    input_per_million_usd: Decimal
    output_per_million_usd: Decimal


class PricingCatalog:
    """Versioned demo catalog. Real provider prices must be revalidated before release."""

    version = "portfolio-demo-2026-08-15"
    effective_at = "2026-08-15"

    def __init__(self) -> None:
        self._prices = {
            "portfolio-model": ModelPrice(Decimal("1.00"), Decimal("2.00")),
            "Qwen/Qwen2.5-7B-Instruct": ModelPrice(Decimal("0"), Decimal("0")),
            "Qwen/Qwen2.5-7B-Instruct-AWQ": ModelPrice(Decimal("0"), Decimal("0")),
        }

    def price_for(self, model: str) -> ModelPrice:
        try:
            return self._prices[model]
        except KeyError as error:
            raise ValueError(
                f"model {model!r} is absent from pricing catalog {self.version}"
            ) from error

    @staticmethod
    def estimate_prompt_tokens(messages: list[Message]) -> int:
        pieces = sum(len(re.findall(r"\w+|[^\w\s]", message.content)) + 4 for message in messages)
        return max(1, pieces + 2)

    @staticmethod
    def estimate_text_tokens(text: str) -> int:
        return max(1, len(re.findall(r"\w+|[^\w\s]", text)) + 2)

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> Decimal:
        price = self.price_for(model)
        total = (
            Decimal(prompt_tokens) * price.input_per_million_usd
            + Decimal(completion_tokens) * price.output_per_million_usd
        ) / Decimal(1_000_000)
        return total.quantize(Decimal("0.000001"), rounding=ROUND_UP)
