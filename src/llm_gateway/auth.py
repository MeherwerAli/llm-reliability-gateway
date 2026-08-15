import hashlib
import hmac
from dataclasses import dataclass


@dataclass(frozen=True)
class APIPrincipal:
    project_id: str
    key_id: str


class APIKeyRegistry:
    """Small demo registry; production adapters can load the same hash shape from PostgreSQL."""

    def __init__(self, raw_key: str, project_id: str = "portfolio-demo") -> None:
        self._salt = b"llm-reliability-gateway-v1"
        self._digest = self._hash(raw_key)
        self._principal = APIPrincipal(project_id=project_id, key_id="demo-key")

    def _hash(self, raw_key: str) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", raw_key.encode(), self._salt, 200_000)

    def authenticate(self, raw_key: str) -> APIPrincipal | None:
        if hmac.compare_digest(self._hash(raw_key), self._digest):
            return self._principal
        return None
