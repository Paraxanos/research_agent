import json
import re
from urllib import error, request

from research_agent.config import AppSettings


class OpenRouterTextService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings

    @property
    def available(self) -> bool:
        return bool(self._settings.openrouter_api_key)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
    ) -> str:
        if not self.available:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter generation.")

        payload = {
            "model": self._settings.openrouter_generation_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        try:
            response = self._post(payload)
        except RuntimeError as exc:
            retry_tokens = self._retry_budget_tokens(
                error_text=str(exc),
                requested_tokens=max_output_tokens,
            )
            if retry_tokens is None:
                raise
            payload["max_tokens"] = retry_tokens
            response = self._post(payload)
        text = self._extract_text(response)
        if not text:
            raise RuntimeError("OpenRouter response did not include output text.")
        return text

    @staticmethod
    def _retry_budget_tokens(*, error_text: str, requested_tokens: int) -> int | None:
        lower = (error_text or "").lower()
        budget_markers = (
            "requires more credits",
            "fewer max_tokens",
            "can only afford",
            "insufficient credits",
            "not enough credits",
            "credit limit",
            "insufficient_balance",
        )
        if not any(marker in lower for marker in budget_markers):
            return None

        requested = max(1, int(requested_tokens))
        affordable_match = re.search(r"can only afford\s+(\d+)", error_text, flags=re.IGNORECASE)
        affordable = int(affordable_match.group(1)) if affordable_match else 0
        fallback = int(requested * 0.6)
        candidate = affordable if affordable > 0 else fallback
        # Keep enough room for completion text while still shrinking meaningfully.
        min_retry = 128
        candidate = max(min_retry, candidate)
        if candidate >= requested:
            candidate = max(min_retry, requested - 64)
        if candidate >= requested:
            return None
        return candidate

    def _post(self, payload: dict) -> dict:
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            endpoint,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "research-agent/0.1",
            },
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter request failed: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenRouter connection failed: {exc.reason}") from exc

    @staticmethod
    def _extract_text(payload: dict) -> str:
        collected: list[str] = []
        for choice in payload.get("choices", []) or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                collected.append(content.strip())
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    collected.append(text.strip())
        return "\n".join(collected).strip()
