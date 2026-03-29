from __future__ import annotations

import re
import time

from research_agent.config import AppSettings
from research_agent.services.gemini_text import GeminiTextService
from research_agent.services.groq_text import GroqTextService


class TextGenerationService:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._groq = GroqTextService(settings)
        self._gemini = GeminiTextService(settings)
        self._cooldowns: dict[str, float] = {}

    @property
    def available(self) -> bool:
        provider = self._provider()
        if provider == "groq":
            return self._groq.available
        if provider == "gemini":
            return self._gemini.available
        return self._groq.available or self._gemini.available

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
    ) -> str:
        provider = self._provider()
        if provider == "groq":
            return self._generate_with_preferred_provider(
                preferred="groq",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        if provider == "gemini":
            return self._generate_with_preferred_provider(
                preferred="gemini",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        return self._generate_with_auto_fallback(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def _generate_with_preferred_provider(
        self,
        *,
        preferred: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        attempts: list[str] = []
        service = self._service_for(preferred)
        if service is None or not service.available:
            return self._generate_with_auto_fallback(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        try:
            return service.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        except Exception as error:  # pragma: no cover - provider-specific network errors
            attempts.append(f"{preferred}: {str(error)[:180]}")
            self._mark_provider_cooldown(preferred, error)

        for provider_name in self._provider_order():
            if provider_name == preferred:
                continue
            fallback_service = self._service_for(provider_name)
            if fallback_service is None or not fallback_service.available:
                continue
            if self._in_cooldown(provider_name):
                continue
            try:
                return fallback_service.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            except Exception as error:  # pragma: no cover - provider-specific network errors
                attempts.append(f"{provider_name}: {str(error)[:180]}")
                self._mark_provider_cooldown(provider_name, error)

        raise RuntimeError(f"Generation failed after provider failover. {' | '.join(attempts)}")

    def _generate_with_auto_fallback(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        attempts: list[str] = []
        providers = self._provider_order()

        attempted_any = False
        for name in providers:
            service = self._service_for(name)
            if service is None or not service.available:
                continue
            if self._in_cooldown(name):
                attempts.append(f"{name}: cooling down")
                continue
            attempted_any = True
            try:
                return service.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            except Exception as error:  # pragma: no cover - provider-specific network errors
                attempts.append(f"{name}: {str(error)[:180]}")
                self._mark_provider_cooldown(name, error)
                continue

        # If every available provider is in cooldown, retry one pass ignoring cooldown.
        if not attempted_any:
            for name in providers:
                service = self._service_for(name)
                if service is None or not service.available:
                    continue
                try:
                    return service.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    )
                except Exception as error:  # pragma: no cover - provider-specific network errors
                    attempts.append(f"{name}: {str(error)[:180]}")
                    self._mark_provider_cooldown(name, error)

        if attempts:
            joined = " | ".join(attempts)
            raise RuntimeError(f"All generation providers failed. {joined}")

        raise RuntimeError(
            "No generation provider configured. Set GROQ_API_KEY or GEMINI_API_KEY."
        )

    def _provider_order(self) -> list[str]:
        configured = [
            token.strip().lower()
            for token in (self._settings.generation_fallback_order or "").split(",")
            if token.strip()
        ]
        ordered: list[str] = []
        for name in configured + ["gemini", "groq"]:
            if name not in {"groq", "gemini"}:
                continue
            if name in ordered:
                continue
            ordered.append(name)
        return ordered

    def _service_for(self, provider: str) -> GroqTextService | GeminiTextService | None:
        if provider == "groq":
            return self._groq
        if provider == "gemini":
            return self._gemini
        return None

    def _in_cooldown(self, provider: str) -> bool:
        until = float(self._cooldowns.get(provider, 0.0))
        return until > time.time()

    def _mark_provider_cooldown(self, provider: str, error: Exception) -> None:
        text = str(error or "")
        lower = text.lower()
        if not self._is_transient_provider_error(lower):
            return
        cooldown = self._extract_retry_seconds(text)
        if cooldown <= 0:
            cooldown = max(30, int(self._settings.generation_provider_cooldown_seconds))
        self._cooldowns[provider] = time.time() + float(cooldown)

    @staticmethod
    def _is_transient_provider_error(lower_text: str) -> bool:
        markers = (
            "rate_limit_exceeded",
            "rate limit",
            "quota",
            "resource_exhausted",
            "tokens per day",
            "429",
            "temporarily unavailable",
            "service unavailable",
            "timeout",
            "timed out",
            "connection failed",
            "connection reset",
            "overloaded",
        )
        return any(marker in lower_text for marker in markers)

    @staticmethod
    def _extract_retry_seconds(text: str) -> int:
        match = re.search(r"try again in\s+(\d+)\s*m(?:in)?\s*(\d+(?:\.\d+)?)\s*s", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1)) * 60 + int(float(match.group(2)))
        match = re.search(r"try again in\s+(\d+(?:\.\d+)?)\s*s", text, flags=re.IGNORECASE)
        if match:
            return int(float(match.group(1)))
        match = re.search(r"retry[-\s]?after[:\s]+(\d+(?:\.\d+)?)\s*s", text, flags=re.IGNORECASE)
        if match:
            return int(float(match.group(1)))
        return 0

    def _provider(self) -> str:
        provider = (self._settings.generation_provider or "auto").strip().lower()
        if provider in {"auto", "groq", "gemini"}:
            return provider
        return "auto"
