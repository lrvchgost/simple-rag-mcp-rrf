"""Обёртка над локальным REST API Ollama: генерация текста LLM."""

from __future__ import annotations

import logging
import time

import requests

from src.config import Settings
from src.domain.errors import OllamaUnavailableError
from src.domain.interfaces import LLMService

logger = logging.getLogger(__name__)


class OllamaService(LLMService):
    """Вызовы /api/generate и /api/tags c retry и экспоненциальным backoff."""

    def __init__(self, settings: Settings) -> None:
        self._base = settings.ollama_base_url.rstrip("/")
        self._model = settings.llm_model
        self._timeout = settings.ollama_timeout
        self._retries = max(0, settings.ollama_retries)

    def check_availability(self) -> None:
        url = self._base + "/api/tags"
        try:
            requests.get(url, timeout=5).raise_for_status()
        except requests.RequestException as exc:
            raise OllamaUnavailableError(
                f"Ollama недоступна по адресу {self._base}. "
                "Запустите: docker compose up -d ollama (или 'ollama serve')."
            ) from exc

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = False,
    ) -> str:
        payload: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        if system_prompt:
            payload["system"] = system_prompt
        if json_mode:
            payload["format"] = "json"

        started = time.monotonic()
        logger.info(
            "ollama generate model=%s prompt_chars=%d json=%s",
            self._model, len(prompt), json_mode,
        )
        last_exc: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                resp = requests.post(
                    self._base + "/api/generate", json=payload, timeout=self._timeout
                )
                resp.raise_for_status()
                answer = str(resp.json().get("response", "")).strip()
                logger.info(
                    "ollama done model=%s answer_chars=%d elapsed=%.1fs",
                    self._model, len(answer), time.monotonic() - started,
                )
                return answer
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self._retries:
                    time.sleep(2**attempt)
        raise OllamaUnavailableError(
            f"Ошибка генерации в Ollama ({self._base}, модель {self._model}): {last_exc}"
        ) from last_exc
