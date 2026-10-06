"""Unified OpenAI-compatible LLM client with disk-backed caching and retries."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from .config import ModelConfig, ModelRegistry, default_registry


class LLMCallStats(BaseModel):
    """Summary metadata for a single model call."""

    model_id: str
    duration_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    attempt_count: int = 1
    cache_hit: bool = False


class LLMClient:
    """Small, cache-aware OpenAI-compatible model client."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        cache_dir: str | Path | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.registry = registry or default_registry()
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "crosscheck"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "llm_cache.sqlite3"
        self._initialize_db()
        self.last_call_stats: LLMCallStats | None = None
        self.call_events: list[LLMCallStats] = []

    def _initialize_db(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _cache_key(self, model_id: str, prompt: str, params: dict[str, Any]) -> str:
        payload = {
            "model_id": model_id,
            "prompt": prompt,
            "params": params,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return digest

    def _read_cache(self, cache_key: str) -> str | None:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT payload FROM llm_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return row[0]

    def _write_cache(self, cache_key: str, model_id: str, payload: str) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO llm_cache (cache_key, model_id, payload) VALUES (?, ?, ?)",
                (cache_key, model_id, payload),
            )
            connection.commit()

    def resolve_model(self, model_id: str) -> ModelConfig:
        """Return a model config by identifier without hardcoded names."""

        return self.registry.get(model_id)

    def chat(
        self,
        model_id: str,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float = 1.0,
        extra_body: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Ask a configured OpenAI-compatible model for a text completion."""

        model = self.resolve_model(model_id)
        params: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "system_prompt": system_prompt,
        }
        if extra_body:
            params.update(extra_body)
        if response_format:
            params["response_format"] = response_format

        cache_key = self._cache_key(model_id, prompt, params)
        cached = self._read_cache(cache_key)
        if cached is not None:
            self.last_call_stats = LLMCallStats(model_id=model_id, duration_ms=0.0, attempt_count=0, cache_hit=True)
            self.call_events.append(self.last_call_stats)
            return cached

        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        request_body: dict[str, Any] = {
            "model": model.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False,
        }
        if extra_body:
            request_body.update(extra_body)
        if response_format:
            request_body["response_format"] = response_format

        for attempt in range(1, self.max_retries + 1):
            started_at = time.perf_counter()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        model.base_url.rstrip("/") + "/chat/completions",
                        headers={"Authorization": f"Bearer {model.api_key}"},
                        json=request_body,
                    )
                response.raise_for_status()
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                if not isinstance(content, str):
                    raise TypeError("Unexpected response format from model provider")

                usage = payload.get("usage", {})
                self.last_call_stats = LLMCallStats(
                    model_id=model_id,
                    duration_ms=(time.perf_counter() - started_at) * 1000,
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    total_tokens=int(usage.get("total_tokens", 0)),
                    attempt_count=attempt,
                )
                self.call_events.append(self.last_call_stats)
                self._write_cache(cache_key, model_id, content)
                return content
            except Exception as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Model call failed after {self.max_retries} attempts: {exc}") from exc
                time.sleep(2 ** (attempt - 1))

        raise RuntimeError("Model call did not return a result")


def build_default_client() -> LLMClient:
    """Return a client configured from the project's default registry."""

    return LLMClient()
