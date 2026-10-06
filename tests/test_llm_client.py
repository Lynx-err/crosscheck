from __future__ import annotations

from pathlib import Path

import pytest

from crosscheck.config import ModelConfig, ModelRegistry, load_model_registry
from crosscheck.llm_client import LLMClient


def test_load_model_registry_reads_yaml(tmp_path: Path) -> None:
    config = tmp_path / "models.yaml"
    config.write_text(
        """
nli_model: test-nli
models:
  - id: test-primary
    provider: ollama
    model_name: llama3.2:3b
    base_url: http://localhost:11434/v1
    api_key: ollama
    role: primary
    enabled: true
  - id: test-verifier
    provider: ollama
    model_name: qwen2.5:3b
    base_url: http://localhost:11434/v1
    api_key: ollama
    role: verifier
    enabled: true
""".strip(),
        encoding="utf-8",
    )

    registry = load_model_registry(config)
    assert registry.primary_models()[0].id == "test-primary"
    assert registry.get("test-verifier").role == "verifier"


def test_llm_client_caches_repeated_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeResponse:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return self._payload

    def fake_post(self, url: str, headers: dict[str, str], json: dict[str, object]):
        calls.append({"url": url, "headers": headers, "json": json})
        return FakeResponse({
            "choices": [{"message": {"content": "cached answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        })

    monkeypatch.setattr("httpx.Client.post", fake_post)
    client = LLMClient(
        registry=ModelRegistry(
            nli_model="test-nli",
            models=[
                ModelConfig(
                    id="demo-model",
                    provider="ollama",
                    model_name="llama3.2:3b",
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                    role="primary",
                )
            ]
        ),
        cache_dir=tmp_path,
    )

    first = client.chat("demo-model", "What is 2+2?", temperature=0.0, max_tokens=32)
    second = client.chat("demo-model", "What is 2+2?", temperature=0.0, max_tokens=32)

    assert first == "cached answer"
    assert second == "cached answer"
    assert len(calls) == 1


def test_llm_client_retries_transient_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    class FakeResponse:
        def __init__(self, success: bool) -> None:
            self.success = success

        def raise_for_status(self) -> None:
            if not self.success:
                raise RuntimeError("temporary outage")

        def json(self) -> dict[str, object]:
            return {
                "choices": [{"message": {"content": "after retry"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            }

    def fake_post(self, url: str, headers: dict[str, str], json: dict[str, object]):
        attempts.append(len(attempts) + 1)
        if len(attempts) < 2:
            raise RuntimeError("temporary outage")
        return FakeResponse(True)

    monkeypatch.setattr("httpx.Client.post", fake_post)
    client = LLMClient(
        registry=ModelRegistry(
            nli_model="test-nli",
            models=[
                ModelConfig(
                    id="retry-model",
                    provider="ollama",
                    model_name="qwen2.5:3b",
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                    role="verifier",
                )
            ]
        ),
        cache_dir=tmp_path,
        max_retries=3,
    )

    result = client.chat("retry-model", "Explain the moon.")
    assert result == "after retry"
    assert len(attempts) == 2
