"""Tests for independent verifier-panel behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from crosscheck.config import ModelConfig, ModelRegistry
from crosscheck.llm_client import LLMClient
from crosscheck.verifier_panel import VerifierPanel


def make_client(tmp_path: Path, registry: ModelRegistry) -> LLMClient:
    """Build a client with a test registry and an unused local cache."""

    return LLMClient(registry=registry, cache_dir=tmp_path)


def test_panel_queries_each_enabled_verifier_with_only_the_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ModelRegistry(
        nli_model="test-nli",
        models=[
            ModelConfig(
                id="primary",
                provider="test",
                model_name="primary",
                base_url="http://localhost/v1",
                role="primary",
            ),
            ModelConfig(
                id="verifier-one",
                provider="test",
                model_name="verifier-one",
                base_url="http://localhost/v1",
                role="verifier",
            ),
            ModelConfig(
                id="disabled-verifier",
                provider="test",
                model_name="disabled-verifier",
                base_url="http://localhost/v1",
                role="verifier",
                enabled=False,
            ),
            ModelConfig(
                id="verifier-two",
                provider="test",
                model_name="verifier-two",
                base_url="http://localhost/v1",
                role="verifier",
            ),
        ]
    )
    client = make_client(tmp_path, registry)
    calls: list[tuple[str, str, str | None]] = []

    def mocked_chat(
        model_id: str, prompt: str, *, system_prompt: str | None = None, **kwargs: object
    ) -> str:
        calls.append((model_id, prompt, system_prompt))
        return f"answer from {model_id}"

    monkeypatch.setattr(client, "chat", mocked_chat)

    results = VerifierPanel(client).run("What is the capital of France?")

    assert [result.model_id for result in results] == ["verifier-one", "verifier-two"]
    assert [result.answer for result in results] == [
        "answer from verifier-one",
        "answer from verifier-two",
    ]
    assert calls == [
        (
            "verifier-one",
            "What is the capital of France?",
            (
                "Answer the user's question independently. Do not assume any prior "
                "answer or claim has been provided."
            ),
        ),
        (
            "verifier-two",
            "What is the capital of France?",
            (
                "Answer the user's question independently. Do not assume any prior "
                "answer or claim has been provided."
            ),
        ),
    ]


def test_panel_requires_an_enabled_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ModelRegistry(
        nli_model="test-nli",
        models=[
            ModelConfig(
                id="primary",
                provider="test",
                model_name="primary",
                base_url="http://localhost/v1",
                role="primary",
            )
        ]
    )
    client = make_client(tmp_path, registry)
    monkeypatch.setattr(
        client,
        "chat",
        lambda *args, **kwargs: pytest.fail("chat must not run without enabled verifiers"),
    )

    with pytest.raises(ValueError, match="No enabled verifier models"):
        VerifierPanel(client).run("What is the answer?")


def test_explicit_panel_can_select_a_disabled_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = ModelRegistry(
        nli_model="test-nli",
        models=[
            ModelConfig(
                id="disabled-verifier",
                provider="test",
                model_name="disabled-verifier",
                base_url="http://localhost/v1",
                role="verifier",
                enabled=False,
            )
        ],
    )
    client = make_client(tmp_path, registry)
    calls: list[str] = []
    monkeypatch.setattr(
        client,
        "chat",
        lambda model_id, *args, **kwargs: calls.append(model_id) or "answer",
    )

    results = VerifierPanel(client).run("Question?", model_ids=["disabled-verifier"])

    assert [answer.model_id for answer in results] == ["disabled-verifier"]
    assert calls == ["disabled-verifier"]
