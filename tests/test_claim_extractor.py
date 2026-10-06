"""Tests for claim extraction and malformed-output retries."""

from __future__ import annotations

from pathlib import Path

import pytest

from crosscheck.claim_extractor import ClaimExtractionError, ClaimExtractor
from crosscheck.config import ModelConfig, ModelRegistry
from crosscheck.llm_client import LLMClient


def make_client(tmp_path: Path) -> LLMClient:
    """Build a client with a model stub so tests never call a live endpoint."""

    return LLMClient(
        registry=ModelRegistry(
            nli_model="test-nli",
            models=[
                ModelConfig(
                    id="extractor",
                    provider="test",
                    model_name="extractor",
                    base_url="http://localhost/v1",
                )
            ]
        ),
        cache_dir=tmp_path,
    )


def test_extract_validates_and_normalizes_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client(tmp_path)
    monkeypatch.setattr(client, "chat", lambda *args, **kwargs: '{"claims":[" The moon orbits Earth. "]}')

    result = ClaimExtractor(client, "extractor").extract("What orbits Earth?", "The moon orbits Earth.")

    assert result.claims == ["The moon orbits Earth."]


def test_extract_retries_malformed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client(tmp_path)
    prompts: list[str] = []
    responses = iter(["not json", '{"claims":["Earth orbits the Sun."]}'])

    def mocked_chat(model_id: str, prompt: str, **kwargs: object) -> str:
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(client, "chat", mocked_chat)

    result = ClaimExtractor(client, "extractor", max_retries=2).extract(
        "What does Earth orbit?", "Earth orbits the Sun."
    )

    assert result.claims == ["Earth orbits the Sun."]
    assert len(prompts) == 2
    assert "not json" not in prompts[1]
    assert "did not match the required JSON schema" in prompts[1]


def test_extract_fails_after_all_outputs_are_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client(tmp_path)
    monkeypatch.setattr(client, "chat", lambda *args, **kwargs: '{"claims":[3]}')

    with pytest.raises(ClaimExtractionError, match="after 2 attempts"):
        ClaimExtractor(client, "extractor", max_retries=2).extract("Question?", "Answer.")
