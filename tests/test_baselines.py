"""Tests for evaluation baselines with fully mocked model calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from crosscheck.aggregator import AggregationStrategy
from crosscheck.baselines.selfcheck_nli import SelfCheckNLIBaseline
from crosscheck.baselines.single_verifier import SingleVerifierBaseline
from crosscheck.claim_extractor import ClaimExtractor
from crosscheck.config import ModelConfig, ModelRegistry
from crosscheck.llm_client import LLMClient
from crosscheck.nli_scorer import NLIScorer
from crosscheck.pipeline import CrossCheckPipeline
from crosscheck.verifier_panel import VerifierPanel


def make_client(tmp_path: Path) -> LLMClient:
    """Construct a test-only configured LLM client."""

    return LLMClient(
        registry=ModelRegistry(
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
                    id="verifier-a",
                    provider="test",
                    model_name="verifier-a",
                    base_url="http://localhost/v1",
                    role="verifier",
                ),
                ModelConfig(
                    id="verifier-b",
                    provider="test",
                    model_name="verifier-b",
                    base_url="http://localhost/v1",
                    role="verifier",
                ),
            ],
        ),
        cache_dir=tmp_path,
    )


def test_selfcheck_samples_five_independent_temperature_one_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client(tmp_path)
    calls: list[tuple[str, float, int]] = []

    def mocked_chat(
        model_id: str, prompt: str, *, temperature: float = 0.7, max_tokens: int = 256, **kwargs: object
    ) -> str:
        calls.append((prompt, temperature, max_tokens))
        return f"sample {len(calls)}"

    monkeypatch.setattr(client, "chat", mocked_chat)
    baseline = SelfCheckNLIBaseline(
        client,
        NLIScorer(
            "test-nli",
            cache_dir=tmp_path,
            pair_classifier=lambda pairs: [
                {"entailment": 0.8, "neutral": 0.1, "contradiction": 0.1}
                for _ in pairs
            ],
        ),
        "primary",
    )

    answers = baseline.sample_answers("Write a short biography.")

    assert len(answers) == 5
    assert len({call[0] for call in calls}) == 5
    assert all(call[1] == 1.0 for call in calls)
    assert all(call[2] == 256 for call in calls)
    assert baseline.hallucination_score("A factual sentence.", answers) == pytest.approx(0.2)


def test_single_verifier_runs_pipeline_with_only_selected_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client(tmp_path)
    calls: list[str] = []

    def mocked_chat(model_id: str, prompt: str, **kwargs: object) -> str:
        calls.append(model_id)
        return f"answer from {model_id}"

    monkeypatch.setattr(client, "chat", mocked_chat)
    panel = VerifierPanel(client)
    pipeline = CrossCheckPipeline(
        ClaimExtractor(client, "primary"),
        panel,
        NLIScorer(
            "test-nli",
            cache_dir=tmp_path,
            pair_classifier=lambda pairs: [
                {"entailment": 0.7, "neutral": 0.2, "contradiction": 0.1}
                for _ in pairs
            ],
        ),
    )
    baseline = SingleVerifierBaseline(panel, pipeline, "verifier-b")

    answer = baseline.answer("Write a short biography.")
    score = baseline.hallucination_score(
        "Write a short biography.",
        "A factual sentence.",
        answer,
        AggregationStrategy.MEAN_ENTAILMENT,
    )

    assert calls == ["verifier-b"]
    assert score == pytest.approx(0.3)
