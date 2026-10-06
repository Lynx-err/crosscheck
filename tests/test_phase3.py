"""Unit tests for batched NLI scoring, aggregation, and pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from crosscheck.aggregator import AggregationStrategy, Aggregator
from crosscheck.claim_extractor import ClaimExtraction, ClaimExtractor
from crosscheck.config import ModelConfig, ModelRegistry
from crosscheck.llm_client import LLMClient
from crosscheck.nli_scorer import NLIScorer
from crosscheck.pipeline import CrossCheckPipeline
from crosscheck.verifier_panel import VerifierAnswer, VerifierPanel


def make_client(tmp_path: Path) -> LLMClient:
    """Make an offline model client for pipeline tests."""

    return LLMClient(
        registry=ModelRegistry(
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
            nli_model="test-nli-model",
        ),
        cache_dir=tmp_path,
    )


def test_nli_scorer_batches_pairs_and_maps_results(tmp_path: Path) -> None:
    batches: list[list[tuple[str, str]]] = []

    def fake_classifier(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        batches.append(pairs)
        return [
            {"entailment": 0.8, "neutral": 0.1, "contradiction": 0.1}
            for _ in pairs
        ]

    scorer = NLIScorer(
        "configured-nli-model",
        batch_size=2,
        cache_dir=tmp_path,
        pair_classifier=fake_classifier,
    )
    evidence = scorer.score(
        ["Claim A", "Claim B"],
        [
            VerifierAnswer(model_id="v1", answer="Answer one"),
            VerifierAnswer(model_id="v2", answer="Answer two"),
        ],
    )

    assert len(batches) == 2
    assert all(len(batch) == 2 for batch in batches)
    assert batches[0] == [("Answer one", "Claim A"), ("Answer two", "Claim A")]
    assert evidence[1].verifiers[1].probabilities.entailment == pytest.approx(0.8)
    assert evidence[1].verifiers[1].stance == "entailment"


def test_nli_scorer_persists_pair_scores_in_sqlite(tmp_path: Path) -> None:
    calls: list[list[tuple[str, str]]] = []

    def classifier(pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        calls.append(pairs)
        return [
            {"entailment": 0.6, "neutral": 0.3, "contradiction": 0.1}
            for _ in pairs
        ]

    answers = [VerifierAnswer(model_id="v1", answer="Evidence")]
    NLIScorer("cache-test-model", cache_dir=tmp_path, pair_classifier=classifier).score(
        ["A claim"], answers
    )
    cached_result = NLIScorer(
        "cache-test-model",
        cache_dir=tmp_path,
        pair_classifier=lambda pairs: pytest.fail("cached pairs must not be recomputed"),
    ).score(["A claim"], answers)

    assert len(calls) == 1
    assert cached_result[0].verifiers[0].probabilities.entailment == pytest.approx(0.6)


def test_passage_splitting_windows_long_sentences_without_losing_text(
    tmp_path: Path,
) -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [int(token) for token in text.split() if token.isdigit()]

        def __call__(
            self, premise: str, hypothesis: str, **kwargs: object
        ) -> dict[str, list[int]]:
            return {"input_ids": self.encode(premise) + self.encode(hypothesis) + [999]}

        def num_special_tokens_to_add(self, pair: bool = True) -> int:
            return 1

        def decode(self, token_ids: list[int], **kwargs: object) -> str:
            return " ".join(map(str, token_ids))

    scorer = NLIScorer(
        "test-nli",
        max_length=6,
        cache_dir=tmp_path,
    )
    scorer._tokenizer = FakeTokenizer()
    windows, would_truncate = scorer._split_passage(
        "1 2 3 4 5 6 7 8 9 10", "42"
    )

    assert would_truncate
    assert len(windows) > 1
    reconstructed = {int(token) for window in windows for token in window.split()}
    assert reconstructed == set(range(1, 11))


def test_window_scoring_keeps_independent_max_entailment_and_contradiction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [int(token) for token in text.split() if token.isdigit()]

        def __call__(
            self, premise: str, hypothesis: str, **kwargs: object
        ) -> dict[str, list[int]]:
            return {"input_ids": self.encode(premise) + self.encode(hypothesis) + [999]}

        def num_special_tokens_to_add(self, pair: bool = True) -> int:
            return 1

        def decode(self, token_ids: list[int], **kwargs: object) -> str:
            return " ".join(map(str, token_ids))

    scorer = NLIScorer("test-nli", max_length=6, cache_dir=tmp_path)
    scorer._tokenizer = FakeTokenizer()
    monkeypatch.setattr(scorer, "_load_model", lambda: None)
    monkeypatch.setattr(
        scorer,
        "_classify_pairs",
        lambda pairs: [
            {"entailment": 0.8, "neutral": 0.1, "contradiction": 0.1},
            {"entailment": 0.2, "neutral": 0.1, "contradiction": 0.7},
            {"entailment": 0.1, "neutral": 0.3, "contradiction": 0.6},
        ],
    )

    evidence = scorer.score(
        ["42"],
        [VerifierAnswer(model_id="v1", answer="1 2 3 4 5 6 7 8")],
    )[0].verifiers[0]

    assert evidence.window_count == 3
    assert evidence.would_truncate_without_windowing
    assert evidence.probabilities.entailment == pytest.approx(0.8)
    assert evidence.probabilities.contradiction == pytest.approx(0.7)
    assert evidence.probabilities.neutral == pytest.approx(0.0)


def test_aggregator_supports_all_strategies(tmp_path: Path) -> None:
    scorer = NLIScorer(
        "test",
        cache_dir=tmp_path,
        pair_classifier=lambda pairs: [
            {"entailment": 0.8, "neutral": 0.1, "contradiction": 0.1},
            {"entailment": 0.2, "neutral": 0.1, "contradiction": 0.7},
        ],
    )
    evidence = scorer.score(
        ["Claim"],
        [
            VerifierAnswer(model_id="v1", answer="Supports the claim."),
            VerifierAnswer(model_id="v2", answer="Contradicts the claim."),
        ],
    )
    aggregator = Aggregator()

    scores = {
        strategy: aggregator.aggregate(evidence, strategy)[0].agreement_score
        for strategy in AggregationStrategy
    }

    assert scores[AggregationStrategy.MEAN_ENTAILMENT] == pytest.approx(0.5)
    assert scores[AggregationStrategy.MIN] == pytest.approx(0.2)
    assert scores[AggregationStrategy.CONTRADICTION_WEIGHTED] == pytest.approx(0.1)
    assert 0.0 <= scores[AggregationStrategy.ENTROPY] <= 1.0


def test_pipeline_orchestrates_mocked_extraction_verification_and_nli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client(tmp_path)
    monkeypatch.setattr(
        ClaimExtractor,
        "extract",
        lambda self, question, answer: type("Extraction", (), {"claims": ["Claim A", "Claim B"]})(),
    )
    monkeypatch.setattr(
        VerifierPanel,
        "run",
        lambda self, question: [
            VerifierAnswer(model_id="verifier-a", answer="Evidence for A."),
            VerifierAnswer(model_id="verifier-b", answer="Evidence for B."),
        ],
    )
    scorer = NLIScorer(
        client.registry.nli_model,
        cache_dir=tmp_path,
        pair_classifier=lambda pairs: [
            {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05},
            {"entailment": 0.7, "neutral": 0.2, "contradiction": 0.1},
            {"entailment": 0.2, "neutral": 0.2, "contradiction": 0.6},
            {"entailment": 0.4, "neutral": 0.4, "contradiction": 0.2},
        ],
    )
    pipeline = CrossCheckPipeline(
        ClaimExtractor(client, "primary"),
        VerifierPanel(client),
        scorer,
    )

    result = pipeline.score("Question?", "Primary answer.", AggregationStrategy.MEAN_ENTAILMENT)

    assert len(result.claims) == 2
    assert result.claims[0].claim == "Claim A"
    assert result.claims[0].agreement_score == pytest.approx(0.8)
    assert result.answer_score == pytest.approx(0.55)
    assert result.claims[1].verifiers[0].answer == "Evidence for A."


def test_pipeline_marks_answer_without_extracted_claims_as_unscored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client(tmp_path)
    extractor = ClaimExtractor(client, "primary")
    verifier_panel = VerifierPanel(client)
    monkeypatch.setattr(
        extractor,
        "extract",
        lambda question, answer: ClaimExtraction(claims=[]),
    )
    monkeypatch.setattr(verifier_panel, "run", lambda question: [])
    result = CrossCheckPipeline(
        extractor,
        verifier_panel,
        NLIScorer("test-nli-model"),
    ).score("Question?", "I do not know.")

    assert result.claims == []
    assert result.answer_score is None
