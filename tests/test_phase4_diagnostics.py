"""Offline tests for Phase 4 diagnostic scoring and panel selection."""

from __future__ import annotations

from types import SimpleNamespace

from crosscheck.config import ModelConfig, ModelRegistry
from crosscheck.eval.datasets import HaluEvalQARecord, WikiBioRecord
from crosscheck.eval.run_diagnostics import (
    _family_panels,
    _qa_diagnostic,
    _resolve_panel_models,
    _wiki_bio_diagnostic,
)
from crosscheck.nli_scorer import ClaimEvidence, NLIProbabilities, VerifierEvidence
from crosscheck.verifier_panel import VerifierAnswer


def _registry() -> ModelRegistry:
    return ModelRegistry(
        nli_model="offline-nli",
        models=[
            ModelConfig(
                id="qwen-3b",
                provider="test",
                model_name="qwen-3b",
                base_url="http://localhost/v1",
                role="verifier",
                family="qwen",
            ),
            ModelConfig(
                id="qwen-7b",
                provider="test",
                model_name="qwen-7b",
                base_url="http://localhost/v1",
                role="verifier",
                family="qwen",
            ),
            ModelConfig(
                id="phi-mini",
                provider="test",
                model_name="phi-mini",
                base_url="http://localhost/v1",
                role="verifier",
                family="phi",
            ),
            ModelConfig(
                id="gemma-small",
                provider="test",
                model_name="gemma-small",
                base_url="http://localhost/v1",
                role="verifier",
                family="gemma",
            ),
        ],
    )


class _FakeScorer:
    max_length = 256

    def score(
        self, claims: list[str], answers: list[VerifierAnswer]
    ) -> list[ClaimEvidence]:
        results = []
        for claim in claims:
            verifiers = [
                VerifierEvidence(
                    model_id=answer.model_id,
                    answer=answer.answer,
                    probabilities=NLIProbabilities(
                        entailment=0.7,
                        neutral=0.2,
                        contradiction=0.1,
                    ),
                    window_count=2,
                    would_truncate_without_windowing=True,
                )
                for answer in answers
            ]
            results.append(ClaimEvidence(claim=claim, verifiers=verifiers))
        return results


class _FakePanel:
    def run(self, prompt: str, model_ids: list[str]) -> list[VerifierAnswer]:
        return [
            VerifierAnswer(model_id=model_id, answer=f"Independent answer: {prompt}")
            for model_id in model_ids
        ]


class _FakeSelfCheck:
    def sample_answers(self, prompt: str) -> list[VerifierAnswer]:
        return [
            VerifierAnswer(model_id=f"primary:sample-{index}", answer=prompt)
            for index in range(5)
        ]


def test_family_panels_select_qwen_pair_and_equal_sized_cross_family() -> None:
    same_family, cross_family = _family_panels(_registry())

    assert same_family == ["qwen-3b", "qwen-7b"]
    assert cross_family == ["qwen-3b", "phi-mini"]


def test_explicit_panel_resolution_keeps_order_and_includes_disabled_models() -> None:
    registry = _registry()
    registry.models[1].enabled = False

    selected = _resolve_panel_models(registry, ["qwen-7b", "qwen-3b"])

    assert [model.id for model in selected] == ["qwen-7b", "qwen-3b"]


def test_wikibio_and_qa_diagnostics_score_mocked_records() -> None:
    registry = _registry()
    client = SimpleNamespace(call_events=[])
    scorer = _FakeScorer()
    panel = _FakePanel()
    selfcheck = _FakeSelfCheck()
    wiki_records = [
        WikiBioRecord(
            gpt3_text=f"Fact {index}.",
            wiki_bio_text=f"Subject {index} was a researcher.",
            gpt3_sentences=[f"Fact {index}."],
            annotation=[label],
            wiki_bio_test_idx=index,
        )
        for index, label in ((1, "accurate"), (2, "major_inaccurate"))
    ]
    wiki = _wiki_bio_diagnostic(
        wiki_records,
        registry,
        client,
        scorer,
        panel,
        selfcheck,
        seed=42,
    )

    assert wiki["n_concepts"] == 2
    assert wiki["truncation_diagnostic"]["pairs_that_would_truncate_as_whole_passage"] == 8
    assert wiki["nli_mass_by_sentence_class"]["accurate"]["mean_neutral"] == 0.2

    qa = _qa_diagnostic(
        [
            HaluEvalQARecord(
                question=f"Question {index}?",
                right_answer=f"Right {index}.",
                hallucinated_answer=f"Wrong {index}.",
            )
            for index in (1, 2)
        ],
        registry,
        client,
        scorer,
        panel,
        selfcheck,
        seed=42,
    )

    assert qa["n_questions"] == 2
    assert qa["metrics"]["crosscheck"]["positive_prevalence"] == 0.5
