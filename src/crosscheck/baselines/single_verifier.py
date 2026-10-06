"""Single-verifier variant of the CrossCheck scoring method."""

from __future__ import annotations

from crosscheck.aggregator import AggregatedClaim, AggregationStrategy, Aggregator
from crosscheck.nli_scorer import ClaimEvidence
from crosscheck.pipeline import CrossCheckPipeline
from crosscheck.verifier_panel import VerifierAnswer, VerifierPanel


class SingleVerifierBaseline:
    """Run the configured NLI model with one independent verifier answer."""

    def __init__(self, panel: VerifierPanel, pipeline: CrossCheckPipeline, model_id: str) -> None:
        self.panel = panel
        self.pipeline = pipeline
        self.model_id = model_id

    def answer(self, prompt: str) -> VerifierAnswer:
        """Obtain an independent answer from exactly the selected verifier."""

        return self.panel.run(prompt, model_ids=[self.model_id])[0]

    def hallucination_score(
        self,
        prompt: str,
        sentence: str,
        answer: VerifierAnswer,
        strategy: AggregationStrategy | str = AggregationStrategy.MEAN_ENTAILMENT,
    ) -> float:
        """Run the common pipeline with this verifier only and return risk."""

        result = self.pipeline.score_claims(
            prompt,
            sentence,
            [sentence],
            strategy,
            verifier_answers=[answer],
        )
        return 1.0 - result.claims[0].agreement_score

    def hallucination_score_from_evidence(
        self,
        evidence: AggregatedClaim,
        strategy: AggregationStrategy | str = AggregationStrategy.MEAN_ENTAILMENT,
    ) -> float:
        """Reuse panel evidence while aggregating only the selected verifier."""

        selected = [
            verifier for verifier in evidence.verifiers if verifier.model_id == self.model_id
        ]
        if not selected:
            raise ValueError(
                f"Evidence does not contain selected verifier {self.model_id!r}"
            )
        individual = Aggregator().aggregate(
            [ClaimEvidence(claim=evidence.claim, verifiers=selected)],
            strategy,
        )[0]
        return 1.0 - individual.agreement_score
