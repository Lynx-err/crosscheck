"""End-to-end CrossCheck claim-level scoring pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .aggregator import AggregatedClaim, AggregationStrategy, Aggregator
from .claim_extractor import ClaimExtractor
from .nli_scorer import NLIScorer
from .verifier_panel import VerifierAnswer, VerifierPanel


class PipelineResult(BaseModel):
    """Scoring result for one primary answer and its extracted claims."""

    question: str
    primary_answer: str
    aggregation_strategy: AggregationStrategy
    answer_score: float | None = Field(default=None, ge=0.0, le=1.0)
    claims: list[AggregatedClaim]


class CrossCheckPipeline:
    """Orchestrate claim extraction, verifier answers, NLI, and aggregation."""

    def __init__(
        self,
        claim_extractor: ClaimExtractor,
        verifier_panel: VerifierPanel,
        nli_scorer: NLIScorer,
        aggregator: Aggregator | None = None,
    ) -> None:
        self.claim_extractor = claim_extractor
        self.verifier_panel = verifier_panel
        self.nli_scorer = nli_scorer
        self.aggregator = aggregator or Aggregator()

    def score(
        self,
        question: str,
        primary_answer: str,
        strategy: AggregationStrategy | str = AggregationStrategy.MEAN_ENTAILMENT,
    ) -> PipelineResult:
        """Return each answer claim with verifier probabilities and agreement score."""

        extraction = self.claim_extractor.extract(question, primary_answer)
        return self.score_claims(question, primary_answer, extraction.claims, strategy)

    def score_claims(
        self,
        question: str,
        source_text: str,
        claims: list[str],
        strategy: AggregationStrategy | str = AggregationStrategy.MEAN_ENTAILMENT,
        *,
        verifier_model_ids: list[str] | None = None,
        verifier_answers: list[VerifierAnswer] | None = None,
    ) -> PipelineResult:
        """Score pre-segmented claims, optionally using a restricted answer panel."""

        if verifier_answers is not None and verifier_model_ids is not None:
            raise ValueError("Specify verifier_answers or verifier_model_ids, not both")
        if verifier_answers is None:
            if verifier_model_ids is None:
                verifier_answers = self.verifier_panel.run(question)
            else:
                verifier_answers = self.verifier_panel.run(
                    question, model_ids=verifier_model_ids
                )
        evidence = self.nli_scorer.score(claims, verifier_answers)
        aggregated = self.aggregator.aggregate(evidence, strategy)
        answer_score = (
            sum(claim.agreement_score for claim in aggregated) / len(aggregated)
            if aggregated
            else None
        )
        return PipelineResult(
            question=question,
            primary_answer=source_text,
            aggregation_strategy=AggregationStrategy(strategy),
            answer_score=answer_score,
            claims=aggregated,
        )
