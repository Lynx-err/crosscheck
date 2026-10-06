"""Aggregation strategies for claim-level verifier agreement scores."""

from __future__ import annotations

import math
from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, Field

from .nli_scorer import ClaimEvidence, VerifierEvidence


class AggregationStrategy(StrEnum):
    """Supported ways to combine multiple verifier stances."""

    MEAN_ENTAILMENT = "mean_entailment"
    MIN = "min"
    CONTRADICTION_WEIGHTED = "contradiction_weighted"
    ENTROPY = "entropy"


class AggregatedClaim(BaseModel):
    """Claim evidence with its selected scalar agreement score."""

    claim: str
    agreement_score: float = Field(ge=0.0, le=1.0)
    strategy: AggregationStrategy
    verifiers: list[VerifierEvidence]


class Aggregator:
    """Combine NLI evidence into an agreement score in the [0, 1] range."""

    def aggregate(
        self,
        claim_evidence: list[ClaimEvidence],
        strategy: AggregationStrategy | str = AggregationStrategy.MEAN_ENTAILMENT,
    ) -> list[AggregatedClaim]:
        """Apply a strategy independently to each claim's verifier evidence."""

        selected = AggregationStrategy(strategy)
        aggregated: list[AggregatedClaim] = []
        for evidence in claim_evidence:
            if not evidence.verifiers:
                raise ValueError(f"Claim {evidence.claim!r} has no verifier evidence")
            probabilities = [item.probabilities for item in evidence.verifiers]
            entailment = [item.entailment for item in probabilities]

            if selected == AggregationStrategy.MEAN_ENTAILMENT:
                score = sum(entailment) / len(entailment)
            elif selected == AggregationStrategy.MIN:
                score = min(entailment)
            elif selected == AggregationStrategy.CONTRADICTION_WEIGHTED:
                score = max(
                    0.0,
                    sum(
                        item.entailment - item.contradiction
                        for item in probabilities
                    )
                    / len(probabilities),
                )
            else:
                score = self._entropy_score(evidence, sum(entailment) / len(entailment))

            aggregated.append(
                AggregatedClaim(
                    claim=evidence.claim,
                    agreement_score=min(1.0, score),
                    strategy=selected,
                    verifiers=evidence.verifiers,
                )
            )
        return aggregated

    @staticmethod
    def _entropy_score(evidence: ClaimEvidence, mean_entailment: float) -> float:
        stance_counts = Counter(verifier.stance for verifier in evidence.verifiers)
        total = len(evidence.verifiers)
        entropy = -sum(
            (count / total) * math.log(count / total)
            for count in stance_counts.values()
        )
        normalized_entropy = entropy / math.log(3)
        return mean_entailment * (1.0 - normalized_entropy)
