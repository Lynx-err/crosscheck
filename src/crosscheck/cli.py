"""Command-line sample runner for the CrossCheck pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .aggregator import AggregationStrategy, Aggregator
from .claim_extractor import ClaimExtractor
from .config import load_model_registry
from .llm_client import LLMClient
from .nli_scorer import NLIScorer
from .pipeline import CrossCheckPipeline
from .verifier_panel import VerifierPanel

SAMPLE_QUESTIONS = (
    "What is the capital of France?",
    "Which planet is known as the Red Planet?",
    (
        "What was the name of the 14th-century ruler of the small Indian kingdom of Jaffna "
        "who is often credited with founding the Aryacakravarti dynasty?"
    ),
    (
        "In what year did the obscure 17th-century Dutch cartographer Pieter Goos publish "
        "his sea atlas De Zee-Atlas ofte Water-Weereld?"
    ),
    (
        "Which 19th-century botanist first formally described the rare plant "
        "Rafflesia hasseltii, and in what year?"
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run five sample questions through configured primary and verifier models."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to model configuration YAML")
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in AggregationStrategy],
        default=AggregationStrategy.MEAN_ENTAILMENT.value,
    )
    args = parser.parse_args(argv)

    registry = load_model_registry(args.config)
    primary_models = registry.primary_models()
    if not primary_models:
        parser.error("No enabled primary model is configured")

    client = LLMClient(registry, timeout=300.0)
    primary = primary_models[0]
    pipeline = CrossCheckPipeline(
        claim_extractor=ClaimExtractor(client, primary.id),
        verifier_panel=VerifierPanel(client),
        nli_scorer=NLIScorer(registry.nli_model),
        aggregator=Aggregator(),
    )
    strategy = AggregationStrategy(args.strategy)

    for index, question in enumerate(SAMPLE_QUESTIONS, start=1):
        print(f"\n[{index}/5] Question: {question}")
        answer = client.chat(primary.id, question)
        print(f"Primary answer ({primary.id}): {answer}")
        result = pipeline.score(question, answer, strategy)
        if result.answer_score is None:
            print("Answer agreement: N/A (no factual claims extracted)")
        else:
            print(f"Answer agreement: {result.answer_score:.3f} ({strategy.value})")
        for claim in result.claims:
            print(f"  Claim [{claim.agreement_score:.3f}]: {claim.claim}")
            for verifier in claim.verifiers:
                probs = verifier.probabilities
                print(
                    f"    {verifier.model_id}: {verifier.stance} "
                    f"(entail={probs.entailment:.3f}, neutral={probs.neutral:.3f}, "
                    f"contradict={probs.contradiction:.3f})"
                )
                print(f"      Answer: {verifier.answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
