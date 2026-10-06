"""SelfCheckGPT-style sampling and NLI scoring baseline."""

from __future__ import annotations

from crosscheck.llm_client import LLMClient
from crosscheck.nli_scorer import ClaimEvidence, NLIScorer
from crosscheck.verifier_panel import VerifierAnswer


class SelfCheckNLIBaseline:
    """Score sentences against temperature-one samples from the same model."""

    def __init__(
        self,
        client: LLMClient,
        nli_scorer: NLIScorer,
        model_id: str,
        sample_count: int = 5,
    ) -> None:
        if sample_count != 5:
            raise ValueError("The SelfCheck baseline uses exactly five samples")
        self.client = client
        self.nli_scorer = nli_scorer
        self.model_id = model_id
        self.sample_count = sample_count

    def sample_answers(self, prompt: str) -> list[VerifierAnswer]:
        """Generate and cache five independently keyed samples at temperature 1.0."""

        samples: list[VerifierAnswer] = []
        for index in range(self.sample_count):
            sample_prompt = (
                f"{prompt}\n\nGenerate independent response sample {index + 1} of "
                f"{self.sample_count}."
            )
            answer = self.client.chat(
                self.model_id,
                sample_prompt,
                temperature=1.0,
                max_tokens=256,
            )
            samples.append(
                VerifierAnswer(model_id=f"{self.model_id}:sample-{index + 1}", answer=answer)
            )
        return samples

    def hallucination_score(self, sentence: str, samples: list[VerifierAnswer]) -> float:
        """Return one minus mean entailment, so higher scores indicate risk."""

        return self.hallucination_scores([sentence], samples)[0]

    def hallucination_scores(
        self, sentences: list[str], samples: list[VerifierAnswer]
    ) -> list[float]:
        """Score multiple sentences together to use batched NLI inference."""

        evidence = self.score_sentences(sentences, samples)
        return [
            1.0
            - sum(
                verifier.probabilities.entailment for verifier in claim.verifiers
            )
            / len(claim.verifiers)
            for claim in evidence
        ]

    def score_sentences(
        self, sentences: list[str], samples: list[VerifierAnswer]
    ) -> list[ClaimEvidence]:
        """Return complete per-sample stance probabilities for diagnostics."""

        return self.nli_scorer.score(sentences, samples)
