"""Independent question-answer generation across configured verifier models."""

from __future__ import annotations

from pydantic import BaseModel

from .llm_client import LLMClient


class VerifierAnswer(BaseModel):
    """One verifier model's independent response to the user's question."""

    model_id: str
    answer: str


class VerifierPanel:
    """Run selected verifiers without exposing the primary model's answer."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def run(
        self, question: str, model_ids: list[str] | None = None
    ) -> list[VerifierAnswer]:
        """Collect independent answers to the original question from all verifiers."""

        if model_ids is not None:
            verifiers = []
            invalid_ids = []
            for model_id in model_ids:
                try:
                    verifier = self.client.registry.get(model_id)
                except KeyError:
                    invalid_ids.append(model_id)
                    continue
                if verifier.role != "verifier":
                    invalid_ids.append(model_id)
                    continue
                verifiers.append(verifier)
            if invalid_ids:
                raise ValueError(
                    f"Requested models are not configured verifiers: {sorted(invalid_ids)}"
                )
        else:
            verifiers = self.client.registry.verifier_models()
        if not verifiers:
            raise ValueError("No enabled verifier models are configured")

        answers: list[VerifierAnswer] = []
        for verifier in verifiers:
            answer = self.client.chat(
                verifier.id,
                question,
                system_prompt=(
                    "Answer the user's question independently. Do not assume any prior "
                    "answer or claim has been provided."
                ),
            )
            answers.append(VerifierAnswer(model_id=verifier.id, answer=answer))

        return answers
