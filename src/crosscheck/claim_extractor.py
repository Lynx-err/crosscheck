"""LLM-based extraction and validation of atomic answer claims."""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .llm_client import LLMClient


class ClaimExtraction(BaseModel):
    """Validated list of atomic claims extracted from an answer."""

    model_config = ConfigDict(extra="forbid")

    claims: list[Annotated[str, Field(min_length=1)]]

    @field_validator("claims")
    @classmethod
    def normalize_claims(cls, claims: list[str]) -> list[str]:
        normalized = [claim.strip() for claim in claims]
        if any(not claim for claim in normalized):
            raise ValueError("claims cannot contain blank strings")
        return normalized


class ClaimExtractionError(ValueError):
    """Raised when model output remains invalid after the configured retries."""


class ClaimExtractor:
    """Extract atomic factual statements from an answer using a configured model."""

    def __init__(self, client: LLMClient, model_id: str, max_retries: int = 2) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        self.client = client
        self.model_id = model_id
        self.max_retries = max_retries

    def extract(self, question: str, answer: str) -> ClaimExtraction:
        """Return schema-validated claims grounded only in the supplied answer."""

        prompt = (
            "Extract the answer's distinct, independently verifiable factual claims. "
            "Keep each claim atomic, preserve its meaning, and do not add facts. "
            "Return only a JSON object with a 'claims' array of strings.\n\n"
            f"Question:\n{question}\n\nAnswer:\n{answer}"
        )
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "claim_extraction",
                "strict": True,
                "schema": ClaimExtraction.model_json_schema(),
            },
        }
        last_error: json.JSONDecodeError | ValidationError | None = None

        for attempt in range(self.max_retries):
            if attempt:
                prompt += (
                    "\n\nYour previous response did not match the required JSON schema. "
                    "Return corrected JSON only, with each claim as a non-empty string."
                )

            content = self.client.chat(
                self.model_id,
                prompt,
                system_prompt="You extract concise atomic factual claims from answers.",
                temperature=0.0,
                response_format=response_format,
            )
            try:
                return ClaimExtraction.model_validate_json(content)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc

        raise ClaimExtractionError(
            f"Could not extract valid claims after {self.max_retries} attempts: {last_error}"
        ) from last_error
