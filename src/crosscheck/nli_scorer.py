"""Batched claim-evidence scoring with a configured Transformers NLI model."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .verifier_panel import VerifierAnswer


class NLIProbabilities(BaseModel):
    """Entailment, neutral, and contradiction probabilities for one claim-answer pair."""

    model_config = ConfigDict(extra="forbid")

    entailment: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)
    contradiction: float = Field(ge=0.0, le=1.0)


class VerifierEvidence(BaseModel):
    """A verifier's evidence and NLI stance for a single claim."""

    model_id: str
    answer: str
    probabilities: NLIProbabilities
    window_count: int = 1
    would_truncate_without_windowing: bool = False

    @property
    def stance(self) -> str:
        """Return the highest-probability stance with stable tie-breaking."""

        scores = self.probabilities.model_dump()
        return max(("entailment", "neutral", "contradiction"), key=scores.__getitem__)


class ClaimEvidence(BaseModel):
    """NLI evidence collected for a claim across the verifier panel."""

    claim: str
    verifiers: list[VerifierEvidence]


class NLIScorer:
    """Score claim/verifier-answer pairs in batches using the configured NLI model."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        *,
        cpu_threads: int = 4,
        max_length: int = 256,
        cache_dir: str | Path | None = None,
        pair_classifier: Callable[
            [list[tuple[str, str]]], list[dict[str, float]]
        ] | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if cpu_threads < 1:
            raise ValueError("cpu_threads must be at least 1")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        self.model_name = model_name
        self.batch_size = batch_size
        self.cpu_threads = cpu_threads
        self.max_length = max_length
        self._pair_classifier = pair_classifier
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "crosscheck"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "nli_cache.sqlite3"
        with closing(sqlite3.connect(self.cache_path)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS nli_cache "
                "(cache_key TEXT PRIMARY KEY, probabilities TEXT NOT NULL)"
            )
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None
        self._label_indices: dict[str, int] | None = None

    def score(
        self, claims: list[str], verifier_answers: list[VerifierAnswer]
    ) -> list[ClaimEvidence]:
        """Return per-claim probabilities for every verifier answer."""

        if not claims:
            return []
        if not verifier_answers:
            raise ValueError("At least one verifier answer is required for NLI scoring")

        real_model = self._pair_classifier is None
        if real_model:
            self._load_model()

        windows_by_pair: list[list[str]] = []
        truncation_by_pair: list[bool] = []
        pairs: list[tuple[str, str]] = []
        for claim in claims:
            for answer in verifier_answers:
                if real_model:
                    windows, would_truncate = self._split_passage(answer.answer, claim)
                else:
                    windows, would_truncate = [answer.answer], False
                windows_by_pair.append(windows)
                truncation_by_pair.append(would_truncate)
                pairs.extend((window, claim) for window in windows)
        window_probabilities = self._score_cached_pairs(pairs)

        results: list[ClaimEvidence] = []
        offset = 0
        window_offset = 0
        for claim in claims:
            evidence: list[VerifierEvidence] = []
            for index, answer in enumerate(verifier_answers):
                pair_index = offset + index
                windows = windows_by_pair[pair_index]
                scores = window_probabilities[window_offset : window_offset + len(windows)]
                window_offset += len(windows)
                max_entailment = max(score["entailment"] for score in scores)
                max_contradiction = max(score["contradiction"] for score in scores)
                evidence.append(
                    VerifierEvidence(
                        model_id=answer.model_id,
                        answer=answer.answer,
                        probabilities=NLIProbabilities(
                            entailment=max_entailment,
                            contradiction=max_contradiction,
                            neutral=max(0.0, 1.0 - max_entailment - max_contradiction),
                        ),
                        window_count=len(windows),
                        would_truncate_without_windowing=truncation_by_pair[pair_index],
                    )
                )
            results.append(ClaimEvidence(claim=claim, verifiers=evidence))
            offset += len(verifier_answers)
        return results

    def _split_passage(self, passage: str, hypothesis: str) -> tuple[list[str], bool]:
        """Split a verifier passage into token-bounded sentences/windows."""

        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError("NLI tokenizer was not initialized")
        full_pair = tokenizer(
            passage,
            hypothesis,
            truncation=False,
            add_special_tokens=True,
        )["input_ids"]
        would_truncate = len(full_pair) > self.max_length
        hypothesis_tokens = tokenizer.encode(hypothesis, add_special_tokens=False)
        special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
        source_budget = self.max_length - len(hypothesis_tokens) - special_tokens
        if source_budget < 1:
            raise ValueError(
                "Claim exceeds NLI max_length after pair special tokens; "
                "increase max_length or shorten the claim"
            )

        sentence_parts = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", passage)
            if sentence.strip()
        ]
        if not sentence_parts:
            sentence_parts = [passage]

        windows: list[str] = []
        for sentence in sentence_parts:
            token_ids = tokenizer.encode(sentence, add_special_tokens=False)
            if len(token_ids) <= source_budget:
                windows.append(sentence)
                continue
            step = max(1, source_budget // 2)
            starts = list(range(0, max(1, len(token_ids) - source_budget + 1), step))
            final_start = max(0, len(token_ids) - source_budget)
            if not starts or starts[-1] != final_start:
                starts.append(final_start)
            windows.extend(
                tokenizer.decode(
                    token_ids[start : start + source_budget],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                for start in starts
            )
        return windows, would_truncate

    def _score_cached_pairs(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        keys = [
            hashlib.sha256(
                json.dumps(
                    [self.model_name, self.max_length, premise, hypothesis],
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            for premise, hypothesis in pairs
        ]
        with closing(sqlite3.connect(self.cache_path)) as connection:
            cached_rows = connection.execute(
                f"SELECT cache_key, probabilities FROM nli_cache WHERE cache_key IN "
                f"({','.join('?' for _ in keys)})",
                keys,
            ).fetchall() if keys else []
        cached = {key: json.loads(value) for key, value in cached_rows}
        missing_pairs: list[tuple[str, str]] = []
        missing_keys: list[str] = []
        for key, pair in zip(keys, pairs):
            if key not in cached:
                missing_keys.append(key)
                missing_pairs.append(pair)

        if missing_pairs:
            fresh = self._classify_pairs(missing_pairs)
            if len(fresh) != len(missing_keys):
                raise ValueError(
                    f"NLI classifier returned {len(fresh)} results for {len(missing_keys)} pairs"
                )
            with closing(sqlite3.connect(self.cache_path)) as connection, connection:
                connection.executemany(
                    "INSERT OR REPLACE INTO nli_cache "
                    "(cache_key, probabilities) VALUES (?, ?)",
                    [
                        (key, json.dumps(probabilities, sort_keys=True))
                        for key, probabilities in zip(missing_keys, fresh)
                    ],
                )
            cached.update(zip(missing_keys, fresh))

        return [cached[key] for key in keys]

    def _classify_pairs(
        self, pairs: list[tuple[str, str]]
    ) -> list[dict[str, float]]:
        if self._pair_classifier is not None:
            results: list[dict[str, float]] = []
            for start in range(0, len(pairs), self.batch_size):
                results.extend(self._pair_classifier(pairs[start : start + self.batch_size]))
            return results

        self._load_model()
        import torch

        results = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            premises, hypotheses = zip(*batch)
            encoded = self._tokenizer(
                list(premises),
                list(hypotheses),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self._model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1).cpu().tolist()

            label_indices = self._label_indices
            if label_indices is None:
                raise RuntimeError("NLI model labels were not initialized")
            for scores in probabilities:
                results.append(
                    {
                        label: float(scores[label_indices[label]])
                        for label in ("entailment", "neutral", "contradiction")
                    }
                )
        return results

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "NLI scoring requires the 'torch' and 'transformers' dependencies"
            ) from exc

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._device.type == "cpu":
            torch.set_num_threads(min(self.cpu_threads, os.cpu_count() or 1))
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()

        labels = self._model.config.id2label
        label_indices: dict[str, int] = {}
        for index, raw_label in labels.items():
            normalized = str(raw_label).strip().lower()
            if normalized.startswith("label_"):
                raise ValueError(
                    f"NLI checkpoint {self.model_name!r} has unmapped class {raw_label!r}; "
                    "expected entailment, neutral, and contradiction labels"
                )
            if normalized in {"entailment", "neutral", "contradiction"}:
                label_indices[normalized] = int(index)
        expected_labels = {"entailment", "neutral", "contradiction"}
        if label_indices.keys() != expected_labels:
            raise ValueError(
                f"NLI checkpoint {self.model_name!r} must expose exactly the labels "
                "entailment, neutral, and contradiction"
            )
        self._label_indices = label_indices
