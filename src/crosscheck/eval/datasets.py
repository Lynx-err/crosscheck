"""Cached loader and deterministic sampling for the WikiBio GPT-3 dataset."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DATASET_NAME = "potsawee/wiki_bio_gpt3_hallucination"
DATASET_CONFIG = "default"
DATASET_SPLIT = "evaluation"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


class WikiBioRecord(BaseModel):
    """One dataset biography and its sentence-level annotations."""

    model_config = ConfigDict(extra="ignore")

    gpt3_text: str
    wiki_bio_text: str
    gpt3_sentences: list[str]
    annotation: list[str]
    wiki_bio_test_idx: int
    gpt3_text_samples: list[str] = Field(default_factory=list)

    def concept(self) -> str:
        """Derive the subject name from the supplied ground-truth biography."""

        biography = re.sub(r"\s+", " ", self.wiki_bio_text.strip())
        subject_match = re.search(
            r"\s+(?:was|is|were|are)\s+|,\s*(?:born|died)\b",
            biography,
            flags=re.IGNORECASE,
        )
        if subject_match:
            subject = biography[: subject_match.start()]
        else:
            first_sentence = re.split(
                r"(?<=[!?])\s+|(?<=\.)\s+(?=[A-Z])",
                biography,
                maxsplit=1,
            )[0]
            subject = first_sentence
        subject = re.sub(r"\s*\([^)]*\)\s*$", "", subject).strip(" ,")
        return subject or biography

    def validate_sentences(self) -> None:
        """Ensure dataset annotations are aligned one-to-one with GPT-3 sentences."""

        if len(self.gpt3_sentences) != len(self.annotation):
            raise ValueError(
                f"Dataset record {self.wiki_bio_test_idx} has {len(self.gpt3_sentences)} "
                f"sentences but {len(self.annotation)} annotations"
            )
        valid_labels = {"accurate", "minor_inaccurate", "major_inaccurate"}
        invalid = set(self.annotation) - valid_labels
        if invalid:
            raise ValueError(
                f"Dataset record {self.wiki_bio_test_idx} has unknown annotations: "
                f"{sorted(invalid)}"
            )


class WikiBioDataset:
    """Fetch and persist the official dataset rows from Hugging Face's viewer API."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        page_size: int = 100,
        timeout: float = 60.0,
    ) -> None:
        if page_size < 1:
            raise ValueError("page_size must be at least 1")
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "crosscheck" / "datasets"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "wiki_bio_gpt3_hallucination_evaluation.json"
        self.page_size = page_size
        self.timeout = timeout

    def load(self) -> list[WikiBioRecord]:
        """Load all 238 evaluation rows, reusing the local raw-row cache."""

        if self.cache_file.exists():
            payload = json.loads(self.cache_file.read_text(encoding="utf-8"))
        else:
            payload = self._download_rows()
            temporary_file = self.cache_file.with_suffix(".json.tmp")
            temporary_file.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            temporary_file.replace(self.cache_file)

        try:
            records = [WikiBioRecord.model_validate(row) for row in payload]
        except ValidationError as exc:
            raise ValueError(f"Cached WikiBio rows do not match the dataset schema: {exc}") from exc
        for record in records:
            record.validate_sentences()
        return records

    def sample(self, n_samples: int, seed: int) -> list[WikiBioRecord]:
        """Choose a reproducible subset from the cached dataset rows."""

        records = self.load()
        if n_samples < 1 or n_samples > len(records):
            raise ValueError(f"n_samples must be between 1 and {len(records)}")
        return random.Random(seed).sample(records, n_samples)

    def _download_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        total: int | None = None
        with httpx.Client(timeout=self.timeout) as client:
            while total is None or offset < total:
                response = client.get(
                    ROWS_ENDPOINT,
                    params={
                        "dataset": DATASET_NAME,
                        "config": DATASET_CONFIG,
                        "split": DATASET_SPLIT,
                        "offset": offset,
                        "length": self.page_size,
                    },
                )
                response.raise_for_status()
                page = response.json()
                total = int(page["num_rows_total"])
                rows.extend(item["row"] for item in page["rows"])
                offset += len(page["rows"])
                if not page["rows"]:
                    break
        if total is None or len(rows) != total:
            raise RuntimeError(
                f"WikiBio download returned {len(rows)} rows, expected {total}"
            )
        return rows


class HaluEvalQARecord(BaseModel):
    """HaluEval QA question with its supplied correct and hallucinated answers."""

    question: str
    right_answer: str
    hallucinated_answer: str
    knowledge: str = ""


class HaluEvalQADataset:
    """Fetch the official HaluEval QA JSONL into a reusable local cache."""

    source_url = (
        "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
    )

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path.home() / ".cache" / "crosscheck" / "datasets"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "hal_eval_qa_data.jsonl"
        self.timeout = timeout

    def load(self) -> list[HaluEvalQARecord]:
        """Return the cached official JSONL examples, downloading them if absent."""

        if not self.cache_file.exists():
            temporary_file = self.cache_file.with_suffix(".jsonl.tmp")
            with httpx.stream("GET", self.source_url, timeout=self.timeout) as response:
                response.raise_for_status()
                with temporary_file.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
            temporary_file.replace(self.cache_file)

        records: list[HaluEvalQARecord] = []
        with self.cache_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(HaluEvalQARecord.model_validate_json(line))
                except ValidationError as exc:
                    raise ValueError(
                        f"Invalid HaluEval QA row at line {line_number}: {exc}"
                    ) from exc
        if not records:
            raise ValueError(f"HaluEval QA dataset is empty: {self.cache_file}")
        return records

    def sample(self, n_samples: int, seed: int) -> list[HaluEvalQARecord]:
        """Select a seeded subset without modifying any supplied answers."""

        records = self.load()
        if n_samples < 1 or n_samples > len(records):
            raise ValueError(f"n_samples must be between 1 and {len(records)}")
        return random.Random(seed).sample(records, n_samples)
