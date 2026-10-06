"""Tests for WikiBio row semantics and sentence/annotation alignment."""

from __future__ import annotations

import pytest

from crosscheck.eval.datasets import HaluEvalQADataset, HaluEvalQARecord, WikiBioRecord


def test_concept_is_derived_from_supplied_biography_not_generated_passage() -> None:
    record = WikiBioRecord(
        gpt3_text="Invented subject claim.",
        wiki_bio_text=(
            "Sir John Russell Reynolds, 1st Baronet (22 May 1828) was a British "
            "neurologist and physician."
        ),
        gpt3_sentences=["Invented subject claim."],
        annotation=["major_inaccurate"],
        wiki_bio_test_idx=1,
    )

    assert record.concept() == "Sir John Russell Reynolds, 1st Baronet"
    record.validate_sentences()


def test_concept_parser_does_not_split_on_abbreviations_in_birth_dates() -> None:
    record = WikiBioRecord(
        gpt3_text="Thomas Harriot was a poet.",
        wiki_bio_text=(
            "Thomas Harriot (Oxford, c. 1560–1621) was an English astronomer, "
            "mathematician, ethnographer, and translator."
        ),
        gpt3_sentences=["Thomas Harriot was a poet."],
        annotation=["major_inaccurate"],
        wiki_bio_test_idx=3,
    )

    assert record.concept() == "Thomas Harriot"


def test_dataset_rejects_misaligned_sentence_labels() -> None:
    record = WikiBioRecord(
        gpt3_text="A sentence.",
        wiki_bio_text="A subject was a person.",
        gpt3_sentences=["A sentence.", "Another sentence."],
        annotation=["accurate"],
        wiki_bio_test_idx=2,
    )

    with pytest.raises(ValueError, match="sentences but"):
        record.validate_sentences()


def test_dataset_rejects_unknown_annotation() -> None:
    record = WikiBioRecord(
        gpt3_text="A sentence.",
        wiki_bio_text="A subject was a person.",
        gpt3_sentences=["A sentence."],
        annotation=["wrong"],
        wiki_bio_test_idx=2,
    )

    with pytest.raises(ValueError, match="unknown annotations"):
        record.validate_sentences()


def test_halueval_qa_record_preserves_official_right_and_hallucinated_answers(
    tmp_path,
) -> None:
    record = HaluEvalQARecord.model_validate(
        {
            "knowledge": "Background context.",
            "question": "Which item came first?",
            "right_answer": "The earlier item.",
            "hallucinated_answer": "The later item came first.",
        }
    )
    assert record.right_answer == "The earlier item."
    assert record.hallucinated_answer == "The later item came first."
    dataset = HaluEvalQADataset(cache_dir=tmp_path)
    assert dataset.cache_file.parent == tmp_path
