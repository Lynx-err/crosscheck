"""Known-value tests for evaluation metrics and WikiBio label mapping."""

from __future__ import annotations

import pytest

from crosscheck.eval.metrics import (
    auprc,
    auroc,
    binarize_label,
    bootstrap_confidence_intervals,
    brier_score,
    compute_metrics,
    cross_validated_platt,
    expected_calibration_error,
)


@pytest.mark.parametrize(
    ("label", "variant", "expected"),
    [
        ("accurate", "major_and_minor", 0),
        ("minor_inaccurate", "major_and_minor", 1),
        ("major_inaccurate", "major_and_minor", 1),
        ("accurate", "major_only", 0),
        ("minor_inaccurate", "major_only", 0),
        ("major_inaccurate", "major_only", 1),
    ],
)
def test_binarize_wikibio_annotations(label: str, variant: str, expected: int) -> None:
    assert binarize_label(label, variant) == expected


def test_binarize_rejects_unknown_labels_and_variants() -> None:
    with pytest.raises(ValueError, match="Unknown WikiBio annotation"):
        binarize_label("unknown")
    with pytest.raises(ValueError, match="Unknown label variant"):
        binarize_label("accurate", "other")


def test_auroc_known_rank_example() -> None:
    assert auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


def test_auprc_known_example() -> None:
    assert auprc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(5 / 6)


def test_calibration_and_brier_known_values() -> None:
    labels = [0, 1]
    scores = [0.0, 1.0]
    assert expected_calibration_error(labels, scores, bins=2) == pytest.approx(0.0)
    assert brier_score(labels, scores) == pytest.approx(0.0)


def test_auroc_returns_none_for_single_class() -> None:
    assert auroc([1, 1], [0.2, 0.8]) is None


def test_bootstrap_confidence_intervals_are_deterministic_and_cover_point_estimate() -> None:
    labels = [0, 1, 0, 1, 0, 1]
    scores = [0.1, 0.8, 0.2, 0.7, 0.3, 0.9]
    groups = ["a", "a", "b", "b", "c", "c"]

    first = bootstrap_confidence_intervals(
        labels, scores, groups, n_resamples=1000, seed=77
    )
    second = bootstrap_confidence_intervals(
        labels, scores, groups, n_resamples=1000, seed=77
    )

    assert first == second
    assert first["auroc"] is not None
    assert first["auprc"] is not None
    assert first["auroc"][0] <= auroc(labels, scores) <= first["auroc"][1]


def test_cross_validated_platt_keeps_each_group_in_one_fold() -> None:
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    scores = [0.1, 0.9, 0.2, 0.8, 0.25, 0.75, 0.3, 0.7]
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]

    calibrated = cross_validated_platt(labels, scores, groups, seed=19)

    assert len(calibrated) == len(labels)
    assert all(0.0 <= score <= 1.0 for score in calibrated)
    assert calibrated[1] > calibrated[0]


def test_compute_metrics_reports_prevalence_and_only_calibrated_calibration_metrics() -> None:
    labels = [0, 1, 0, 1, 0, 1]
    scores = [0.1, 0.8, 0.2, 0.7, 0.3, 0.9]
    report = compute_metrics(
        labels,
        scores,
        groups=["a", "a", "b", "b", "c", "c"],
        bootstrap_resamples=1000,
    )

    assert report["positive_prevalence"] == pytest.approx(0.5)
    assert "ece_calibrated" in report
    assert "brier_calibrated" in report
    assert "ece" not in report
