"""Binary classification and calibration metrics for hallucination scores."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def binarize_label(label: str, variant: str = "major_and_minor") -> int:
    """Map WikiBio annotations to accurate/inaccurate or major-only labels."""

    normalized = label.strip().lower()
    if normalized not in {"accurate", "minor_inaccurate", "major_inaccurate"}:
        raise ValueError(f"Unknown WikiBio annotation: {label!r}")
    if variant == "major_and_minor":
        return int(normalized in {"minor_inaccurate", "major_inaccurate"})
    if variant == "major_only":
        return int(normalized == "major_inaccurate")
    raise ValueError(f"Unknown label variant: {variant!r}")


def _validate_inputs(labels: Sequence[int], scores: Sequence[float]) -> None:
    if not labels or len(labels) != len(scores):
        raise ValueError("labels and scores must be non-empty and have equal lengths")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must contain only 0 and 1")
    if any(not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("scores must be probabilities in [0, 1]")


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute AUROC using average ranks, returning None for single-class labels."""

    _validate_inputs(labels, scores)
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ordered = sorted(zip(scores, labels))
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        rank_sum += sum(label for _, label in ordered[index:end]) * average_rank
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def auprc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute average precision for the positive hallucination class."""

    _validate_inputs(labels, scores)
    positives = sum(labels)
    if positives == 0:
        return None

    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    seen = 0
    average_precision = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group_positives = sum(label for _, label in ordered[index:end])
        true_positives += group_positives
        seen += end - index
        average_precision += (group_positives / positives) * (true_positives / seen)
        index = end
    return average_precision


def expected_calibration_error(
    labels: Sequence[int], scores: Sequence[float], bins: int = 10
) -> float:
    """Compute equal-width-bin expected calibration error."""

    _validate_inputs(labels, scores)
    if bins < 1:
        raise ValueError("bins must be at least 1")
    total = len(labels)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        indices = [
            index
            for index, score in enumerate(scores)
            if lower <= score < upper or (bin_index == bins - 1 and score == 1.0)
        ]
        if indices:
            confidence = sum(scores[index] for index in indices) / len(indices)
            accuracy = sum(labels[index] for index in indices) / len(indices)
            error += len(indices) / total * abs(accuracy - confidence)
    return error


def brier_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute mean squared probability error."""

    _validate_inputs(labels, scores)
    return sum((label - score) ** 2 for label, score in zip(labels, scores)) / len(labels)


def _fit_platt(scores: Sequence[float], labels: Sequence[int]) -> tuple[float, float]:
    """Fit a one-dimensional logistic calibrator over clipped score logits."""

    logits = [
        math.log(min(1.0 - 1e-6, max(1e-6, score)) / (1.0 - min(1.0 - 1e-6, max(1e-6, score))))
        for score in scores
    ]
    intercept = math.log((sum(labels) + 0.5) / (len(labels) - sum(labels) + 0.5))
    coefficient = 0.0
    for _ in range(100):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for value, label in zip(logits, labels):
            linear = max(-30.0, min(30.0, intercept + coefficient * value))
            probability = 1.0 / (1.0 + math.exp(-linear))
            residual = label - probability
            weight = probability * (1.0 - probability)
            g0 += residual
            g1 += residual * value
            h00 += weight
            h01 += weight * value
            h11 += weight * value * value
        h00 += 1e-6
        h11 += 1e-6
        determinant = h00 * h11 - h01 * h01
        if determinant <= 1e-12:
            break
        delta0 = (h11 * g0 - h01 * g1) / determinant
        delta1 = (h00 * g1 - h01 * g0) / determinant
        intercept += delta0
        coefficient += delta1
        if max(abs(delta0), abs(delta1)) < 1e-7:
            break
    return intercept, coefficient


def cross_validated_platt(
    labels: Sequence[int],
    scores: Sequence[float],
    groups: Sequence[str | int],
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> list[float]:
    """Create out-of-fold Platt-calibrated scores, keeping groups intact."""

    _validate_inputs(labels, scores)
    if len(groups) != len(labels):
        raise ValueError("groups must match labels and scores in length")
    unique_groups = list(dict.fromkeys(groups))
    if len(unique_groups) < 2:
        raise ValueError("cross-validated calibration requires at least two groups")
    n_splits = min(n_splits, len(unique_groups))
    shuffled = unique_groups.copy()
    random.Random(seed).shuffle(shuffled)
    folds: list[list[str | int]] = [[] for _ in range(n_splits)]
    for index, group in enumerate(shuffled):
        folds[index % n_splits].append(group)

    calibrated = [0.0] * len(labels)
    for held_out in folds:
        held_out_set = set(held_out)
        train_indices = [i for i, group in enumerate(groups) if group not in held_out_set]
        test_indices = [i for i, group in enumerate(groups) if group in held_out_set]
        train_labels = [labels[i] for i in train_indices]
        if not any(train_labels) or all(train_labels):
            prior = (sum(train_labels) + 0.5) / (len(train_labels) + 1.0)
            for index in test_indices:
                calibrated[index] = prior
            continue
        intercept, coefficient = _fit_platt(
            [scores[i] for i in train_indices],
            train_labels,
        )
        for index in test_indices:
            score = min(1.0 - 1e-6, max(1e-6, scores[index]))
            value = math.log(score / (1.0 - score))
            linear = max(-30.0, min(30.0, intercept + coefficient * value))
            calibrated[index] = 1.0 / (1.0 + math.exp(-linear))
    return calibrated


def bootstrap_confidence_intervals(
    labels: Sequence[int],
    scores: Sequence[float],
    groups: Sequence[str | int] | None = None,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, list[float] | None]:
    """Cluster-bootstrap AUROC and AUPRC confidence intervals."""

    _validate_inputs(labels, scores)
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    cluster_ids = list(groups) if groups is not None else list(range(len(labels)))
    if len(cluster_ids) != len(labels):
        raise ValueError("groups must match labels and scores in length")
    samples_by_group: dict[str | int, list[int]] = {}
    for index, group in enumerate(cluster_ids):
        samples_by_group.setdefault(group, []).append(index)
    unique_groups = list(samples_by_group)
    if len(unique_groups) < 2:
        return {"auroc": None, "auprc": None}

    rng = random.Random(seed)
    auc_values: list[float] = []
    ap_values: list[float] = []
    attempts = 0
    max_attempts = n_resamples * 20
    while len(auc_values) < n_resamples and attempts < max_attempts:
        attempts += 1
        sampled_groups = [rng.choice(unique_groups) for _ in unique_groups]
        indices = [
            index
            for group in sampled_groups
            for index in samples_by_group[group]
        ]
        sample_labels = [labels[index] for index in indices]
        sample_scores = [scores[index] for index in indices]
        auc = auroc(sample_labels, sample_scores)
        ap = auprc(sample_labels, sample_scores)
        if auc is not None and ap is not None:
            auc_values.append(auc)
            ap_values.append(ap)
    if len(auc_values) < max(1, n_resamples // 2):
        return {"auroc": None, "auprc": None}

    lower_percentile = (1.0 - confidence) / 2.0
    upper_percentile = 1.0 - lower_percentile

    def interval(values: list[float]) -> list[float]:
        ordered = sorted(values)
        lower = ordered[round(lower_percentile * (len(ordered) - 1))]
        upper = ordered[round(upper_percentile * (len(ordered) - 1))]
        return [lower, upper]

    return {"auroc": interval(auc_values), "auprc": interval(ap_values)}


def compute_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    groups: Sequence[str | int] | None = None,
    bootstrap_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float | list[float] | None]:
    """Calculate ranking metrics, group-bootstrap CIs, prevalence, and OOF calibration."""

    _validate_inputs(labels, scores)
    group_ids: Sequence[str | int] = groups if groups is not None else list(range(len(labels)))
    calibrated = cross_validated_platt(labels, scores, group_ids, seed=seed)
    intervals = bootstrap_confidence_intervals(
        labels,
        scores,
        group_ids,
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    return {
        "auroc": auroc(labels, scores),
        "auprc": auprc(labels, scores),
        "positive_prevalence": sum(labels) / len(labels),
        "auroc_ci95": intervals["auroc"],
        "auprc_ci95": intervals["auprc"],
        "ece_calibrated": expected_calibration_error(labels, calibrated),
        "brier_calibrated": brier_score(labels, calibrated),
        "calibration": "5-fold out-of-group Platt scaling",
    }
