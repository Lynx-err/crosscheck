"""Run the six targeted Phase 4 diagnostics on WikiBio and HaluEval-QA."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from crosscheck.baselines.random_baseline import random_score
from crosscheck.baselines.selfcheck_nli import SelfCheckNLIBaseline
from crosscheck.config import ModelConfig, ModelRegistry, load_model_registry
from crosscheck.llm_client import LLMClient
from crosscheck.nli_scorer import ClaimEvidence, NLIScorer
from crosscheck.verifier_panel import VerifierPanel

from .datasets import HaluEvalQADataset, HaluEvalQARecord, WikiBioDataset, WikiBioRecord
from .metrics import binarize_label, compute_metrics

METHODS = (
    "crosscheck",
    "single_verifier",
    "selfcheck_nli",
    "random",
)


def _family_panels(registry: ModelRegistry) -> tuple[list[str], list[str]]:
    """Choose two same-family models and an equal-sized diverse comparison panel."""

    enabled = registry.verifier_models()
    families: dict[str, list[str]] = defaultdict(list)
    for model in enabled:
        if model.family:
            families[model.family].append(model.id)
    repeated_family = next(
        (models for models in families.values() if len(models) >= 2),
        None,
    )
    if repeated_family is None:
        raise ValueError(
            "Same-family comparison requires two enabled verifier models with a shared family"
        )
    first_family = next(
        model.family for model in enabled if model.id == repeated_family[0]
    )
    other_family_model = next(
        model
        for model in enabled
        if model.family is not None and model.family != first_family
    )
    return repeated_family[:2], [repeated_family[0], other_family_model.id]


def _family_panel_status(
    registry: ModelRegistry,
) -> tuple[list[str], list[str], str | None]:
    try:
        same_family, cross_family = _family_panels(registry)
    except ValueError as exc:
        return [], [], str(exc)
    return same_family, cross_family, None


def _model_families(registry: ModelRegistry) -> dict[str, str]:
    return {
        model.id: model.family or "unspecified"
        for model in registry.verifier_models()
    }


def _resolve_panel_models(
    registry: ModelRegistry, panel_ids: Sequence[str] | None
) -> list[ModelConfig]:
    if panel_ids is None:
        models = registry.verifier_models()
        if not models:
            raise ValueError("No enabled verifier models are configured")
        return models
    if not panel_ids or len(set(panel_ids)) != len(panel_ids):
        raise ValueError("An explicit panel must contain unique verifier IDs")
    models: list[ModelConfig] = []
    for model_id in panel_ids:
        model = registry.get(model_id)
        if model.role != "verifier":
            raise ValueError(f"Selected model is not a verifier: {model_id}")
        models.append(model)
    return models


def _panel_details(models: Sequence[ModelConfig]) -> list[dict[str, str | None]]:
    return [
        {
            "id": model.id,
            "family": model.family,
            "parameter_size": model.parameter_size,
        }
        for model in models
    ]


def _explicit_panel_status(models: Sequence[ModelConfig]) -> dict[str, Any]:
    families = {model.family for model in models}
    comparison_type = "same-family" if len(families) == 1 else "cross-family"
    return {
        "status": "compare_across_explicit_panel_runs",
        "comparison_type": comparison_type,
        "panel": _panel_details(models),
        "note": (
            "Panels differ slightly in parameter size, so family and size effects "
            "cannot be fully separated."
        ),
    }


def _mean_entailment(
    evidence: ClaimEvidence, model_ids: set[str] | None = None
) -> float:
    values = [
        verifier.probabilities.entailment
        for verifier in evidence.verifiers
        if model_ids is None or verifier.model_id in model_ids
    ]
    if not values:
        raise ValueError("No verifier evidence selected for entailment aggregation")
    return sum(values) / len(values)


def _append_prediction(
    values: dict[str, list[Any]],
    label: int,
    score: float,
    group: str | int,
) -> None:
    values["labels"].append(label)
    values["scores"].append(score)
    values["groups"].append(group)


def _metric_report(
    data: dict[str, dict[str, list[Any]]], seed: int
) -> dict[str, dict[str, float | list[float] | None]]:
    return {
        method: compute_metrics(
            values["labels"],
            values["scores"],
            groups=values["groups"],
            seed=seed,
        )
        for method, values in data.items()
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _stage_event(
    client: LLMClient,
    stage: str,
    event_start: int,
    started: float,
) -> dict[str, int | float | str]:
    calls = client.call_events[event_start:]
    return {
        "stage": stage,
        "duration_ms": (time.perf_counter() - started) * 1000,
        "model_calls": len(calls),
        "prompt_tokens": sum(call.prompt_tokens for call in calls),
        "completion_tokens": sum(call.completion_tokens for call in calls),
    }


def _format_interval(value: float | None, interval: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(interval, list):
        return f"{value:.3f} [{interval[0]:.3f},{interval[1]:.3f}]"
    return f"{value:.3f}"


def _print_table(
    title: str, report: dict[str, dict[str, float | list[float] | None]]
) -> None:
    print(
        f"\n{title} (raw AUROC/AUPRC, cluster bootstrap 95% CI; "
        "ECE/Brier are out-of-group Platt calibrated)"
    )
    print(
        f"{'Method':<18} {'AUROC [95% CI]':>25} {'AUPRC [95% CI]':>25} "
        f"{'Prev':>7} {'ECE*':>8} {'Brier*':>8}"
    )
    for method, metrics in report.items():
        print(
            f"{method:<18} "
            f"{_format_interval(metrics['auroc'], metrics['auroc_ci95']):>25} "
            f"{_format_interval(metrics['auprc'], metrics['auprc_ci95']):>25} "
            f"{metrics['positive_prevalence']:>7.3f} "
            f"{metrics['ece_calibrated']:>8.4f} "
            f"{metrics['brier_calibrated']:>8.4f}"
        )


def _class_distribution(
    masses: dict[str, dict[str, list[float]]]
) -> dict[str, dict[str, float | int | None]]:
    return {
        label: {
            "count": len(values["entailment"]),
            "mean_entailment": _mean(values["entailment"]),
            "mean_neutral": _mean(values["neutral"]),
            "mean_contradiction": _mean(values["contradiction"]),
        }
        for label, values in masses.items()
    }


def _new_masses() -> dict[str, dict[str, list[float]]]:
    return {
        label: {"entailment": [], "neutral": [], "contradiction": []}
        for label in ("accurate", "hallucinated")
    }


def _record_masses(
    masses: dict[str, dict[str, list[float]]],
    category: str,
    evidence: ClaimEvidence,
) -> None:
    if not evidence.verifiers:
        return
    for name in ("entailment", "neutral", "contradiction"):
        mass = sum(
            getattr(item.probabilities, name) for item in evidence.verifiers
        ) / len(evidence.verifiers)
        masses[category][name].append(mass)


def _wiki_bio_diagnostic(
    records: list[WikiBioRecord],
    registry: ModelRegistry,
    client: LLMClient,
    scorer: NLIScorer,
    panel: VerifierPanel,
    selfcheck: SelfCheckNLIBaseline,
    *,
    seed: int,
    panel_models: Sequence[ModelConfig] | None = None,
) -> dict[str, Any]:
    selected_models = list(panel_models) if panel_models is not None else registry.verifier_models()
    model_ids = [model.id for model in selected_models]
    same_ids, cross_ids, family_block_reason = (
        ([], [], "explicit panel comparison is reported across panel runs")
        if panel_models is not None
        else _family_panel_status(registry)
    )
    methods = list(METHODS)
    if panel_models is None and family_block_reason is None:
        methods.extend(("same_family_2", "cross_family_2"))
    single_id = model_ids[0]
    predictions: dict[str, dict[str, list[Any]]] = {
        method: {"labels": [], "scores": [], "groups": []}
        for method in methods
    }
    masses = _new_masses()
    stage_events: list[dict[str, int | float | str]] = []
    pretruncation_pairs = 0
    total_pairs = 0
    total_windows = 0
    concept_started = time.perf_counter()

    for record_index, record in enumerate(records, start=1):
        prompt = f"Write a short Wikipedia-style passage about {record.concept()}."
        call_start = len(client.call_events)
        stage_started = time.perf_counter()
        verifier_answers = panel.run(prompt, model_ids=model_ids)
        stage_events.append(
            _stage_event(client, "wiki_verifier_generation", call_start, stage_started)
        )
        call_start = len(client.call_events)
        stage_started = time.perf_counter()
        sample_answers = selfcheck.sample_answers(prompt)
        stage_events.append(
            _stage_event(client, "wiki_selfcheck_generation", call_start, stage_started)
        )
        print(
            f"WikiBio verifier/sample generation {record_index}/{len(records)}: "
            f"{record.concept()}"
        )
        sentences = record.gpt3_sentences
        call_start = len(client.call_events)
        stage_started = time.perf_counter()
        panel_evidence = scorer.score(sentences, verifier_answers)
        stage_events.append(
            _stage_event(client, "wiki_panel_nli", call_start, stage_started)
        )
        call_start = len(client.call_events)
        stage_started = time.perf_counter()
        sample_evidence = scorer.score(sentences, sample_answers)
        stage_events.append(
            _stage_event(client, "wiki_selfcheck_nli", call_start, stage_started)
        )
        for sentence_index, (_, evidence, sampled) in enumerate(
            zip(sentences, panel_evidence, sample_evidence)
        ):
            label_text = record.annotation[sentence_index]
            hallucinated = binarize_label(label_text, "major_and_minor")
            major_only = binarize_label(label_text, "major_only")
            group = record.wiki_bio_test_idx
            category = "hallucinated" if hallucinated else "accurate"
            _record_masses(masses, category, evidence)
            total_pairs += len(evidence.verifiers)
            total_windows += sum(item.window_count for item in evidence.verifiers)
            pretruncation_pairs += sum(
                item.would_truncate_without_windowing for item in evidence.verifiers
            )

            scores = {
                "crosscheck": 1.0 - _mean_entailment(evidence),
                "single_verifier": 1.0 - _mean_entailment(evidence, {single_id}),
                "selfcheck_nli": 1.0 - _mean_entailment(sampled),
                "random": random_score(seed, f"{group}:{sentence_index}"),
            }
            if panel_models is None and family_block_reason is None:
                scores.update(
                    {
                        "same_family_2": 1.0
                        - _mean_entailment(evidence, set(same_ids)),
                        "cross_family_2": 1.0
                        - _mean_entailment(evidence, set(cross_ids)),
                    }
                )
            for method, score in scores.items():
                _append_prediction(
                    predictions[method], hallucinated, score, group
                )
                predictions[method].setdefault("major_only_labels", []).append(major_only)
            if record_index == 5 and record_index < len(records):
                elapsed = time.perf_counter() - concept_started
                remaining = elapsed / record_index * (len(records) - record_index)
                print(
                    f"Estimated remaining WikiBio runtime: {remaining / 60:.1f} minutes"
                )

    all_report = _metric_report(predictions, seed)
    major_report = {
        method: compute_metrics(
            values["major_only_labels"],
            values["scores"],
            groups=values["groups"],
            seed=seed,
        )
        for method, values in predictions.items()
    }
    family_report: dict[str, Any]
    if panel_models is not None:
        family_report = _explicit_panel_status(selected_models)
    elif family_block_reason is None:
        family_report = {
            "status": "completed",
            "same_family_panel": same_ids,
            "cross_family_panel": cross_ids,
            "metrics": {
                method: all_report[method]
                for method in ("same_family_2", "cross_family_2")
            },
        }
    else:
        family_report = {"status": "blocked", "reason": family_block_reason}
    truncation = {
        "pairs": total_pairs,
        "pairs_that_would_truncate_as_whole_passage": pretruncation_pairs,
        "pair_truncation_rate": pretruncation_pairs / total_pairs if total_pairs else 0.0,
        "windows_scored": total_windows,
        "max_length": scorer.max_length,
        "aggregation": "max entailment and max contradiction across all passage windows",
    }
    print(
        f"WikiBio whole-passage truncation before fix: {pretruncation_pairs}/{total_pairs} "
        f"({truncation['pair_truncation_rate']:.1%}); scored windows={total_windows}"
    )
    _print_table("WikiBio major + minor inaccurate", all_report)
    _print_table("WikiBio major inaccurate only", major_report)
    if panel_models is not None:
        print(
            f"\nExplicit {family_report['comparison_type']} panel: "
            + " + ".join(
                f"{model.id} ({model.parameter_size or 'size unspecified'})"
                for model in selected_models
            )
        )
    elif family_block_reason is None:
        _print_table(
            "Same-family vs cross-family (equal panel size 2)",
            family_report["metrics"],
        )
    else:
        print(f"\nSame-family vs cross-family comparison BLOCKED: {family_block_reason}")
    distributions = _class_distribution(masses)
    print("\nCrossCheck mean NLI mass by sentence class:")
    for category, values in distributions.items():
        print(
            f"  {category}: n={values['count']} "
            f"entailment={values['mean_entailment']:.3f} "
            f"neutral={values['mean_neutral']:.3f} "
            f"contradiction={values['mean_contradiction']:.3f}"
        )
    return {
        "n_concepts": len(records),
        "n_sentences": sum(len(record.gpt3_sentences) for record in records),
        "seed": seed,
        "panel": model_ids,
        "model_families": {
            model.id: model.family or "unspecified" for model in selected_models
        },
        "panel_details": _panel_details(selected_models),
        "same_family_panel": same_ids,
        "cross_family_panel": cross_ids,
        "family_comparison_status": family_report,
        "all_methods_major_and_minor": all_report,
        "all_methods_major_only": major_report,
        "family_comparison": family_report,
        "nli_mass_by_sentence_class": distributions,
        "truncation_diagnostic": truncation,
        "stage_events": stage_events,
    }


def _qa_diagnostic(
    records: list[HaluEvalQARecord],
    registry: ModelRegistry,
    client: LLMClient,
    scorer: NLIScorer,
    panel: VerifierPanel,
    selfcheck: SelfCheckNLIBaseline,
    *,
    seed: int,
    panel_models: Sequence[ModelConfig] | None = None,
) -> dict[str, Any]:
    qa_started = time.perf_counter()
    selected_models = list(panel_models) if panel_models is not None else registry.verifier_models()
    model_ids = [model.id for model in selected_models]
    same_ids, cross_ids, family_block_reason = (
        ([], [], "explicit panel comparison is reported across panel runs")
        if panel_models is not None
        else _family_panel_status(registry)
    )
    methods = list(METHODS)
    if panel_models is None and family_block_reason is None:
        methods.extend(("same_family_2", "cross_family_2"))
    single_id = model_ids[0]
    predictions: dict[str, dict[str, list[Any]]] = {
        method: {"labels": [], "scores": [], "groups": []}
        for method in methods
    }
    masses = _new_masses()
    stage_events: list[dict[str, int | float | str]] = []
    pretruncation_pairs = 0
    total_pairs = 0
    total_windows = 0

    for question_index, record in enumerate(records):
        prompt = f"Answer this question independently: {record.question}"
        question_call_start = len(client.call_events)
        call_start = len(client.call_events)
        stage_started = time.perf_counter()
        verifier_answers = panel.run(prompt, model_ids=model_ids)
        stage_events.append(
            _stage_event(client, "qa_verifier_generation", call_start, stage_started)
        )
        call_start = len(client.call_events)
        stage_started = time.perf_counter()
        sample_answers = selfcheck.sample_answers(prompt)
        stage_events.append(
            _stage_event(client, "qa_selfcheck_generation", call_start, stage_started)
        )
        candidates = [record.right_answer, record.hallucinated_answer]
        call_start = len(client.call_events)
        stage_started = time.perf_counter()
        evidence_rows = scorer.score(candidates, verifier_answers)
        stage_events.append(_stage_event(client, "qa_panel_nli", call_start, stage_started))
        call_start = len(client.call_events)
        stage_started = time.perf_counter()
        sample_rows = scorer.score(candidates, sample_answers)
        stage_events.append(
            _stage_event(client, "qa_selfcheck_nli", call_start, stage_started)
        )
        for answer_index, (_, evidence, sampled) in enumerate(
            zip(candidates, evidence_rows, sample_rows)
        ):
            label = answer_index
            category = "hallucinated" if label else "accurate"
            _record_masses(masses, category, evidence)
            total_pairs += len(evidence.verifiers)
            total_windows += sum(item.window_count for item in evidence.verifiers)
            pretruncation_pairs += sum(
                item.would_truncate_without_windowing for item in evidence.verifiers
            )
            values = {
                "crosscheck": 1.0 - _mean_entailment(evidence),
                "single_verifier": 1.0 - _mean_entailment(evidence, {single_id}),
                "selfcheck_nli": 1.0 - _mean_entailment(sampled),
                "random": random_score(seed, f"qa:{question_index}:{answer_index}"),
            }
            if panel_models is None and family_block_reason is None:
                values.update(
                    {
                        "same_family_2": 1.0
                        - _mean_entailment(evidence, set(same_ids)),
                        "cross_family_2": 1.0
                        - _mean_entailment(evidence, set(cross_ids)),
                    }
                )
            for method, score in values.items():
                _append_prediction(
                    predictions[method], label, score, question_index
                )
        if question_index == 4 and len(records) > 5:
            print(
                f"Estimated remaining HaluEval-QA runtime after first 5 questions: "
                f"{(time.perf_counter() - qa_started) / 5 * (len(records) - 5) / 60:.1f} "
                "minutes"
            )
        if (question_index + 1) % 5 == 0:
            print(
                f"HaluEval-QA {question_index + 1}/{len(records)} questions; "
                f"cached/live model calls={len(client.call_events) - question_call_start}"
            )

    report = _metric_report(predictions, seed)
    distributions = _class_distribution(masses)
    _print_table("HaluEval-QA (right vs hallucinated answers)", report)
    print("\nHaluEval-QA mean NLI mass by candidate class:")
    for category, values in distributions.items():
        print(
            f"  {category}: n={values['count']} "
            f"entailment={values['mean_entailment']:.3f} "
            f"neutral={values['mean_neutral']:.3f} "
            f"contradiction={values['mean_contradiction']:.3f}"
        )
    return {
        "dataset": "RUCAIBox/HaluEval QA",
        "n_questions": len(records),
        "n_candidates": 2 * len(records),
        "seed": seed,
        "panel": model_ids,
        "metrics": report,
        "family_comparison_status": (
            _explicit_panel_status(selected_models)
            if panel_models is not None
            else
            {"status": "completed", "same_family_panel": same_ids, "cross_family_panel": cross_ids}
            if family_block_reason is None
            else {"status": "blocked", "reason": family_block_reason}
        ),
        "panel_details": _panel_details(selected_models),
        "nli_mass_by_answer_class": distributions,
        "truncation_diagnostic": {
            "pairs": total_pairs,
            "pairs_that_would_truncate_as_whole_passage": pretruncation_pairs,
            "pair_truncation_rate": pretruncation_pairs / total_pairs if total_pairs else 0.0,
            "windows_scored": total_windows,
            "max_length": scorer.max_length,
        },
        "stage_events": stage_events,
    }


def run_diagnostics(
    *,
    wiki_samples: int = 20,
    qa_questions: int = 50,
    seed: int = 42,
    output_dir: str | Path = "results",
    config_path: str | None = None,
    panel_ids: Sequence[str] | None = None,
) -> Path:
    """Run diagnostics 1–6 and persist all tables in a JSON report."""

    registry = load_model_registry(config_path)
    selected_models = _resolve_panel_models(registry, panel_ids)
    wiki_records = WikiBioDataset().sample(wiki_samples, seed)
    qa_records = HaluEvalQADataset().sample(qa_questions, seed)
    client = LLMClient(registry, timeout=300.0)
    scorer = NLIScorer(registry.nli_model, batch_size=16, cpu_threads=4)
    panel = VerifierPanel(client)
    primary = next(iter(registry.primary_models()), None)
    if primary is None:
        raise ValueError("No enabled primary model configured")
    selfcheck = SelfCheckNLIBaseline(client, scorer, primary.id)
    started = time.perf_counter()

    wiki_report = _wiki_bio_diagnostic(
        wiki_records,
        registry,
        client,
        scorer,
        panel,
        selfcheck,
        seed=seed,
        panel_models=selected_models if panel_ids is not None else None,
    )
    qa_started = time.perf_counter()
    qa_report = _qa_diagnostic(
        qa_records,
        registry,
        client,
        scorer,
        panel,
        selfcheck,
        seed=seed,
        panel_models=selected_models if panel_ids is not None else None,
    )
    report = {
        "seed": seed,
        "nli_model": registry.nli_model,
        "nli_max_length": scorer.max_length,
        "bootstrap_resamples": 1000,
        "calibration": "5-fold out-of-group Platt scaling; raw AUROC/AUPRC",
        "panel": _panel_details(selected_models),
        "panel_size_caveat": (
            "Panels differ slightly in parameter size, so family and size effects "
            "cannot be fully separated."
        ),
        "wiki_bio": wiki_report,
        "halueval_qa": qa_report,
        "elapsed_seconds": time.perf_counter() - started,
        "qa_elapsed_seconds": time.perf_counter() - qa_started,
        "llm_calls": len(client.call_events),
        "llm_cache_hits": sum(call.cache_hit for call in client.call_events),
        "prompt_tokens": sum(call.prompt_tokens for call in client.call_events),
        "completion_tokens": sum(call.completion_tokens for call in client.call_events),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    panel_suffix = (
        ""
        if panel_ids is None
        else "_panel_" + "_".join(panel_ids)
    )
    output_file = output_path / f"phase4_diagnostics_seed{seed}{panel_suffix}.json"
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDiagnostic report saved to {output_file}")
    return output_file


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for the targeted Phase 4 diagnostics, deliberately not a 200-row run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki-samples", type=int, default=20)
    parser.add_argument("--qa-questions", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--panel",
        help=(
            "Comma-separated verifier IDs to run explicitly. Explicit selection can "
            "include configured-but-disabled verifiers; without this flag the enabled "
            "default panel is unchanged."
        ),
    )
    parser.add_argument("--config")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)
    if args.wiki_samples > 20:
        parser.error("This diagnostics command is capped at 20 WikiBio concepts")
    if args.qa_questions != 50:
        parser.error("This diagnostics command requires exactly 50 QA questions")
    run_diagnostics(
        wiki_samples=args.wiki_samples,
        qa_questions=args.qa_questions,
        seed=args.seed,
        output_dir=args.output_dir,
        config_path=args.config,
        panel_ids=(
            [model_id.strip() for model_id in args.panel.split(",") if model_id.strip()]
            if args.panel is not None
            else None
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
