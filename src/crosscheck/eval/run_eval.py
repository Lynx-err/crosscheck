"""CLI for cached WikiBio hallucination-baseline evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path

from crosscheck.aggregator import AggregationStrategy, Aggregator
from crosscheck.baselines.random_baseline import random_score
from crosscheck.baselines.selfcheck_nli import SelfCheckNLIBaseline
from crosscheck.baselines.single_verifier import SingleVerifierBaseline
from crosscheck.claim_extractor import ClaimExtractor
from crosscheck.config import ModelRegistry, load_model_registry
from crosscheck.llm_client import LLMCallStats, LLMClient
from crosscheck.nli_scorer import NLIScorer
from crosscheck.pipeline import CrossCheckPipeline
from crosscheck.verifier_panel import VerifierAnswer, VerifierPanel

from .datasets import WikiBioDataset, WikiBioRecord
from .metrics import binarize_label, compute_metrics

BASELINES = ("crosscheck", "single_verifier", "selfcheck_nli", "random")
LABEL_VARIANTS = ("major_and_minor", "major_only")


def _select_panel(registry: ModelRegistry, requested: str) -> list[str]:
    configured = [model.id for model in registry.verifier_models()]
    if requested.strip().lower() == "all":
        selected = configured
    else:
        selected = [model_id.strip() for model_id in requested.split(",") if model_id.strip()]
        unknown = set(selected) - set(configured)
        if unknown:
            raise ValueError(f"Panel contains unknown or disabled verifier ids: {sorted(unknown)}")
        if not selected:
            raise ValueError("--panel must be 'all' or a comma-separated verifier id list")
    return list(dict.fromkeys(selected))


def _run_key(
    records: list[WikiBioRecord],
    seed: int,
    panel_ids: list[str],
    primary_model: str,
    nli_model: str,
    strategy: AggregationStrategy,
) -> str:
    payload = {
        "dataset": "potsawee/wiki_bio_gpt3_hallucination",
        "indices": [record.wiki_bio_test_idx for record in records],
        "seed": seed,
        "panel": panel_ids,
        "primary": primary_model,
        "nli_model": nli_model,
        "nli_max_length": 256,
        "selfcheck_samples": 5,
        "prompt_version": 3,
        "aggregator": strategy.value,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _initialize_progress(database_path: Path) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sentence_predictions (
                run_key TEXT NOT NULL,
                dataset_idx INTEGER NOT NULL,
                sentence_idx INTEGER NOT NULL,
                baseline TEXT NOT NULL,
                score REAL NOT NULL,
                major_and_minor INTEGER NOT NULL,
                major_only INTEGER NOT NULL,
                PRIMARY KEY (run_key, dataset_idx, sentence_idx, baseline)
            )
            """
        )


def _read_prediction(
    database_path: Path, run_key: str, dataset_idx: int, sentence_idx: int, baseline: str
) -> float | None:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT score FROM sentence_predictions
            WHERE run_key=? AND dataset_idx=? AND sentence_idx=? AND baseline=?
            """,
            (run_key, dataset_idx, sentence_idx, baseline),
        ).fetchone()
    return float(row[0]) if row else None


def _save_prediction(
    database_path: Path,
    run_key: str,
    record: WikiBioRecord,
    sentence_idx: int,
    baseline: str,
    score: float,
) -> None:
    label = record.annotation[sentence_idx]
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO sentence_predictions
            (run_key, dataset_idx, sentence_idx, baseline, score, major_and_minor, major_only)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_key,
                record.wiki_bio_test_idx,
                sentence_idx,
                baseline,
                score,
                binarize_label(label, "major_and_minor"),
                binarize_label(label, "major_only"),
            ),
        )


def _write_stage_log(
    path: Path,
    run_key: str,
    dataset_idx: int,
    sentence_idx: int | None,
    baseline: str,
    stage: str,
    duration_ms: float,
    stats: list[LLMCallStats] | None = None,
) -> None:
    calls = stats or []
    event = {
        "run_key": run_key,
        "dataset_idx": dataset_idx,
        "sentence_idx": sentence_idx,
        "baseline": baseline,
        "stage": stage,
        "duration_ms": duration_ms,
        "llm_calls": len(calls),
        "cache_hits": sum(call.cache_hit for call in calls),
        "prompt_tokens": sum(call.prompt_tokens for call in calls),
        "completion_tokens": sum(call.completion_tokens for call in calls),
        "total_tokens": sum(call.total_tokens for call in calls),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def _metrics_from_progress(
    database_path: Path,
    run_key: str,
    records: list[WikiBioRecord],
) -> dict[str, dict[str, dict[str, float | list[float] | None]]]:
    report: dict[str, dict[str, dict[str, float | list[float] | None]]] = {}
    with sqlite3.connect(database_path) as connection:
        for baseline in BASELINES:
            rows = connection.execute(
                """
                SELECT score, major_and_minor, major_only, dataset_idx
                FROM sentence_predictions
                WHERE run_key=? AND baseline=?
                ORDER BY dataset_idx, sentence_idx
                """,
                (run_key, baseline),
            ).fetchall()
            expected_count = sum(len(record.gpt3_sentences) for record in records)
            if len(rows) != expected_count:
                raise RuntimeError(
                    f"Missing {baseline} scores: found {len(rows)} of {expected_count} sentences"
                )
            scores = [float(row[0]) for row in rows]
            labels: dict[str, list[int]] = {
                "major_and_minor": [int(row[1]) for row in rows],
                "major_only": [int(row[2]) for row in rows],
            }
            groups = [int(row[3]) for row in rows]
            report[baseline] = {
                variant: compute_metrics(
                    labels[variant],
                    scores,
                    groups=groups,
                    seed=int(run_key[:8], 16),
                )
                for variant in LABEL_VARIANTS
            }
    return report


def _print_metrics_table(
    report: dict[str, dict[str, dict[str, float | list[float] | None]]]
) -> None:
    print(
        "\nWikiBio metrics (rank metrics are raw scores with cluster-bootstrap 95% CI; "
        "ECE/Brier use out-of-group Platt calibration)"
    )
    print(
        f"{'Label variant':<17} {'Method':<17} {'AUROC [95% CI]':>25} "
        f"{'AUPRC [95% CI]':>25} {'Prev':>7} {'ECE*':>8} {'Brier*':>8}"
    )
    for variant in LABEL_VARIANTS:
        for baseline in BASELINES:
            metrics = report[baseline][variant]
            auc_interval = metrics["auroc_ci95"]
            ap_interval = metrics["auprc_ci95"]
            auc = "N/A" if metrics["auroc"] is None else f"{metrics['auroc']:.4f}"
            ap = "N/A" if metrics["auprc"] is None else f"{metrics['auprc']:.4f}"
            auc_text = (
                f"{auc} [{auc_interval[0]:.3f},{auc_interval[1]:.3f}]"
                if isinstance(auc_interval, list)
                else auc
            )
            ap_text = (
                f"{ap} [{ap_interval[0]:.3f},{ap_interval[1]:.3f}]"
                if isinstance(ap_interval, list)
                else ap
            )
            print(
                f"{variant:<17} {baseline:<17} "
                f"{auc_text:>25} {ap_text:>25} "
                f"{metrics['positive_prevalence']:>7.3f} "
                f"{metrics['ece_calibrated']:>8.4f} {metrics['brier_calibrated']:>8.4f}"
            )


def _evaluate_record(
    *,
    record: WikiBioRecord,
    client: LLMClient,
    pipeline: CrossCheckPipeline,
    panel: VerifierPanel,
    selfcheck: SelfCheckNLIBaseline,
    single_verifier: SingleVerifierBaseline,
    selected_panel: list[str],
    strategy: AggregationStrategy,
    seed: int,
    run_key: str,
    database_path: Path,
    stage_log_path: Path,
) -> None:
    concept_prompt = (
        f"Write a short Wikipedia-style passage about {record.concept()}. "
        "Use no more than three concise sentences."
    )
    needs: dict[str, list[int]] = {
        baseline: [
            index
            for index in range(len(record.gpt3_sentences))
            if _read_prediction(
                database_path, run_key, record.wiki_bio_test_idx, index, baseline
            )
            is None
        ]
        for baseline in BASELINES
    }
    if not any(needs.values()):
        return

    verifier_answers: list[VerifierAnswer] = []
    if needs["crosscheck"] or needs["single_verifier"]:
        event_start = len(client.call_events)
        stage_started = time.perf_counter()
        verifier_answers = panel.run(concept_prompt, model_ids=selected_panel)
        _write_stage_log(
            stage_log_path,
            run_key,
            record.wiki_bio_test_idx,
            None,
            "crosscheck",
            "verifier_generation",
            (time.perf_counter() - stage_started) * 1000,
            client.call_events[event_start:],
        )
    samples: list[VerifierAnswer] = []
    if needs["selfcheck_nli"]:
        event_start = len(client.call_events)
        stage_started = time.perf_counter()
        samples = selfcheck.sample_answers(concept_prompt)
        _write_stage_log(
            stage_log_path,
            run_key,
            record.wiki_bio_test_idx,
            None,
            "selfcheck_nli",
            "sample_generation",
            (time.perf_counter() - stage_started) * 1000,
            client.call_events[event_start:],
        )

    panel_indices = sorted(set(needs["crosscheck"]) | set(needs["single_verifier"]))
    panel_results = []
    if panel_indices:
        started = time.perf_counter()
        panel_results = pipeline.score_claims(
            concept_prompt,
            record.gpt3_text,
            [record.gpt3_sentences[index] for index in panel_indices],
            strategy,
            verifier_answers=verifier_answers,
        ).claims
        _write_stage_log(
            stage_log_path,
            run_key,
            record.wiki_bio_test_idx,
            None,
            "crosscheck_and_single_verifier",
            "batched_nli_and_aggregation",
            (time.perf_counter() - started) * 1000,
        )
    for sentence_idx, result in zip(panel_indices, panel_results):
        if sentence_idx in needs["crosscheck"]:
            _save_prediction(
                database_path,
                run_key,
                record,
                sentence_idx,
                "crosscheck",
                1.0 - result.agreement_score,
            )
        if sentence_idx in needs["single_verifier"]:
            score = single_verifier.hallucination_score_from_evidence(result, strategy)
            _save_prediction(
                database_path, run_key, record, sentence_idx, "single_verifier", score
            )

    selfcheck_indices = needs["selfcheck_nli"]
    if selfcheck_indices:
        started = time.perf_counter()
        scores = selfcheck.hallucination_scores(
            [record.gpt3_sentences[index] for index in selfcheck_indices],
            samples,
        )
        _write_stage_log(
            stage_log_path,
            run_key,
            record.wiki_bio_test_idx,
            None,
            "selfcheck_nli",
            "batched_nli_score",
            (time.perf_counter() - started) * 1000,
        )
        for sentence_idx, score in zip(selfcheck_indices, scores):
            _save_prediction(
                database_path, run_key, record, sentence_idx, "selfcheck_nli", score
            )

    for sentence_idx in needs["random"]:
        _save_prediction(
            database_path,
            run_key,
            record,
            sentence_idx,
            "random",
            random_score(seed, f"{record.wiki_bio_test_idx}:{sentence_idx}"),
        )


def run_evaluation(
    *,
    n_samples: int,
    seed: int,
    panel_arg: str,
    strategy_arg: str,
    config_path: str | None = None,
    output_dir: str | Path = "results",
) -> Path:
    """Run the selected cached WikiBio subset and write its metrics JSON."""

    registry = load_model_registry(config_path)
    selected_panel = _select_panel(registry, panel_arg)
    if not selected_panel:
        raise ValueError("At least one enabled verifier is required")
    primary_models = registry.primary_models()
    if not primary_models:
        raise ValueError("No enabled primary model is configured for SelfCheck sampling")
    primary_id = primary_models[0].id
    strategy = AggregationStrategy(strategy_arg)
    records = WikiBioDataset().sample(n_samples, seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path = output_path / ".cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    database_path = cache_path / "wikibio_eval.sqlite3"
    _initialize_progress(database_path)
    run_key = _run_key(records, seed, selected_panel, primary_id, registry.nli_model, strategy)
    stage_log_path = output_path / f"stage_log_{run_key}.jsonl"

    client = LLMClient(registry, timeout=300.0)
    nli_scorer = NLIScorer(registry.nli_model)
    verifier_panel = VerifierPanel(client)
    pipeline = CrossCheckPipeline(
        ClaimExtractor(client, primary_id),
        verifier_panel,
        nli_scorer,
        Aggregator(),
    )
    selfcheck = SelfCheckNLIBaseline(client, nli_scorer, primary_id)
    single = SingleVerifierBaseline(verifier_panel, pipeline, selected_panel[0])
    total_sentences = sum(len(record.gpt3_sentences) for record in records)
    print(
        f"Dataset: {len(records)} concepts, {total_sentences} sentences; "
        f"seed={seed}; panel={','.join(selected_panel)}; aggregator={strategy.value}"
    )
    overall_started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        _evaluate_record(
            record=record,
            client=client,
            pipeline=pipeline,
            panel=verifier_panel,
            selfcheck=selfcheck,
            single_verifier=single,
            selected_panel=selected_panel,
            strategy=strategy,
            seed=seed,
            run_key=run_key,
            database_path=database_path,
            stage_log_path=stage_log_path,
        )
        print(f"Completed concept {index}/{len(records)}: {record.concept()}")
        if index == 5 and index < len(records):
            elapsed = time.perf_counter() - overall_started
            estimate_seconds = elapsed / index * (len(records) - index)
            print(
                f"Estimated remaining runtime after first 5 samples: "
                f"{estimate_seconds / 60:.1f} minutes"
            )

    report = _metrics_from_progress(database_path, run_key, records)
    _print_metrics_table(report)
    result = {
        "dataset": "potsawee/wiki_bio_gpt3_hallucination",
        "split": "evaluation",
        "n_samples": len(records),
        "n_sentences": total_sentences,
        "seed": seed,
        "primary_model": primary_id,
        "panel": selected_panel,
        "nli_model": registry.nli_model,
        "nli_max_length": 256,
        "aggregator": strategy.value,
        "label_variants": report,
        "bootstrap_resamples": 1000,
        "calibration": "5-fold out-of-group Platt scaling for ECE/Brier only",
        "run_key": run_key,
        "stage_log": stage_log_path.name,
        "elapsed_seconds": time.perf_counter() - overall_started,
    }
    output_file = output_path / f"wikibio_{len(records)}_seed{seed}_{run_key}.json"
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_file}")
    return output_file


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI options and start WikiBio baseline evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--panel",
        default="all",
        help="all enabled verifiers or comma-separated configured verifier IDs",
    )
    parser.add_argument(
        "--aggregator",
        choices=[strategy.value for strategy in AggregationStrategy],
        default=AggregationStrategy.MEAN_ENTAILMENT.value,
    )
    parser.add_argument("--config", help="Path to model registry YAML")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)
    try:
        run_evaluation(
            n_samples=args.n_samples,
            seed=args.seed,
            panel_arg=args.panel,
            strategy_arg=args.aggregator,
            config_path=args.config,
            output_dir=args.output_dir,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
