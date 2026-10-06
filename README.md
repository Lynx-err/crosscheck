# CrossCheck

CrossCheck is an open-source research project for claim-level hallucination detection using disagreement across a diverse ensemble of small language models.

## Phase 4: WikiBio evaluation

The evaluation harness uses the official `potsawee/wiki_bio_gpt3_hallucination`
`evaluation` split. It scores the existing GPT-3 sentences against independent
verifier passages; it does not regenerate the reference GPT-3 passages. Verifier
prompts use the subject derived from the supplied Wikipedia biography.

The harness compares CrossCheck, a single-verifier pipeline, SelfCheck-NLI
(five temperature-1 samples from the configured primary model), and a seeded
random baseline. It reports AUROC, AUPRC, ECE, and Brier for both binary label
variants: `major_and_minor` and `major_only`.

The default NLI checkpoint is `cross-encoder/nli-deberta-v3-small`, a
DeBERTa-v3 model trained on SNLI and MultiNLI. It is loaded with Transformers
and scores premise/hypothesis pairs in batches. Verifier passages are split
into sentences, then overlapping token windows where necessary, so each pair
fits the model's 256-token limit; per-verifier entailment and contradiction are
the maxima across its windows. The report records how often whole-passage
scoring would have truncated. CPU inference is capped at four PyTorch threads;
the checkpoint is read from `config/models.yaml`, while token/thread limits can
be adjusted in `NLIScorer` without changing the dataset interface.

```powershell
python -m crosscheck.eval.run_eval --n-samples 20 --seed 42 --panel all --aggregator mean_entailment
```

`--panel` accepts `all` or comma-separated enabled verifier IDs. Supported
`--aggregator` values are `mean_entailment`, `min`, `contradiction_weighted`,
and `entropy`. Evaluation rows and NLI pairs are cached in SQLite; model calls
use the shared SQLite response cache. Completed sentence scores are checkpointed
in `results/.cache/wikibio_eval.sqlite3`, so rerunning the same seed and
configuration resumes from cached work. Results and per-stage latency/token
summaries are written under `results/`.

Run the Phase 4 diagnostic set (20 seeded WikiBio concepts and 50 HaluEval-QA
questions) with:

```powershell
python -m crosscheck.eval.run_diagnostics --wiki-samples 20 --qa-questions 50 --seed 42
```

An explicit `--panel` overrides the enabled default panel for that run and may
name configured-but-disabled verifiers. Pass comma-separated verifier IDs;
each explicit panel is saved to a distinct result JSON file:

```powershell
python -m crosscheck.eval.run_diagnostics --panel qwen2.5-3b,qwen2.5-1.5b
python -m crosscheck.eval.run_diagnostics --panel qwen2.5-3b,gemma-2-2b
python -m crosscheck.eval.run_diagnostics --panel qwen2.5-3b,phi-3.5-mini
```

The diagnostics report the configured parameter sizes. The requested panels
differ slightly in size, so family and size effects cannot be fully separated.

This prints raw AUROC/AUPRC with 1,000-resample concept/question-clustered
bootstrap intervals, the positive prevalence beside each AUPRC, and ECE/Brier
from out-of-group Platt calibration. It also reports NLI probability masses
by label, whole-passage truncation counts, and an equal-count same-family
versus cross-family panel comparison when two same-family verifiers are
enabled. If the configured panel cannot form that pair, the report marks that
comparison blocked and still reports the other diagnostics. Full results and
runtime/model call totals are saved to
`results/phase4_diagnostics_seed42.json`. Both model responses and NLI pairs
use the shared disk caches, so reruns reuse completed work.
