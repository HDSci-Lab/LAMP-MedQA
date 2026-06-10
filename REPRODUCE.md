# Reproducing the paper

This file lists the exact commands used to produce the numbers in
"LAMP-MedQA: A Lightweight Multi-Agent System for Patient-Oriented Medical
Question Answering". Run them from the repository root.

## Prerequisites

- Python 3.11, dependencies from `requirements.txt`.
- A CUDA GPU. The paper used a single H100 (80 GB); a 7B + 3.8B fp16
  pipeline run on 200 samples takes roughly 4–12 hours.
- `data/MeDiSumQA.jsonl` in place (see [data/README.md](data/README.md)).
- Environment variables:
  ```bash
  export HF_TOKEN=...            # Hugging Face read token (Qwen is gated)
  export OPENAI_API_KEY=...      # GPT-5 baseline + GPT-4 LLM-Judge scoring
  ```
- Long jobs are checkpointed. To resume an interrupted pipeline run, add
  `--resume <path_to_checkpoint.json>` to the same command.

> **MIMIC / PhysioNet compliance.** The steps below that hit OpenAI (the
> baselines in §2 and any `--llm-judge` flag) will send MeDiSumQA-derived
> text — i.e. MIMIC-IV-derived text — to a third-party API. Per the
> PhysioNet DUA this is **only permitted via an endpoint with verified zero
> data retention** (e.g. Azure OpenAI ZDR or an OpenAI Enterprise ZDR
> agreement); the paper's runs used a secure research workspace. A default
> consumer `OPENAI_API_KEY` is not sufficient. If you cannot meet that
> requirement, omit `--llm-judge` and skip §2. See the "Compliance" section
> of the README for the full policy.

## Order of operations

| # | Step | Outputs |
|---|---|---|
| 1 | Tune extraction threshold τ on dev set | `results/dev_threshold_tune.json` |
| 2 | Edit `EXTRACTION_THRESHOLDS["sbert_eq"]` in `evaluate_lightweight_pipeline.py` to the tuned τ (paper: 0.671) | — |
| 3 | Run baselines (GPT-5, Qwen-32B zero/one-shot) | `results/evaluate_200_predictions_*.json` |
| 4 | Run single-model ablations (Qwen-7B, Phi-3.5-mini zero-shot) | `results/ablation/evaluate_200_predictions_*.json` |
| 5 | Run LAMP-MedQA (no tools and with tools) | `results/final_results/lightweight_pipeline*_traces_*.json` |
| 6 | Run pipeline ablations (verifier/reviewer off) | `results/ablation/lightweight_pipeline_abl_*_traces_*.json` |
| 7 | Statistical tests vs baselines and ablations | `results/statistical_tests_results.json`, `results/ablation/statistical_tests_ablations.json` |

## 1. Tune the extraction threshold τ

Runs Stage-1 extraction once per dev sample (loads Qwen2.5-7B-Instruct
only — no OpenAI calls needed), picks τ by balanced-accuracy maximisation
(25th-percentile fallback if dev labels are degenerate).

```bash
python tune_extraction_threshold.py \
    --data data/MeDiSumQA.jsonl \
    --n-dev 20 --n-test 200 \
    --output results/dev_threshold_tune.json
```

Open `evaluate_lightweight_pipeline.py`, find `EXTRACTION_THRESHOLDS`, and
set `"sbert_eq"` to the value printed by the tuner. The paper uses 0.671.

## 2. Baselines (GPT-5, Qwen-32B zero-shot, Qwen-32B one-shot)

```bash
python evaluate_200.py \
    --data data/MeDiSumQA.jsonl \
    --output results \
    --n-dev 0 --n-test 200 \
    --llm-judge
```

## 3. Single-model ablations (Qwen2.5-7B-Instruct, Phi-3.5-Mini-Instruct, zero-shot)

```bash
python evaluate_200.py \
    --data data/MeDiSumQA.jsonl \
    --output results/ablation \
    --n-dev 0 --n-test 200 \
    --llm-judge \
    --approaches qwen7b_zero phi35_zero
```

## 4. LAMP-MedQA on the held-out test set

Headline number (no tools):

```bash
python evaluate_lightweight_pipeline.py \
    --data data/MeDiSumQA.jsonl \
    --output results/final_results \
    --split test --n-dev 20 --n-test 200 \
    --gate-mode sbert \
    --llm-judge --verbose
```

Tool-augmented variant (offline MedlinePlus glossary):

```bash
python evaluate_lightweight_pipeline.py \
    --data data/MeDiSumQA.jsonl \
    --output results/final_results \
    --split test --n-dev 20 --n-test 200 \
    --gate-mode sbert \
    --use-tools --glossary data/medlineplus_glossary.json \
    --llm-judge --verbose
```

## 5. Pipeline ablations

Without readability verifier:

```bash
python evaluate_lightweight_pipeline.py \
    --data data/MeDiSumQA.jsonl \
    --output results/ablation \
    --split test --n-dev 0 --n-test 200 \
    --gate-mode sbert \
    --no-readability-validation \
    --llm-judge --verbose
```

Without accuracy reviewer:

```bash
python evaluate_lightweight_pipeline.py \
    --data data/MeDiSumQA.jsonl \
    --output results/ablation \
    --split test --n-dev 0 --n-test 200 \
    --gate-mode sbert \
    --no-extraction-validation \
    --llm-judge --verbose
```

Both off:

```bash
python evaluate_lightweight_pipeline.py \
    --data data/MeDiSumQA.jsonl \
    --output results/ablation \
    --split test --n-dev 0 --n-test 200 \
    --gate-mode sbert \
    --no-extraction-validation --no-readability-validation \
    --llm-judge --verbose
```

## 6. Statistical tests

Main table (LAMP-MedQA vs GPT-5 / Qwen-32B baselines):

```bash
python statistical_tests.py \
    --baseline-predictions results/evaluate_200_predictions_<TIMESTAMP>.json \
    --pipeline-traces results/final_results/lightweight_pipeline_traces_<TIMESTAMP>.json \
    --baseline-models "GPT-5" "Qwen2.5-32B (zero-shot)" "Qwen2.5-32B (one-shot)" \
    --n-bootstrap 10000 \
    --device auto \
    --output results/statistical_tests_results.json
```

Ablation table (each ablation vs full LAMP-MedQA no-tools):

```bash
python statistical_tests_ablations.py \
    --baseline results/final_results/lightweight_pipeline_traces_<TIMESTAMP>.json \
    --baseline-name "lightweight (no tools)" \
    --ablations \
        results/ablation/lightweight_pipeline_abl_noreadval_traces_<TS>.json \
        results/ablation/lightweight_pipeline_abl_noaccval_traces_<TS>.json \
        results/ablation/lightweight_pipeline_abl_noaccval_noreadval_traces_<TS>.json \
    --extra-predictions results/ablation/evaluate_200_predictions_<TS>.json \
    --extra-predictions-models "Qwen2.5-7B (zero-shot)" "Phi-3.5-mini (zero-shot)" \
    --n-bootstrap 10000 --alpha 0.05 --m-total 90 \
    --device auto \
    --output results/ablation/statistical_tests_ablations.json
```

Significance markers in the paper denote paired bootstrap p-values after
Bonferroni correction (α = 0.05, m = 90); Wilcoxon signed-rank tests are
reported as a robustness check.

## Alternative extraction gate (parallel experiment)

The paper compares the SBERT-threshold gate against an LLM-as-classifier
gate (Model B returns `{"Classification_result": <bool>, "False_evidence": "..."}`
per candidate). To run the LLM-gate variant on the same configurations,
add `--gate-mode llm` in place of `--gate-mode sbert`. Output filenames
get an `_llmgate` infix so the two variants coexist in the same output
directories.
