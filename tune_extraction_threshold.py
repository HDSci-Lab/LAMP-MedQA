"""
Tune the SBERT(extraction, question) gate threshold on the dev split.
======================================================================

The lightweight pipeline's extraction validator was changed to a single
reference-free signal: SBERT cosine similarity between the extraction and
the patient's question.  This script picks the threshold tau on the
20-sample dev split, with no reference leakage to inference time.

Procedure
---------
For each dev sample, generate a single first-attempt extraction with
Model A (Qwen2.5-7B-Instruct) and record:

  sbert_eq  = SBERT(extraction, question)              # the new gate signal
  sari      = SARI(extraction, reference, ds)          # tuning only -- gold-aware
  token_f1  = token-F1(extraction, reference)          # tuning only
  sbert_pr  = SBERT(extraction, reference)             # tuning only
  good      = (sari > 45) AND (token_f1 > 0.30) AND (sbert_pr > 0.65)

`good` is the binary label given by the OLD reference-based gate.  We then
choose tau to maximise balanced accuracy of (sbert_eq > tau) against `good`
across the dev set.  If `good` is degenerate (all-true or all-false), we
fall back to tau = 25th percentile of sbert_eq on dev (so ~75 % pass on
first attempt).

The chosen threshold is written to results/dev_threshold_tune.json along
with the per-sample table, the method, balanced accuracy, and the dev
pass-rate.  Wire the value into EXTRACTION_THRESHOLDS["sbert_eq"] in
evaluate_lightweight_pipeline.py.

Usage
-----
    python tune_extraction_threshold.py \
        --data data/MeDiSumQA.jsonl \
        --n-dev 20 \
        --n-test 200 \
        --output results/dev_threshold_tune.json
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np

# Heavy imports (Model A loader, prompts, metric helpers, dev split).
from evaluate_lightweight_pipeline import (
    MODEL_A_ID,
    EXTRACT_SYSTEM,
    EXTRACT_PROMPT,
    load_model,
    LangChainLocalLLM,
    build_chain,
    load_and_split,
    sample_sari,
    sample_token_f1,
    sample_sbert,
)


GOOD_SARI = 45.0
GOOD_TF1 = 0.30
GOOD_SBERT_PR = 0.65


def best_threshold(scores, labels):
    """Pick tau that maximises balanced accuracy of (s > tau) vs labels.

    Returns (tau, balanced_accuracy) or (None, None) if labels are
    degenerate.
    """
    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, None

    candidates = sorted(set(scores))
    # Also try thresholds infinitesimally below the smallest and above the
    # largest, so 'all pass' / 'all fail' are considered.
    candidates = [candidates[0] - 1e-6] + candidates + [candidates[-1] + 1e-6]

    best_tau, best_bal = None, -1.0
    for tau in candidates:
        preds = [s > tau for s in scores]
        tp = sum(p and l for p, l in zip(preds, labels))
        tn = sum((not p) and (not l) for p, l in zip(preds, labels))
        bal = 0.5 * (tp / n_pos + tn / n_neg)
        if bal > best_bal:
            best_tau, best_bal = float(tau), float(bal)
    return best_tau, best_bal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/MeDiSumQA.jsonl")
    parser.add_argument("--n-dev", type=int, default=20,
                        help="Dev split size (default: 20, matching evaluate_lightweight_pipeline.py)")
    parser.add_argument("--n-test", type=int, default=200,
                        help="Test split size (must match the test run for split alignment)")
    parser.add_argument("--output", default="results/dev_threshold_tune.json")
    args = parser.parse_args()

    print(f"Loading dev split ({args.n_dev} samples) from {args.data}...")
    dev_data, _ = load_and_split(args.data, args.n_dev, args.n_test)
    print(f"Dev split: {len(dev_data)} samples")

    print(f"\nLoading Model A: {MODEL_A_ID}...")
    tok_a, model_a = load_model(MODEL_A_ID)
    llm_a = LangChainLocalLLM(tok_a, model_a, system_prompt=EXTRACT_SYSTEM)
    extractor = build_chain(EXTRACT_PROMPT, llm_a)
    print("Model A loaded.")

    rows = []
    for i, ex in enumerate(dev_data):
        question = ex.get("Question", "").strip()
        ds = ex.get("discharge_summary", "")
        reference = ex.get("Answer", "").strip()
        ds_trunc = ds[:42000] + "..." if len(ds) > 42000 else ds

        extraction = extractor.invoke({
            "question": question,
            "discharge_summary": ds_trunc,
        })

        sbert_eq = sample_sbert(extraction, question)
        sari = sample_sari(extraction, reference, ds)
        tf1 = sample_token_f1(extraction, reference)
        sbert_pr = sample_sbert(extraction, reference)
        good = bool((sari > GOOD_SARI) and (tf1 > GOOD_TF1) and (sbert_pr > GOOD_SBERT_PR))

        rows.append({
            "index": i,
            "note_id": ex.get("note_id", ""),
            "question": question,
            "extraction": extraction,
            "sbert_eq": float(sbert_eq),
            "sari": float(sari),
            "token_f1": float(tf1),
            "sbert_pr": float(sbert_pr),
            "good": good,
        })
        print(f"  [{i+1:>2}/{len(dev_data)}] sbert_eq={sbert_eq:.3f} "
              f"sari={sari:5.1f} tf1={tf1:.2f} sbert_pr={sbert_pr:.2f} good={good}")

    scores = [r["sbert_eq"] for r in rows]
    labels = [r["good"] for r in rows]
    n_good = sum(labels)
    print(f"\nDev set: {n_good}/{len(rows)} 'good' extractions per the old reference-based gate.")

    tau, bal_acc = best_threshold(scores, labels)
    if tau is None:
        method = "p25_fallback"
        tau = float(np.percentile(scores, 25))
        bal_acc = None
        print(f"  Labels degenerate (all-good or all-bad on dev) -- "
              f"falling back to 25th-percentile rule.")
    else:
        method = "balanced_accuracy_max"

    pass_rate = sum(s > tau for s in scores) / len(scores)
    print(f"\nChosen tau = {tau:.4f}  (method = {method})")
    if bal_acc is not None:
        print(f"  balanced accuracy on dev = {bal_acc:.3f}")
    print(f"  dev pass-rate (sbert_eq > tau) = {pass_rate:.2%}")

    out = {
        "model_a": MODEL_A_ID,
        "data_path": args.data,
        "n_dev": len(rows),
        "n_test_used_for_split": args.n_test,
        "n_good_per_old_gate": int(n_good),
        "good_definition": {
            "sari_gt": GOOD_SARI,
            "token_f1_gt": GOOD_TF1,
            "sbert_pr_gt": GOOD_SBERT_PR,
        },
        "method": method,
        "tau": float(tau),
        "balanced_accuracy_on_dev": (float(bal_acc) if bal_acc is not None else None),
        "dev_pass_rate": float(pass_rate),
        "samples": rows,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {args.output}")
    print(f"\nNext step: open evaluate_lightweight_pipeline.py and set")
    print(f"  EXTRACTION_THRESHOLDS = {{'sbert_eq': {tau:.2f}}}")
    print(f"then re-run the affected configurations on the test split.")


if __name__ == "__main__":
    main()
