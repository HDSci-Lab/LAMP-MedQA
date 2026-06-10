"""
Statistical significance tests for Medical QA evaluation results.

Computes per-sample metrics from saved predictions, then runs:
  1. Paired bootstrap resampling (standard in ACL papers)
  2. Wilcoxon signed-rank test (non-parametric paired test)

Compares each approach pair-wise against a chosen baseline.

Usage:
  python statistical_tests.py \
      --baseline-predictions results/evaluate_200_predictions_TIMESTAMP.json \
      --pipeline-traces results/lightweight_pipeline_traces_TIMESTAMP.json \
      [--pipeline-tools-traces results/lightweight_pipeline_tools_traces_TIMESTAMP.json] \
      [--n-bootstrap 10000] \
      [--alpha 0.05]
"""

import json
import re
import os
import argparse
import random
import warnings
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
from scipy import stats

try:
    import torch
except ImportError:
    torch = None

try:
    import textstat
except ImportError:
    textstat = None
    warnings.warn("textstat not installed — FKGL per-sample scores will be unavailable")

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None
    warnings.warn("rouge_score not installed — ROUGE-L per-sample scores will be unavailable")

try:
    from bert_score import score as bert_score_fn
except ImportError:
    bert_score_fn = None
    warnings.warn("bert_score not installed — BERTScore per-sample scores will be unavailable")

try:
    from sacrebleu.metrics import BLEU
except ImportError:
    BLEU = None
    warnings.warn("sacrebleu not installed — BLEU per-sample scores will be unavailable")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    SentenceTransformer = None
    warnings.warn("sentence-transformers not installed — SBERT per-sample scores will be unavailable")


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ============================================================================
# PER-SAMPLE METRIC FUNCTIONS
# ============================================================================

def _ngrams(text: str, n: int):
    tokens = re.findall(r"\w+", text.lower())
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def per_sample_sari(preds, refs, originals) -> List[float]:
    """Compute SARI per sample (averaged over n-gram orders 1-4)."""
    N = 4
    scores = []
    for orig, pred, ref in zip(originals, preds, refs):
        sent = 0.0
        for n in range(1, N+1):
            o = set(_ngrams(orig, n))
            p = set(_ngrams(pred, n))
            r = set(_ngrams(ref, n))
            keep_p = p & o
            keep_r = r & o
            keep = len(keep_p & keep_r) / len(keep_p) if keep_p else (1.0 if not keep_r else 0.0)
            add_p = p - o
            add_r = r - o
            add = len(add_p & add_r) / len(add_p) if add_p else 1.0
            del_p = o - p
            del_r = o - r
            delete = len(del_p & del_r) / len(del_r) if del_r else 1.0
            sent += (keep + add + delete) / 3
        scores.append((sent / N) * 100.0)
    return scores


def per_sample_fkgl(preds) -> List[float]:
    """Compute FKGL per sample."""
    if textstat is None:
        return [0.0] * len(preds)
    return [textstat.flesch_kincaid_grade(p) for p in preds]


def per_sample_token_f1(preds, refs) -> List[float]:
    """Compute Token F1 per sample."""
    def _f1(a, b):
        ta = re.findall(r"\w+", a.lower())
        tb = re.findall(r"\w+", b.lower())
        common = Counter(ta) & Counter(tb)
        n = sum(common.values())
        if n == 0:
            return 0.0
        p = n / len(ta) if ta else 0.0
        r = n / len(tb) if tb else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return [_f1(p, r) for p, r in zip(preds, refs)]


def per_sample_bleu(preds, refs) -> List[float]:
    """Compute sentence BLEU per sample, scaled to [0, 1]."""
    if BLEU is None:
        return [0.0] * len(preds)
    bleu = BLEU(effective_order=True)
    return [bleu.sentence_score(p, [r]).score / 100.0 for p, r in zip(preds, refs)]


def per_sample_rouge_l(preds, refs) -> List[float]:
    """Compute ROUGE-L F1 per sample."""
    if rouge_scorer is None:
        return [0.0] * len(preds)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return [scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(preds, refs)]


def per_sample_bert_score(preds, refs, device: str = "cpu") -> List[float]:
    """Compute BERTScore F1 per sample."""
    if bert_score_fn is None:
        return [0.0] * len(preds)
    P, R, F1 = bert_score_fn(preds, refs, lang='en',
                              model_type='distilbert-base-uncased',
                              device=device, verbose=False)
    return F1.tolist()


def per_sample_sbert(preds, refs, sbert_model) -> List[float]:
    """Compute SBERT cosine similarity per sample."""
    embp = sbert_model.encode(preds, convert_to_numpy=True)
    embr = sbert_model.encode(refs, convert_to_numpy=True)
    return [float(cosine_similarity([a], [b])[0, 0]) for a, b in zip(embp, embr)]


def per_sample_output_words(preds) -> List[float]:
    """Compute output length in words per sample."""
    return [float(len(re.findall(r"\w+", p))) for p in preds]


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def paired_bootstrap(scores_a: np.ndarray, scores_b: np.ndarray,
                     n_bootstrap: int = 10000, seed: int = SEED) -> dict:
    """
    Paired bootstrap resampling test.
    Tests H0: mean(scores_a) == mean(scores_b).
    Returns p-value and confidence interval for the difference (a - b).

    Standard method for NLP evaluation (Koehn, 2004; Berg-Kirkpatrick et al., 2012).
    """
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    assert len(scores_b) == n

    observed_diff = np.mean(scores_a) - np.mean(scores_b)

    # Bootstrap the difference
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        diffs[i] = np.mean(scores_a[idx]) - np.mean(scores_b[idx])

    # Two-sided p-value: proportion of bootstrap diffs on the other side of 0
    if observed_diff >= 0:
        p_value = np.mean(diffs <= 0) * 2  # two-sided
    else:
        p_value = np.mean(diffs >= 0) * 2
    p_value = min(p_value, 1.0)

    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "n_bootstrap": int(n_bootstrap),
        "significant": p_value < 0.05,
    }


def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """
    Wilcoxon signed-rank test for paired samples.
    Non-parametric alternative to paired t-test.
    """
    diff = scores_a - scores_b
    # Remove zeros (ties)
    nonzero = diff[diff != 0]
    if len(nonzero) < 10:
        return {"statistic": None, "p_value": 1.0, "significant": False,
                "note": "Too few non-zero differences"}

    stat, p_value = stats.wilcoxon(nonzero, alternative='two-sided')
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[dict]:
    """Apply Bonferroni correction for multiple comparisons."""
    m = len(p_values)
    adjusted = [min(p * m, 1.0) for p in p_values]
    return [{"original_p": p, "adjusted_p": adj, "significant": adj < alpha}
            for p, adj in zip(p_values, adjusted)]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_baseline_predictions(path: str) -> Dict[str, List[dict]]:
    """Load predictions from evaluate_200.py output."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pipeline_traces(path: str) -> List[dict]:
    """Load traces from evaluate_lightweight_pipeline.py output."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Traces may be nested under a split key
    if isinstance(data, dict):
        for key in ["test", "traces", "dev"]:
            if key in data:
                data = data[key]
                break
        if isinstance(data, dict):
            for key in ["traces", "predictions"]:
                if key in data:
                    data = data[key]
                    break
    return data


def _entry_key(entry: dict) -> str:
    """Return a stable identifier for a sample: note_id if available, else question text."""
    nid = entry.get("note_id", "")
    if nid:
        return nid
    return entry.get("question", entry.get("Question", "")).strip()


def extract_preds_refs_origs(entries: List[dict]):
    """Extract predictions, references, originals, keys, and per-sample judge scores."""
    preds = [e.get("prediction", "") or "" for e in entries]
    refs = [e.get("reference", "") for e in entries]
    origs = [e.get("original", "") for e in entries]
    keys = [_entry_key(e) for e in entries]
    judge = [e.get("llm_judge_score") for e in entries]
    return preds, refs, origs, keys, judge


def extract_from_traces(traces: List[dict]):
    """Extract predictions, references, originals, keys, and per-sample judge scores."""
    preds, refs, origs, keys, judge = [], [], [], [], []
    for t in traces:
        preds.append(t.get("final_answer", "") or t.get("prediction", "") or "")
        refs.append(t.get("reference", "") or "")
        origs.append(t.get("original", "") or t.get("discharge_summary", "") or "")
        keys.append(_entry_key(t))
        judge.append(t.get("llm_judge_score"))
    return preds, refs, origs, keys, judge


def align_approaches(approaches: Dict[str, tuple]) -> Dict[str, tuple]:
    """
    Align all approaches so they cover the same samples in the same order.
    Each value is (preds, refs, origs, keys, judge_scores).
    Finds the intersection of keys across all approaches and reorders each
    approach to match.  Warns if samples are dropped.
    """
    # Collect key sets
    key_sets = {name: set(data[3]) for name, data in approaches.items()}
    common_keys = set.intersection(*key_sets.values()) if key_sets else set()

    if not common_keys:
        raise ValueError("No overlapping samples found across approaches — "
                         "check that both evaluations used the same data split "
                         "(same --n-dev and --n-test values).")

    for name, ks in key_sets.items():
        diff = len(ks) - len(common_keys)
        if diff > 0:
            warnings.warn(f"  {name}: dropping {diff} samples not present in all approaches")

    # Use key order from the first approach for determinism
    first_name = next(iter(approaches))
    ordered_keys = [k for k in approaches[first_name][3] if k in common_keys]

    aligned = {}
    for name, (preds, refs, origs, keys, judge) in approaches.items():
        lookup = {k: i for i, k in enumerate(keys)}
        indices = [lookup[k] for k in ordered_keys]
        aligned[name] = (
            [preds[i] for i in indices],
            [refs[i] for i in indices],
            [origs[i] for i in indices],
            ordered_keys,
            [judge[i] for i in indices],
        )

    print(f"  Aligned {len(ordered_keys)} common samples across {len(approaches)} approaches")
    return aligned


# ============================================================================
# MAIN
# ============================================================================

def compute_per_sample_metrics(preds, refs, originals,
                                sbert_model=None,
                                bert_device: str = "cpu",
                                metric_names=None) -> Dict[str, List[float]]:
    """Compute all per-sample metrics."""
    if metric_names is None:
        metric_names = ["sari", "fkgl", "token_f1", "bleu", "rouge_l", "bert_score",
                        "sbert", "output_words"]

    results = {}
    if "sari" in metric_names:
        print("    Computing per-sample SARI...")
        results["sari"] = per_sample_sari(preds, refs, originals)
    if "fkgl" in metric_names:
        print("    Computing per-sample FKGL...")
        results["fkgl"] = per_sample_fkgl(preds)
    if "token_f1" in metric_names:
        print("    Computing per-sample Token F1...")
        results["token_f1"] = per_sample_token_f1(preds, refs)
    if "bleu" in metric_names:
        print("    Computing per-sample BLEU...")
        results["bleu"] = per_sample_bleu(preds, refs)
    if "rouge_l" in metric_names:
        print("    Computing per-sample ROUGE-L...")
        results["rouge_l"] = per_sample_rouge_l(preds, refs)
    if "bert_score" in metric_names and bert_score_fn is not None:
        print("    Computing per-sample BERTScore...")
        results["bert_score"] = per_sample_bert_score(preds, refs, device=bert_device)
    if "sbert" in metric_names and sbert_model is not None:
        print("    Computing per-sample SBERT...")
        results["sbert"] = per_sample_sbert(preds, refs, sbert_model)
    if "output_words" in metric_names:
        print("    Computing per-sample output words...")
        results["output_words"] = per_sample_output_words(preds)

    return results


def run_pairwise_tests(metrics_a: Dict[str, List[float]],
                       metrics_b: Dict[str, List[float]],
                       name_a: str, name_b: str,
                       n_bootstrap: int = 10000) -> Dict[str, dict]:
    """Run all statistical tests between two systems for all shared metrics."""
    results = {}
    shared_metrics = set(metrics_a.keys()) & set(metrics_b.keys())

    for metric in sorted(shared_metrics):
        arr_a = np.array(metrics_a[metric])
        arr_b = np.array(metrics_b[metric])
        assert len(arr_a) == len(arr_b), f"Mismatched lengths for {metric}"

        result = {
            "mean_a": float(np.mean(arr_a)),
            "mean_b": float(np.mean(arr_b)),
            "bootstrap": paired_bootstrap(arr_a, arr_b, n_bootstrap),
            "wilcoxon": wilcoxon_test(arr_a, arr_b),
        }

        results[metric] = result

    return results


def print_results(all_results: dict, alpha: float = 0.05, n_bootstrap: int = 10000,
                  m_total: int = None):
    """Print formatted results table.

    m_total overrides Bonferroni m. If None, m = number of tests in this run.
    Useful for paper-wide consistency (e.g. m=90 across the whole paper).
    """
    # Collect all p-values for Bonferroni correction
    all_p_values = []
    p_value_labels = []

    for comparison, metrics in all_results.items():
        for metric, result in metrics.items():
            all_p_values.append(result["bootstrap"]["p_value"])
            p_value_labels.append(f"{comparison} / {metric}")

    if m_total is None:
        m_total = len(all_p_values)
    corrections = [
        {"original_p": p, "adjusted_p": min(p * m_total, 1.0),
         "significant": min(p * m_total, 1.0) < alpha}
        for p in all_p_values
    ]

    print("\n" + "=" * 100)
    print("STATISTICAL SIGNIFICANCE RESULTS")
    print("=" * 100)

    idx = 0
    for comparison, metrics in all_results.items():
        print(f"\n{'─' * 100}")
        print(f"  {comparison}")
        print(f"{'─' * 100}")
        print(f"  {'Metric':<12} {'Mean A':>8} {'Mean B':>8} {'Diff':>8} "
              f"{'Bootstrap p':>12} {'Wilcoxon p':>12} {'Bonf. p':>12} {'Sig?':>6}")
        print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*12} {'─'*12} {'─'*12} {'─'*6}")

        for metric in sorted(metrics.keys()):
            r = metrics[metric]
            boot_p = r["bootstrap"]["p_value"]
            wilc_p = r["wilcoxon"]["p_value"]
            bonf = corrections[idx]
            sig = "*" if bonf["significant"] else ""

            print(f"  {metric:<12} {r['mean_a']:>8.4f} {r['mean_b']:>8.4f} "
                  f"{r['bootstrap']['observed_diff']:>+8.4f} "
                  f"{boot_p:>12.4f} {wilc_p:>12.4f} {bonf['adjusted_p']:>12.4f} "
                  f"{sig:>6}")

            # Print 95% CI
            ci = r["bootstrap"]
            print(f"  {'':>12} {'95% CI':>17}: [{ci['ci_95_lower']:+.4f}, {ci['ci_95_upper']:+.4f}]")

            idx += 1

    print(f"\n{'─' * 100}")
    print(f"  * = significant after Bonferroni correction (alpha={alpha}, "
          f"m={m_total} tests; this run produced {len(all_p_values)} p-values)")
    print(f"  Bootstrap: {n_bootstrap} resamples")
    print(f"{'=' * 100}\n")


def main():
    parser = argparse.ArgumentParser(description="Statistical significance tests")
    parser.add_argument("--baseline-predictions", required=True,
                        help="Path to evaluate_200 predictions JSON")
    parser.add_argument("--pipeline-traces", default=None,
                        help="Path to lightweight pipeline traces JSON (no tools)")
    parser.add_argument("--pipeline-tools-traces", default=None,
                        help="Path to lightweight pipeline traces JSON (with tools)")
    parser.add_argument("--n-bootstrap", type=int, default=10000,
                        help="Number of bootstrap resamples")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level")
    parser.add_argument("--m-total", type=int, default=None,
                        help="Override Bonferroni m (paper-wide test count). "
                             "Defaults to the number of tests in this run.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Device for heavy metrics (BERTScore/SBERT)")
    parser.add_argument("--baseline-models", nargs="+",
                        default=["GPT-5", "Qwen2.5-32B (zero-shot)", "Qwen2.5-32B (one-shot)"],
                        help="Exact baseline model names to include")
    parser.add_argument("--metrics", nargs="+",
                        default=["sari", "fkgl", "token_f1", "bleu", "rouge_l", "bert_score",
                                 "sbert", "output_words"],
                        help="Per-sample metrics to compute (NLI statistical comparison is excluded)")
    parser.add_argument("--output", default=None,
                        help="Path to save results JSON")
    parser.add_argument("--skip-heavy", action="store_true",
                        help="Skip SBERT and BERTScore (faster, no GPU needed)")
    args = parser.parse_args()

    if args.skip_heavy:
        args.metrics = [m for m in args.metrics if m not in ("sbert", "bert_score")]

    if not args.pipeline_traces:
        raise ValueError("--pipeline-traces is required for agentic comparison")

    if args.device == "auto":
        runtime_device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    else:
        runtime_device = args.device

    if runtime_device == "cuda" and (torch is None or not torch.cuda.is_available()):
        warnings.warn("CUDA requested but unavailable; falling back to CPU")
        runtime_device = "cpu"

    # ── Load models for metrics that need them ──
    sbert_model = None

    if "sbert" in args.metrics and SentenceTransformer is not None:
        print(f"Loading SBERT model on {runtime_device}...")
        sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=runtime_device)

    # ── Load predictions ──
    print(f"\nLoading baseline predictions from {args.baseline_predictions}...")
    baseline_data = load_baseline_predictions(args.baseline_predictions)

    # Build approach dict: name -> (preds, refs, originals)
    approaches = {}
    selected_models = [m for m in args.baseline_models if m.upper() != "ALL"]
    if not selected_models:
        selected_models = [k for k in baseline_data.keys() if not k.startswith("_")]

    for name in selected_models:
        entries = baseline_data.get(name)
        if entries is None:
            warnings.warn(f"Requested baseline model not found: {name}")
            continue
        preds, refs, origs, keys, judge = extract_preds_refs_origs(entries)
        if preds:
            approaches[name] = (preds, refs, origs, keys, judge)
            print(f"  {name}: {len(preds)} samples")

    if args.pipeline_traces:
        print(f"Loading pipeline traces from {args.pipeline_traces}...")
        traces = load_pipeline_traces(args.pipeline_traces)
        preds, refs, origs, keys, judge = extract_from_traces(traces)
        approaches["Agentic (no tools)"] = (preds, refs, origs, keys, judge)
        print(f"  Agentic (no tools): {len(preds)} samples")

    if args.pipeline_tools_traces:
        print(f"Loading pipeline+tools traces from {args.pipeline_tools_traces}...")
        traces = load_pipeline_traces(args.pipeline_tools_traces)
        preds, refs, origs, keys, judge = extract_from_traces(traces)
        approaches["Agentic (with tools)"] = (preds, refs, origs, keys, judge)
        print(f"  Agentic (with tools): {len(preds)} samples")

    # ── Align samples across approaches ──
    print("\nAligning samples across approaches...")
    approaches = align_approaches(approaches)

    # ── Compute per-sample metrics ──
    print("\nComputing per-sample metrics...")
    all_metrics = {}
    for name, (preds, refs, origs, _keys, judge) in approaches.items():
        print(f"\n  [{name}]")
        all_metrics[name] = compute_per_sample_metrics(
            preds, refs, origs,
            sbert_model=sbert_model,
            bert_device=runtime_device,
            metric_names=args.metrics,
        )
        # Add pre-computed LLM-Judge scores if available
        if any(s is not None for s in judge):
            all_metrics[name]["llm_judge"] = [s if s is not None else 0.0 for s in judge]
            print("    Loaded pre-computed LLM-Judge scores")

    # ── Run pairwise tests ──
    print("\nRunning pairwise statistical tests...")
    approach_names = list(approaches.keys())
    all_results = {}

    for i in range(len(approach_names)):
        for j in range(i + 1, len(approach_names)):
            name_a = approach_names[i]
            name_b = approach_names[j]
            comparison = f"{name_a}  vs  {name_b}"
            print(f"  {comparison}")
            all_results[comparison] = run_pairwise_tests(
                all_metrics[name_a], all_metrics[name_b],
                name_a, name_b,
                n_bootstrap=args.n_bootstrap,
            )

    # ── Print results ──
    print_results(all_results, alpha=args.alpha, n_bootstrap=args.n_bootstrap,
                  m_total=args.m_total)

    # ── Save results ──
    if args.output:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        serializable = json.loads(json.dumps(all_results, default=convert))
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
