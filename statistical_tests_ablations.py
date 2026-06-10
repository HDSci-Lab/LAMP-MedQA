"""
Statistical significance tests: ablations vs. no-tools lightweight pipeline.

For each of the three ablations (no_extraction_validation, no_readability_validation,
both removed) computes per-sample metrics and runs:
  1. Paired bootstrap resampling (10,000 resamples, two-sided p-value, 95% CI)
  2. Wilcoxon signed-rank test (non-parametric paired test)

Multiple-comparison correction:
  Bonferroni with alpha=0.05 and m=90 tests (paper-wide), so the
  per-test threshold is alpha/m = 5.555...e-4. A '*' in the output flags
  comparisons that meet that threshold.

Blank predictions are retained and receive worst-case scores
(empty -> 0 for every metric, except FKGL which uses 0; LLM-Judge uses 1).

Usage:
  python statistical_tests_ablations.py \
      --baseline results/final_results/lightweight_pipeline_traces_20260317_230930.json \
      --ablations \
          results/ablation/lightweight_pipeline_abl_noreadval_traces_20260407_160323.json \
          results/ablation/lightweight_pipeline_abl_noaccval_traces_20260407_170441.json \
          results/ablation/lightweight_pipeline_abl_noaccval_noreadval_traces_20260407_173006.json \
      --output results/ablation/statistical_tests_ablations.json
"""

import argparse
import json
import os
import re
import warnings
from collections import Counter
from typing import Dict, List, Tuple

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
    warnings.warn("textstat not installed - FKGL per-sample scores will be 0")

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None
    warnings.warn("rouge_score not installed - ROUGE-L per-sample scores will be 0")

try:
    from bert_score import score as bert_score_fn
except ImportError:
    bert_score_fn = None
    warnings.warn("bert_score not installed - BERTScore per-sample scores will be 0")

try:
    from sacrebleu.metrics import BLEU
except ImportError:
    BLEU = None
    warnings.warn("sacrebleu not installed - BLEU per-sample scores will be 0")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    SentenceTransformer = None
    warnings.warn("sentence-transformers not installed - SBERT per-sample scores will be 0")


SEED = 42
np.random.seed(SEED)


# ----------------------------------------------------------------------------
# Per-sample metrics (blank prediction -> worst-case score)
# ----------------------------------------------------------------------------

def _ngrams(text: str, n: int):
    tokens = re.findall(r"\w+", text.lower())
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def per_sample_sari(preds, refs, originals) -> List[float]:
    N = 4
    scores = []
    for orig, pred, ref in zip(originals, preds, refs):
        if not pred:
            scores.append(0.0)
            continue
        sent = 0.0
        for n in range(1, N + 1):
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
    if textstat is None:
        return [0.0] * len(preds)
    out = []
    for p in preds:
        if not p:
            out.append(0.0)
        else:
            try:
                out.append(float(textstat.flesch_kincaid_grade(p)))
            except Exception:
                out.append(0.0)
    return out


def per_sample_token_f1(preds, refs) -> List[float]:
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
    return [0.0 if not pp else _f1(pp, rr) for pp, rr in zip(preds, refs)]


def per_sample_bleu(preds, refs) -> List[float]:
    if BLEU is None:
        return [0.0] * len(preds)
    bleu = BLEU(effective_order=True)
    out = []
    for p, r in zip(preds, refs):
        if not p:
            out.append(0.0)
        else:
            out.append(bleu.sentence_score(p, [r]).score / 100.0)
    return out


def per_sample_rouge_l(preds, refs) -> List[float]:
    if rouge_scorer is None:
        return [0.0] * len(preds)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return [0.0 if not p else scorer.score(r, p)['rougeL'].fmeasure
            for p, r in zip(preds, refs)]


def per_sample_bert_score(preds, refs, device: str = "cpu") -> List[float]:
    if bert_score_fn is None:
        return [0.0] * len(preds)
    safe_preds = [p if p else " " for p in preds]
    P, R, F1 = bert_score_fn(safe_preds, refs, lang='en',
                             model_type='distilbert-base-uncased',
                             device=device, verbose=False)
    out = F1.tolist()
    for i, p in enumerate(preds):
        if not p:
            out[i] = 0.0
    return out


def per_sample_sbert(preds, refs, sbert_model) -> List[float]:
    safe_preds = [p if p else " " for p in preds]
    embp = sbert_model.encode(safe_preds, convert_to_numpy=True)
    embr = sbert_model.encode(refs, convert_to_numpy=True)
    out = []
    for a, b, p in zip(embp, embr, preds):
        if not p:
            out.append(0.0)
        else:
            out.append(float(cosine_similarity([a], [b])[0, 0]))
    return out


def per_sample_output_words(preds) -> List[float]:
    return [float(len(re.findall(r"\w+", p))) for p in preds]


# ----------------------------------------------------------------------------
# Statistical tests
# ----------------------------------------------------------------------------

def paired_bootstrap(scores_a: np.ndarray, scores_b: np.ndarray,
                     n_bootstrap: int = 10000, seed: int = SEED) -> dict:
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    assert len(scores_b) == n
    observed_diff = float(np.mean(scores_a) - np.mean(scores_b))

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        diffs[i] = np.mean(scores_a[idx]) - np.mean(scores_b[idx])

    # Two-sided p-value
    if observed_diff >= 0:
        p_value = float(np.mean(diffs <= 0) * 2)
    else:
        p_value = float(np.mean(diffs >= 0) * 2)
    p_value = min(p_value, 1.0)

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "ci_95_lower": float(np.percentile(diffs, 2.5)),
        "ci_95_upper": float(np.percentile(diffs, 97.5)),
        "n_bootstrap": int(n_bootstrap),
    }


def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    diff = scores_a - scores_b
    nonzero = diff[diff != 0]
    if len(nonzero) < 10:
        return {"statistic": None, "p_value": 1.0,
                "note": "Too few non-zero differences"}
    stat, p_value = stats.wilcoxon(nonzero, alternative='two-sided')
    return {"statistic": float(stat), "p_value": float(p_value)}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_traces(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("test", "dev", "traces"):
            if key in data:
                data = data[key]
                break
        if isinstance(data, dict):
            for key in ("predictions", "traces"):
                if key in data:
                    data = data[key]
                    break
    if not isinstance(data, list):
        raise ValueError(f"Could not find a predictions list in {path}")
    return data


def _entry_key(entry: dict) -> str:
    nid = entry.get("note_id", "")
    if nid:
        return str(nid)
    q = entry.get("question") or entry.get("Question") or ""
    return q.strip()


def extract_fields(traces: List[dict]):
    preds, refs, origs, keys, judge = [], [], [], [], []
    for t in traces:
        # Pipeline traces use 'prediction'; some older formats use 'final_answer'.
        p = t.get("prediction")
        if p is None:
            p = t.get("final_answer", "")
        preds.append(p or "")
        refs.append(t.get("reference", "") or "")
        origs.append(t.get("original", "") or t.get("discharge_summary", "") or "")
        keys.append(_entry_key(t))
        judge.append(t.get("llm_judge_score"))
    return preds, refs, origs, keys, judge


def align(approaches: Dict[str, Tuple]) -> Dict[str, Tuple]:
    key_sets = {n: set(d[3]) for n, d in approaches.items()}
    common = set.intersection(*key_sets.values()) if key_sets else set()
    if not common:
        raise ValueError("No overlapping samples across approaches.")
    for name, ks in key_sets.items():
        d = len(ks) - len(common)
        if d > 0:
            warnings.warn(f"  {name}: dropping {d} samples not present in all approaches")

    first = next(iter(approaches))
    ordered = [k for k in approaches[first][3] if k in common]
    aligned = {}
    for name, (preds, refs, origs, keys, judge) in approaches.items():
        lookup = {k: i for i, k in enumerate(keys)}
        idxs = [lookup[k] for k in ordered]
        aligned[name] = (
            [preds[i] for i in idxs],
            [refs[i] for i in idxs],
            [origs[i] for i in idxs],
            ordered,
            [judge[i] for i in idxs],
        )
    print(f"  Aligned {len(ordered)} common samples across {len(approaches)} approaches")
    return aligned


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def compute_per_sample_metrics(preds, refs, origs, judge,
                                sbert_model, bert_device,
                                metric_names) -> Dict[str, List[float]]:
    out = {}
    if "sari" in metric_names:
        print("    SARI...")
        out["sari"] = per_sample_sari(preds, refs, origs)
    if "fkgl" in metric_names:
        print("    FKGL...")
        out["fkgl"] = per_sample_fkgl(preds)
    if "token_f1" in metric_names:
        print("    Token F1...")
        out["token_f1"] = per_sample_token_f1(preds, refs)
    if "bleu" in metric_names:
        print("    BLEU...")
        out["bleu"] = per_sample_bleu(preds, refs)
    if "rouge_l" in metric_names:
        print("    ROUGE-L...")
        out["rouge_l"] = per_sample_rouge_l(preds, refs)
    if "bert_score" in metric_names and bert_score_fn is not None:
        print("    BERTScore...")
        out["bert_score"] = per_sample_bert_score(preds, refs, device=bert_device)
    if "sbert" in metric_names and sbert_model is not None:
        print("    SBERT...")
        out["sbert"] = per_sample_sbert(preds, refs, sbert_model)
    if "output_words" in metric_names:
        print("    output_words...")
        out["output_words"] = per_sample_output_words(preds)
    if "llm_judge" in metric_names and any(s is not None for s in judge):
        # Worst-case is 1.0 (Likert-1) for missing entries.
        out["llm_judge"] = [float(s) if s is not None else 1.0 for s in judge]
        print("    Loaded pre-computed LLM-Judge scores")
    return out


def run_pairwise(metrics_a, metrics_b, n_bootstrap):
    results = {}
    shared = sorted(set(metrics_a) & set(metrics_b))
    for metric in shared:
        a = np.array(metrics_a[metric])
        b = np.array(metrics_b[metric])
        results[metric] = {
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "bootstrap": paired_bootstrap(a, b, n_bootstrap=n_bootstrap),
            "wilcoxon": wilcoxon_test(a, b),
        }
    return results


def short_name(path: str) -> str:
    base = os.path.basename(path)
    if "noaccval_noreadval" in base:
        return "no_extval + no_readval"
    if "noaccval" in base:
        return "no_extval"
    if "noreadval" in base:
        return "no_readval"
    return base


def print_results(all_results: dict, *, m_total: int, alpha: float,
                  n_bootstrap: int, threshold: float):
    print("\n" + "=" * 110)
    print("STATISTICAL SIGNIFICANCE: ablations vs no-tools lightweight pipeline")
    print(f"  Bonferroni: alpha={alpha}, m={m_total} -> per-test threshold={threshold:.3e}")
    print(f"  Paired bootstrap: {n_bootstrap} resamples")
    print("=" * 110)

    for comparison, metrics in all_results.items():
        print(f"\n--- {comparison} ---")
        print(f"  {'Metric':<14} {'Abl':>9} {'Base':>9} {'Diff':>9} "
              f"{'BootP':>11} {'WilcP':>11} {'BonfBootP':>12} {'Sig':>4}")
        for metric in sorted(metrics):
            r = metrics[metric]
            boot_p = r["bootstrap"]["p_value"]
            wilc_p = r["wilcoxon"]["p_value"]
            bonf_p = min(boot_p * m_total, 1.0)
            sig = "*" if bonf_p < alpha else ""
            print(f"  {metric:<14} {r['mean_a']:>9.4f} {r['mean_b']:>9.4f} "
                  f"{r['bootstrap']['observed_diff']:>+9.4f} "
                  f"{boot_p:>11.4f} {wilc_p:>11.4f} {bonf_p:>12.4f} {sig:>4}")
            ci = r["bootstrap"]
            print(f"  {'':<14} {'95% CI':>30}: [{ci['ci_95_lower']:+.4f}, "
                  f"{ci['ci_95_upper']:+.4f}]")
    print("\n  * = significant after Bonferroni correction "
          f"(alpha={alpha}, m={m_total} tests; uses paired-bootstrap p-value).\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True,
                        help="Path to no-tools lightweight pipeline traces JSON")
    parser.add_argument("--ablations", nargs="+", required=True,
                        help="Paths to ablation traces JSON files")
    parser.add_argument("--extra-predictions", default=None,
                        help="evaluate_200-format predictions JSON (e.g. single-model "
                             "ablations from evaluate_200.py). Models named via "
                             "--extra-predictions-models are loaded as additional "
                             "approaches to compare against the baseline.")
    parser.add_argument("--extra-predictions-models", nargs="+", default=[],
                        help="Model keys inside the --extra-predictions JSON to include "
                             "(e.g. 'Qwen2.5-7B (zero-shot)' 'Phi-3.5-mini (zero-shot)')")
    parser.add_argument("--baseline-name", default="lightweight (no tools)")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--m-total", type=int, default=90,
                        help="Total number of tests across the paper for Bonferroni (default 90)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--metrics", nargs="+",
                        default=["sari", "fkgl", "token_f1", "bleu", "rouge_l",
                                 "bert_score", "sbert", "output_words", "llm_judge"])
    parser.add_argument("--skip-heavy", action="store_true",
                        help="Skip SBERT and BERTScore")
    parser.add_argument("--output", default=None,
                        help="Path to save full results JSON")
    args = parser.parse_args()

    if args.skip_heavy:
        args.metrics = [m for m in args.metrics if m not in ("sbert", "bert_score")]

    if args.device == "auto":
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    else:
        device = args.device
    if device == "cuda" and (torch is None or not torch.cuda.is_available()):
        warnings.warn("CUDA requested but unavailable; falling back to CPU")
        device = "cpu"

    sbert_model = None
    if "sbert" in args.metrics and SentenceTransformer is not None:
        print(f"Loading SBERT (sentence-transformers/all-mpnet-base-v2) on {device}...")
        sbert_model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2", device=device)

    print(f"\nLoading baseline traces: {args.baseline}")
    base_traces = load_traces(args.baseline)
    base = extract_fields(base_traces)
    print(f"  {args.baseline_name}: {len(base[0])} samples")

    approaches = {args.baseline_name: base}
    for path in args.ablations:
        print(f"Loading ablation traces: {path}")
        tr = load_traces(path)
        approaches[short_name(path)] = extract_fields(tr)
        print(f"  {short_name(path)}: {len(approaches[short_name(path)][0])} samples")

    if args.extra_predictions:
        print(f"Loading extra predictions JSON: {args.extra_predictions}")
        with open(args.extra_predictions, "r", encoding="utf-8") as f:
            extra = json.load(f)
        if not args.extra_predictions_models:
            raise SystemExit("--extra-predictions requires --extra-predictions-models")
        for name in args.extra_predictions_models:
            entries = extra.get(name)
            if entries is None:
                raise SystemExit(
                    f"Model '{name}' not found in {args.extra_predictions}; "
                    f"available: {list(extra.keys())}")
            approaches[name] = extract_fields(entries)
            print(f"  {name}: {len(approaches[name][0])} samples")

    print("\nAligning samples across approaches...")
    approaches = align(approaches)

    # Sanity-check LLM-judge availability on ablations
    for name, (_, _, _, _, judge) in approaches.items():
        all_one = all((j is not None and float(j) == 1.0) for j in judge)
        if all_one and name != args.baseline_name:
            warnings.warn(
                f"  '{name}': all llm_judge_score values are 1.0 -- this is the "
                f"placeholder. Run llm_judge_ablations.py before relying on the "
                f"llm_judge comparison.")

    print("\nComputing per-sample metrics...")
    all_metrics = {}
    for name, (preds, refs, origs, _, judge) in approaches.items():
        print(f"\n  [{name}]")
        all_metrics[name] = compute_per_sample_metrics(
            preds, refs, origs, judge,
            sbert_model=sbert_model, bert_device=device,
            metric_names=args.metrics,
        )

    print("\nRunning pairwise tests (each ablation vs baseline)...")
    all_results = {}
    for name in approaches:
        if name == args.baseline_name:
            continue
        comparison = f"{name}  vs  {args.baseline_name}"
        print(f"  {comparison}")
        all_results[comparison] = run_pairwise(
            all_metrics[name], all_metrics[args.baseline_name],
            n_bootstrap=args.n_bootstrap,
        )

    threshold = args.alpha / args.m_total
    print_results(all_results, m_total=args.m_total, alpha=args.alpha,
                  n_bootstrap=args.n_bootstrap, threshold=threshold)

    if args.output:
        out = {
            "config": {
                "baseline": args.baseline,
                "baseline_name": args.baseline_name,
                "ablations": args.ablations,
                "alpha": args.alpha,
                "m_total": args.m_total,
                "per_test_threshold": threshold,
                "n_bootstrap": args.n_bootstrap,
                "metrics": args.metrics,
            },
            "comparisons": {},
        }
        for comparison, metrics in all_results.items():
            out["comparisons"][comparison] = {}
            for metric, r in metrics.items():
                bonf_p = min(r["bootstrap"]["p_value"] * args.m_total, 1.0)
                out["comparisons"][comparison][metric] = {
                    **r,
                    "bonferroni_adjusted_p": bonf_p,
                    "significant_bonf": bonf_p < args.alpha,
                }

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

        def _convert(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.bool_):
                return bool(o)
            raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(json.loads(json.dumps(out, default=_convert)),
                      f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
