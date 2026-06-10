#!/usr/bin/env python3
"""Run GPT-4 LLM-as-judge on ablation traces.

Targets the three ablation traces files and writes per-sample
`llm_judge_score` back into each one (in-place), plus updates the
corpus-level `metrics.llm_judge_score`.

Resumable: if a sample already has a non-default judge score, it is
skipped, so you can rerun safely after a crash / rate-limit.

Uses the exact same prompt as evaluate_lightweight_pipeline.py and
rerun_llm_judge.py for consistency with the no-tools baseline scores.

Usage:
    set OPENAI_API_KEY=...
    python llm_judge_ablations.py                 # all three ablation files
    python llm_judge_ablations.py path1.json ...  # explicit list
    python llm_judge_ablations.py --force ...     # rescore everything
"""

import argparse
import json
import os
import sys
import time
from typing import List, Tuple

from openai import OpenAI


SEED = 42
MODEL = "gpt-4"

DEFAULT_FILES = [
    "results/ablation/lightweight_pipeline_abl_noreadval_traces_20260407_160323.json",
    "results/ablation/lightweight_pipeline_abl_noaccval_traces_20260407_170441.json",
    "results/ablation/lightweight_pipeline_abl_noaccval_noreadval_traces_20260407_173006.json",
]

# A score of exactly 1.0 in the trace files is the placeholder value
# written by the pipeline before any real judge call. We treat it as
# "not yet scored" UNLESS the prediction or reference is empty
# (in which case 1.0 is the legitimate worst-case score).
PLACEHOLDER = 1.0


def llm_judge_score(pred: str, ref: str, client: OpenAI, idx: int = 0,
                    max_retries: int = 5) -> float:
    """Score one (pred, ref) pair on the 1-5 Likert scale."""
    if not pred or not ref:
        return 1.0

    prompt = f"""You are an expert medical QA judge. Rate the semantic similarity of the predicted answer and the reference answer.

Reference Answer (gold standard):
{ref}

Predicted Answer (to evaluate):
{pred}

Rate the semantic similarity on this scale:
1 - Not semantically similar at all
2 - Not very semantically similar
3 - Somewhat semantically similar
4 - Quite semantically similar
5 - Very semantically similar

Return ONLY a single JSON object with no markdown formatting: {{"similarity": <score>}}
"""

    backoff = 2.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=150,
                temperature=0.0,
                seed=SEED,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)
            score = result.get("similarity", 1)
            return float(max(1, min(5, score)))
        except json.JSONDecodeError as e:
            print(f"    [{idx}] JSON parse error (attempt {attempt+1}): {e}; raw={raw[:120]!r}")
        except Exception as e:
            print(f"    [{idx}] API error (attempt {attempt+1}): {e}")
        time.sleep(backoff)
        backoff = min(backoff * 2, 30.0)

    print(f"    [{idx}] giving up after {max_retries} attempts; recording 1.0")
    return 1.0


def needs_scoring(entry: dict, force: bool) -> bool:
    if force:
        return True
    s = entry.get("llm_judge_score")
    pred = entry.get("prediction", "") or ""
    ref = entry.get("reference", "") or ""
    if s is None:
        return True
    # Placeholder 1.0 is only legitimate if pred or ref is empty.
    if float(s) == PLACEHOLDER and pred and ref:
        return True
    return False


def score_file(path: str, client: OpenAI, force: bool, save_every: int) -> Tuple[List[float], float]:
    print(f"\n{'='*72}")
    print(f"Processing: {path}")
    print(f"{'='*72}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Pipeline traces format: {"split": {"predictions": [...], "metrics": {...}}}
    # All three ablation files are this format with split == "test".
    if "test" not in data or "predictions" not in data["test"]:
        raise ValueError(f"Unexpected file structure in {path}; expected data['test']['predictions']")

    split = data["test"]
    entries = split["predictions"]
    n = len(entries)

    todo_indices = [i for i, e in enumerate(entries) if needs_scoring(e, force)]
    already = n - len(todo_indices)
    print(f"  {n} predictions; {already} already scored, {len(todo_indices)} to score")

    scored_count = 0
    for i in todo_indices:
        e = entries[i]
        pred = e.get("prediction", "") or ""
        ref = e.get("reference", "") or ""
        s = llm_judge_score(pred, ref, client, idx=i)
        e["llm_judge_score"] = s
        scored_count += 1

        if scored_count % 10 == 0:
            running_scores = [float(x.get("llm_judge_score", 1.0)) for x in entries]
            avg = sum(running_scores) / len(running_scores)
            print(f"    progress: {scored_count}/{len(todo_indices)} new "
                  f"(corpus avg so far: {avg:.4f})")

        if save_every and scored_count % save_every == 0:
            split.setdefault("metrics", {})["llm_judge_score"] = (
                sum(float(x.get("llm_judge_score", 1.0)) for x in entries) / n
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    all_scores = [float(e.get("llm_judge_score", 1.0)) for e in entries]
    avg_score = sum(all_scores) / n if n else 0.0
    split.setdefault("metrics", {})["llm_judge_score"] = avg_score

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Done. Mean LLM-Judge: {avg_score:.4f}  (n={n}, newly scored={scored_count})")
    return all_scores, avg_score


def main():
    parser = argparse.ArgumentParser(description="Run GPT-4 LLM-judge on ablation traces.")
    parser.add_argument("files", nargs="*",
                        help="Ablation traces JSON files (default: the three abl_* files)")
    parser.add_argument("--force", action="store_true",
                        help="Rescore even entries that already have a non-placeholder score")
    parser.add_argument("--save-every", type=int, default=20,
                        help="Save partial progress every N newly scored items (0 = only at end)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    files = args.files or DEFAULT_FILES
    files = [os.path.abspath(p) for p in files]

    missing = [p for p in files if not os.path.isfile(p)]
    if missing:
        print("ERROR: file(s) not found:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    print("OpenAI client initialised. Using model:", MODEL)

    summary = {}
    for path in files:
        _, avg = score_file(path, client, force=args.force, save_every=args.save_every)
        summary[os.path.basename(path)] = avg

    print("\n" + "=" * 72)
    print("FINAL LLM-JUDGE SCORES")
    print("=" * 72)
    for name, avg in summary.items():
        print(f"  {avg:.4f}   {name}")


if __name__ == "__main__":
    main()
