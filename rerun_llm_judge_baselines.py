#!/usr/bin/env python3
"""Re-run GPT-4 LLM-as-judge on evaluate_200-format prediction files.

The evaluate_200 prediction file has the structure:

    {
      "GPT-5":                 [ {prediction, reference, ...}, ...],
      "Qwen2.5-32B (zero-shot)": [...],
      "Qwen2.5-32B (one-shot)":  [...]
    }

Per-sample llm_judge_score is added to each entry, and the file is
overwritten in place. Resumable: entries that already have a non-default
llm_judge_score are skipped.

Uses the same prompt / model / seed / temperature as the rest of the
paper's LLM-Judge runs (gpt-4, temperature 0, seed 42, max_tokens 150).

Usage
-----
    export OPENAI_API_KEY=sk-proj-...
    python rerun_llm_judge_baselines.py results/final_results/evaluate_200_predictions_20260316_141241.json
    # or restrict to one model:
    python rerun_llm_judge_baselines.py FILE.json --models "GPT-5"
    # or rescore everything:
    python rerun_llm_judge_baselines.py FILE.json --force
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
    raw = ""
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
    # 1.0 is only legitimate if pred or ref is empty.
    if float(s) == PLACEHOLDER and pred and ref:
        return True
    return False


def score_model(model_name: str, entries: List[dict], client: OpenAI,
                force: bool, save_callback, save_every: int) -> Tuple[float, int]:
    print(f"\n[{model_name}] {len(entries)} entries")
    todo = [i for i, e in enumerate(entries) if needs_scoring(e, force)]
    already = len(entries) - len(todo)
    print(f"  {already} already scored, {len(todo)} to score")

    new_count = 0
    for i in todo:
        e = entries[i]
        pred = e.get("prediction", "") or ""
        ref = e.get("reference", "") or ""
        e["llm_judge_score"] = llm_judge_score(pred, ref, client, idx=i)
        new_count += 1

        if new_count % 10 == 0:
            running = [float(x.get("llm_judge_score", 1.0)) for x in entries]
            print(f"    [{model_name}] {new_count}/{len(todo)} new "
                  f"(corpus avg so far: {sum(running)/len(running):.4f})")

        if save_every and new_count % save_every == 0:
            save_callback()

    save_callback()
    avg = (sum(float(x.get("llm_judge_score", 1.0)) for x in entries) / len(entries)) \
          if entries else 0.0
    print(f"  [{model_name}] done. Mean LLM-Judge: {avg:.4f}  (n={len(entries)}, "
          f"newly scored={new_count})")
    return avg, new_count


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="evaluate_200 predictions JSON file (in place)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Limit to these top-level keys (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Rescore even entries that already have a non-placeholder score")
    parser.add_argument("--save-every", type=int, default=20,
                        help="Save partial progress every N newly scored items (0 = only at end)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.path):
        print(f"ERROR: {args.path} not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        print(f"ERROR: top-level JSON must be a dict mapping model names to entry lists.",
              file=sys.stderr)
        sys.exit(1)

    def save():
        with open(args.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    client = OpenAI(api_key=api_key)
    print(f"OpenAI client initialised. Using model: {MODEL}")
    print(f"File: {args.path}")

    targets = args.models or [k for k in data.keys() if isinstance(data[k], list)]
    summary = {}
    for name in targets:
        entries = data.get(name)
        if entries is None:
            print(f"WARN: model '{name}' not in file; skipping.")
            continue
        if not isinstance(entries, list):
            print(f"WARN: '{name}' is not a list (got {type(entries).__name__}); skipping.")
            continue
        if not entries:
            print(f"WARN: '{name}' is empty; skipping.")
            continue
        avg, new_count = score_model(name, entries, client,
                                     force=args.force, save_callback=save,
                                     save_every=args.save_every)
        summary[name] = avg

    print("\n" + "=" * 60)
    print("FINAL LLM-JUDGE SCORES")
    print("=" * 60)
    for name, avg in summary.items():
        print(f"  {avg:.4f}   {name}")


if __name__ == "__main__":
    main()
