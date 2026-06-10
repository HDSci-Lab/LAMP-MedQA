#!/usr/bin/env python3
"""Re-run LLM-Judge scoring on existing pipeline traces.

Usage:
    python rerun_llm_judge.py <traces_file> [<traces_file2> ...]

Loads predictions from each traces JSON, calls GPT-4 LLM-as-judge,
and overwrites the file with updated llm_judge_score values.
"""

import json, os, sys
from openai import OpenAI

SEED = 42


def llm_judge_score(pred, ref, client, idx=0):
    """Score a single (pred, ref) pair using GPT-4."""
    if not pred or not ref:
        return 1.0
    try:
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
        response = client.chat.completions.create(
            model="gpt-4",
            max_tokens=150,
            temperature=0.0,
            seed=SEED,
            messages=[{"role": "user", "content": prompt}],
        )
        result = json.loads(response.choices[0].message.content.strip())
        score = result.get("similarity", 1)
        return float(max(1, min(5, score)))
    except Exception as e:
        print(f"    LLM judge error on example {idx}: {e}")
        return 1.0


def _score_entries(entries, client, label=""):
    """Score a list of prediction entries and return per-sample scores."""
    scores = []
    for i, entry in enumerate(entries):
        pred = entry.get("prediction", "")
        ref = entry.get("reference", "")
        s = llm_judge_score(pred, ref, client, idx=i)
        entry["llm_judge_score"] = s
        scores.append(s)

        if (i + 1) % 10 == 0:
            avg = sum(scores) / len(scores)
            print(f"    {label}Progress: {i+1}/{len(entries)}  (avg so far: {avg:.3f})")

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return scores, avg_score


def rerun_judge(traces_path):
    print(f"\n{'='*60}")
    print(f"Processing: {traces_path}")
    print(f"{'='*60}")

    with open(traces_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = OpenAI()

    # Detect format: evaluate_all.py produces {"model_name": [list of entries]}
    # pipeline traces produce {"split_name": {"predictions": [...], "metrics": {...}}}
    first_val = next(iter(data.values())) if data else None

    if isinstance(first_val, list):
        # evaluate_all.py format: {"ModelName": [{prediction, reference, ...}, ...]}
        for model_name, entries in data.items():
            if not entries:
                print(f"  [{model_name}] No predictions found, skipping.")
                continue
            print(f"  [{model_name}] Scoring {len(entries)} samples ...")
            scores, avg_score = _score_entries(entries, client, label=f"[{model_name}] ")
            print(f"  [{model_name}] Done. Mean LLM-Judge: {avg_score:.4f}")
    else:
        # Pipeline traces format: {"split": {"predictions": [...], "metrics": {...}}}
        for split_name, split_data in data.items():
            preds_list = split_data.get("predictions", [])
            if not preds_list:
                print(f"  [{split_name}] No predictions found, skipping.")
                continue

            print(f"  [{split_name}] Scoring {len(preds_list)} samples ...")
            scores, avg_score = _score_entries(preds_list, client, label=f"[{split_name}] ")
            split_data["metrics"]["llm_judge_score"] = avg_score
            print(f"  [{split_name}] Done. Mean LLM-Judge: {avg_score:.4f}")

    with open(traces_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Updated: {traces_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python rerun_llm_judge.py <traces_file> [<traces_file2> ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        if not os.path.isfile(path):
            print(f"File not found: {path}")
            continue
        rerun_judge(path)

    print("\nDone. You can now re-run statistical_tests.py.")


if __name__ == "__main__":
    main()
