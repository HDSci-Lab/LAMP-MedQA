"""
Recalculate metrics from saved trace JSON files (fixes the \\w+ regex bug).
"""

import json
import re
import sys
import csv
from collections import Counter

import textstat
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score_fn
from transformers import pipeline as hf_pipeline


# ── Per-sample metrics ──────────────────────────────────────────────────────

def _ngrams(sentence, n):
    tokens = re.findall(r"\w+", sentence.lower())
    return [tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]


def sample_sari(pred, ref, orig):
    N = 4
    sent_score = 0.0
    for n in range(1, N + 1):
        orig_ng = set(_ngrams(orig, n))
        pred_ng = set(_ngrams(pred, n))
        ref_ng = set(_ngrams(ref, n))

        keep_pred = pred_ng & orig_ng
        keep_ref = ref_ng & orig_ng
        keep_score = (
            len(keep_pred & keep_ref) / len(keep_pred) if keep_pred
            else (1.0 if not keep_ref else 0.0)
        )

        add_pred = pred_ng - orig_ng
        add_ref = ref_ng - orig_ng
        add_score = (
            len(add_pred & add_ref) / len(add_pred) if add_pred else 1.0
        )

        del_pred = orig_ng - pred_ng
        del_ref = orig_ng - ref_ng
        del_score = (
            len(del_pred & del_ref) / len(del_ref) if del_ref else 1.0
        )

        sent_score += (keep_score + add_score + del_score) / 3
    return (sent_score / N) * 100.0


def sample_token_f1(pred, ref):
    ta = re.findall(r"\w+", pred.lower())
    tb = re.findall(r"\w+", ref.lower())
    common = Counter(ta) & Counter(tb)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(ta) if ta else 0.0
    r = num_same / len(tb) if tb else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ── Corpus metrics ──────────────────────────────────────────────────────────

print("Loading evaluation models ...")
sbert_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

nli_pipe = None
def get_nli():
    global nli_pipe
    if nli_pipe is None:
        print("  Loading NLI model ...")
        nli_pipe = hf_pipeline("text-classification", model="microsoft/deberta-large-mnli", device=-1)
    return nli_pipe


def compute_all(preds, refs, originals):
    print("  SARI ...")
    sari = sum(sample_sari(p, r, o) for p, r, o in zip(preds, refs, originals)) / len(preds)

    print("  FKGL ...")
    fkgl = sum(textstat.flesch_kincaid_grade(p) for p in preds) / len(preds)

    print("  NLI ...")
    nli = get_nli()
    nli_scores = []
    for ref, pred in zip(refs, preds):
        result = nli(f"{ref} [SEP] {pred}", truncation=True, max_length=512)
        label = result[0]["label"].upper()
        nli_scores.append(1.0 if label == "ENTAILMENT" else 0.5 if label == "NEUTRAL" else 0.0)
    nli_avg = sum(nli_scores) / len(nli_scores)

    print("  SBERT ...")
    emb_p = sbert_model.encode(preds, convert_to_numpy=True)
    emb_r = sbert_model.encode(refs, convert_to_numpy=True)
    sbert_cos = sum(float(cosine_similarity([a], [b])[0, 0]) for a, b in zip(emb_p, emb_r)) / len(preds)

    print("  Token F1 ...")
    tf1 = sum(sample_token_f1(p, r) for p, r in zip(preds, refs)) / len(preds)

    print("  BLEU ...")
    bleu = BLEU().corpus_score(preds, [[r for r in refs]]).score / 100.0

    print("  ROUGE-L ...")
    rscorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = sum(rscorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)) / len(preds)

    print("  BERTScore ...")
    try:
        P, R, F1 = bert_score_fn(preds, refs, lang="en", model_type="distilbert-base-uncased", device="cpu")
        bert = F1.mean().item()
    except Exception as e:
        print(f"  BERTScore error: {e}")
        bert = 0.0

    word_counts = [len(re.findall(r"\w+", p)) for p in preds]
    avg_words = sum(word_counts) / len(word_counts)

    return {
        "sari_score": sari,
        "fkgl_score": fkgl,
        "nli_score": nli_avg,
        "sbert_cos_mean": sbert_cos,
        "token_f1": tf1,
        "bleu_score": bleu,
        "rouge_l_score": rouge,
        "bert_score": bert,
        "avg_output_words": avg_words,
    }


# ── Main ────────────────────────────────────────────────────────────────────

trace_files = [
    ("no_tools_dev",  "new_agent_results/lightweight_pipeline_traces_20260311_122105.json"),
    ("no_tools_test", "new_agent_results/lightweight_pipeline_traces_20260311_130332.json"),
    ("tools_dev",     "new_agent_results/lightweight_pipeline_tools_traces_20260311_143124.json"),
    ("tools_test",    "new_agent_results/lightweight_pipeline_tools_traces_20260311_151315.json"),
]

all_results = {}

for label, path in trace_files:
    print(f"\n{'='*60}")
    print(f"Processing: {label} ({path})")
    print(f"{'='*60}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for split_name, split_data in data.items():
        key = f"{label}_{split_name}" if len(data) > 1 else label
        preds_list = split_data["predictions"]

        preds = [p["prediction"] for p in preds_list]
        refs = [p["reference"] for p in preds_list]
        originals = [p["original"] for p in preds_list]

        # Skip empty predictions
        valid = [(p, r, o) for p, r, o in zip(preds, refs, originals) if p.strip()]
        if not valid:
            print(f"  WARNING: No valid predictions for {key}")
            continue

        preds_v, refs_v, origs_v = zip(*valid)
        preds_v, refs_v, origs_v = list(preds_v), list(refs_v), list(origs_v)

        print(f"  {len(preds_v)} valid predictions out of {len(preds)}")

        metrics = compute_all(preds_v, refs_v, origs_v)
        metrics["n_samples"] = len(preds)
        metrics["n_valid"] = len(preds_v)

        all_results[key] = metrics

        print(f"\n  Results for {key}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k:<25} {v:.4f}")
            else:
                print(f"    {k:<25} {v}")

# ── Save CSV ────────────────────────────────────────────────────────────────
csv_path = "new_agent_results/lightweight_pipeline_recalculated_metrics.csv"
if all_results:
    metric_names = [k for k in next(iter(all_results.values())).keys()]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant"] + metric_names)
        for key, metrics in all_results.items():
            row = [key] + [f"{metrics[m]:.4f}" if isinstance(metrics[m], float) else str(metrics[m]) for m in metric_names]
            writer.writerow(row)
    print(f"\nRecalculated metrics saved to {csv_path}")

# ── Print comparison table ──────────────────────────────────────────────────
print(f"\n{'='*80}")
print("RECALCULATED METRICS COMPARISON")
print(f"{'='*80}")
print(f"{'Metric':<25}", end="")
for key in all_results:
    print(f"  {key:>15}", end="")
print()
print("-" * (25 + 17 * len(all_results)))
for m in ["sari_score", "fkgl_score", "nli_score", "sbert_cos_mean", "token_f1",
          "bleu_score", "rouge_l_score", "bert_score", "avg_output_words"]:
    print(f"{m:<25}", end="")
    for key in all_results:
        print(f"  {all_results[key][m]:>15.4f}", end="")
    print()
