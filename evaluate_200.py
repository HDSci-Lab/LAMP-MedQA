"""
Unified Evaluation: 200-Report Test Set
========================================
Compares four approaches on the same 200 MeDiSumQA test reports:
  1. GPT-5 (zero-shot, OpenAI API)
  2. Qwen2.5-32B-Instruct (zero-shot)
  3. Qwen2.5-32B-Instruct (one-shot)
  4. Agentic pipeline (Qwen3.5-9B + Phi-4-mini, multi-step)

All approaches use identical data split:
  - Excludes one-shot exemplar (note_id=14171200-DS-21)
  - Seed=42 shuffle, skip first n_dev, take 200 test samples

Metrics: SARI, FKGL, NLI, SBERT, Token F1, BLEU, ROUGE-L, BERTScore, LLM-Judge
"""

import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import csv
import random
import argparse
import time
from datetime import datetime
from collections import Counter
from typing import List, Dict, Optional

import torch
import nltk
import textstat
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score_fn
from openai import OpenAI

# ============================================================================
# CONSTANTS
# ============================================================================

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

HF_TOKEN = os.environ.get("HF_TOKEN", None)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

MAX_NEW_TOKENS = 2000
ONE_SHOT_EXEMPLAR_NOTE_ID = "14171200-DS-21"

SYSTEM_PROMPT = (
    "You are a medical text summariser/simplifier. A patient asks the following question about their discharge summary. "
    "Please think step by step, first review what the key information that patient is asking, and then extract the "
    "relevant information from the discharge summary and produce a short, clear, patient-facing answer in plain English "
    "(1-2 sentences). Do NOT add medical jargon."
)

ZERO_SHOT_TEMPLATE = """Question:
{q}

Discharge summary:
{ds_excerpt}

Answer:
"""

# One-shot template: includes the full exemplar as in-context example
# The exemplar text is loaded from the dataset at runtime
ONE_SHOT_TEMPLATE_PREFIX = """Please use the following example as a guide:

{exemplar}

Question:
{q}

Discharge summary:
{ds_excerpt}

Answer:
"""

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_enable_fp32_cpu_offload=True,
)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_set(data_path: str, n_dev: int = 0, n_test: int = 200):
    """
    Load MeDiSumQA, exclude one-shot exemplar, return reproducible test split.
    Also returns the exemplar record for building one-shot prompts.
    """
    all_data = []
    exemplar = None
    with open(data_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                if record.get("note_id") == ONE_SHOT_EXEMPLAR_NOTE_ID:
                    exemplar = record
                    continue
                all_data.append(record)

    print(f"Loaded {len(all_data)} records (excluded exemplar {ONE_SHOT_EXEMPLAR_NOTE_ID})")

    random.seed(SEED)
    shuffled = random.sample(all_data, len(all_data))
    test_data = shuffled[n_dev: n_dev + n_test]

    print(f"Test set: {len(test_data)} samples (n_dev={n_dev}, n_test={n_test})")
    return test_data, exemplar


def build_one_shot_exemplar_text(exemplar: dict) -> str:
    """Build the one-shot exemplar string from the exemplar record."""
    # Match the format used in evaluate_all_one_shot.py
    return (
        f'"Answer": "{exemplar["Answer"]}", '
        f'"note_id": "{exemplar["note_id"]}", '
        f'"Question": "{exemplar["Question"]}", '
        f'"discharge_summary": "{exemplar["discharge_summary"]}"'
    )


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def truncate_discharge_summary(ds: str, max_chars: int = 42000) -> str:
    return ds[:max_chars] + "..." if len(ds) > max_chars else ds


def format_prompt(tokenizer, user_text: str, system_text: str = None) -> str:
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            combined = f"{system_text}\n\n{user_text}" if system_text else user_text
            messages = [{"role": "user", "content": combined}]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    return (system_text + "\n\n" + user_text) if system_text else user_text


def extract_gpt5_text(response) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()
    texts = []
    for item in response.output or []:
        if hasattr(item, "content") and item.content:
            for block in item.content:
                if hasattr(block, "text") and block.text:
                    texts.append(block.text)
        if hasattr(item, "text") and item.text:
            texts.append(item.text)
    return "\n".join(texts).strip()


def call_openai(user_prompt: str, model_id: str, client: OpenAI) -> str:
    if model_id.startswith("gpt-5"):
        response = client.responses.create(
            model=model_id,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=MAX_NEW_TOKENS,
        )
        return extract_gpt5_text(response)
    else:
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            seed=SEED,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()


def call_huggingface(user_prompt: str, tokenizer, model, system_prompt: str = None) -> str:
    formatted_prompt = format_prompt(tokenizer, user_prompt, system_prompt)
    inputs = tokenizer(
        formatted_prompt, return_tensors="pt", truncation=True, max_length=42000
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
    )
    return response.strip()


def load_hf_model(model_id: str, quantize: bool = True):
    """Load a HuggingFace model.

    quantize=True  -> 8-bit + CPU offload (needed for the 32B model)
    quantize=False -> plain fp16, single device (used for the 7B/3.8B
                       single-model ablations; matches the loader in
                       evaluate_lightweight_pipeline.py for those models).
    """
    print(f"  Loading {model_id} (quantize={quantize}) ...")
    token = HF_TOKEN if "Qwen" in model_id else None

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=token, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize:
        n_gpus = torch.cuda.device_count()
        mem_map = {i: "42GiB" for i in range(n_gpus)}
        mem_map["cpu"] = "80GB"
        load_kwargs = dict(
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=mem_map,
            token=token,
            trust_remote_code=True,
        )
    else:
        load_kwargs = dict(
            device_map="auto",
            torch_dtype=torch.float16,
            token=token,
            trust_remote_code=True,
        )
        # Phi models require eager attention to avoid SDPA bugs in transformers.
        if "Phi-3" in model_id or "phi-3" in model_id or "Phi-4" in model_id or "phi-4" in model_id:
            load_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    print(f"  {model_id} loaded.")
    return tokenizer, model


# ============================================================================
# METRIC FUNCTIONS
# ============================================================================

def _ngrams(sentence: str, n: int) -> List[tuple]:
    tokens = re.findall(r"\w+", sentence.lower())
    return [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))]


def sari_score(preds, refs, originals):
    if not preds:
        return 0.0
    N = 4
    total = 0.0
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
        total += sent / N
    return (total / len(preds)) * 100.0


def fkgl_score(preds):
    return sum(textstat.flesch_kincaid_grade(p) for p in preds) / len(preds) if preds else 0.0


def nli_score(preds, refs, nli_pipe):
    scores = []
    for ref, pred in zip(refs, preds):
        result = nli_pipe(f"{ref} [SEP] {pred}", truncation=True, max_length=512)
        label = result[0]['label'].upper()
        scores.append(1.0 if label == 'ENTAILMENT' else 0.5 if label == 'NEUTRAL' else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def sbert_cosine(preds, refs, sbert):
    embp = sbert.encode(preds, convert_to_numpy=True)
    embr = sbert.encode(refs, convert_to_numpy=True)
    cos = [float(cosine_similarity([a], [b])[0, 0]) for a, b in zip(embp, embr)]
    return sum(cos) / len(cos) if cos else 0.0


def token_f1(preds, refs):
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
    scores = [_f1(p, r) for p, r in zip(preds, refs)]
    return sum(scores) / len(scores) if scores else 0.0


def bleu_score(preds, refs):
    bleu = BLEU()
    return bleu.corpus_score(preds, [[r for r in refs]]).score / 100.0


def rouge_l_score(preds, refs):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(preds, refs)]
    return sum(scores) / len(scores) if scores else 0.0


def bert_score_metric(preds, refs):
    try:
        P, R, F1 = bert_score_fn(preds, refs, lang='en', model_type='distilbert-base-uncased', device='cpu')
        return F1.mean().item()
    except Exception as e:
        print(f"BERTScore error: {e}")
        return 0.0


def llm_judge(preds, refs, client):
    scores = []
    for i, (pred, ref) in enumerate(zip(preds, refs)):
        if not pred or not ref:
            scores.append(1.0)
            continue
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
            scores.append(float(max(1, min(5, result.get("similarity", 1)))))
        except Exception as e:
            print(f"    LLM judge error on example {i}: {e}")
            scores.append(1.0)

        if (i + 1) % 10 == 0:
            print(f"    LLM judge progress: {i+1}/{len(preds)} (avg: {sum(scores)/len(scores):.4f})")
    return scores


def compute_all_metrics(preds, refs, originals, nli_pipe, sbert, openai_client=None):
    print("  Computing SARI...")
    metrics = {"sari_score": sari_score(preds, refs, originals)}
    print("  Computing FKGL...")
    metrics["fkgl_score"] = fkgl_score(preds)
    print("  Computing NLI...")
    metrics["nli_score"] = nli_score(preds, refs, nli_pipe)
    print("  Computing SBERT...")
    metrics["sbert_cos_mean"] = sbert_cosine(preds, refs, sbert)
    print("  Computing Token F1...")
    metrics["token_f1"] = token_f1(preds, refs)
    print("  Computing BLEU...")
    metrics["bleu_score"] = bleu_score(preds, refs)
    print("  Computing ROUGE-L...")
    metrics["rouge_l_score"] = rouge_l_score(preds, refs)
    print("  Computing BERTScore...")
    metrics["bert_score"] = bert_score_metric(preds, refs)

    word_counts = [len(re.findall(r"\w+", p)) for p in preds]
    metrics["avg_output_words"] = sum(word_counts) / len(word_counts) if word_counts else 0.0

    per_sample_judge = None
    if openai_client:
        print("  Computing LLM Judge...")
        per_sample_judge = llm_judge(preds, refs, openai_client)
        metrics["llm_judge_score"] = sum(per_sample_judge) / len(per_sample_judge) if per_sample_judge else 0.0

    return metrics, per_sample_judge


# ============================================================================
# PER-MODEL RUNNERS
# ============================================================================

def run_gpt5(test_data, openai_client, checkpoint, checkpoint_key, checkpoint_path, template):
    """Run GPT-5 inference via OpenAI API. Checkpoints every 5 samples."""
    predictions = checkpoint.get(checkpoint_key, {}).get("predictions", [])
    inference_times = checkpoint.get(checkpoint_key, {}).get("inference_times", [])
    start_idx = len(predictions)

    if start_idx >= len(test_data):
        print(f"  Already complete ({start_idx} predictions)")
        return predictions, inference_times

    print(f"  Starting from sample {start_idx}/{len(test_data)}")

    for i in range(start_idx, len(test_data)):
        ex = test_data[i]
        q = ex["Question"].strip()
        ds = truncate_discharge_summary(ex["discharge_summary"])
        user_prompt = template.format(q=q, ds_excerpt=ds)

        try:
            t0 = time.time()
            pred = call_openai(user_prompt, "gpt-5", openai_client)
            elapsed = time.time() - t0
        except Exception as e:
            print(f"  Error on example {i}: {e}")
            pred = ""
            elapsed = 0.0

        predictions.append({
            "index": i, "note_id": ex.get("note_id", ""),
            "question": q, "reference": ex["Answer"].strip(),
            "prediction": pred, "original": ex["discharge_summary"],
            "inference_time_s": elapsed,
        })
        inference_times.append(elapsed)

        if (i + 1) % 5 == 0:
            checkpoint[checkpoint_key] = {"predictions": predictions, "inference_times": inference_times}
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False)
            avg_t = sum(inference_times) / len(inference_times)
            print(f"  [{i+1}/{len(test_data)}] avg_time={avg_t:.2f}s  [checkpoint saved]")

    return predictions, inference_times


def run_hf_model(test_data, model_id, template, checkpoint, checkpoint_key, checkpoint_path,
                 quantize: bool = True):
    """Run HuggingFace model inference. Checkpoints every 5 samples."""
    predictions = checkpoint.get(checkpoint_key, {}).get("predictions", [])
    inference_times = checkpoint.get(checkpoint_key, {}).get("inference_times", [])
    start_idx = len(predictions)

    if start_idx >= len(test_data):
        print(f"  Already complete ({start_idx} predictions)")
        return predictions, inference_times

    print(f"  Starting from sample {start_idx}/{len(test_data)}")
    tokenizer, model = load_hf_model(model_id, quantize=quantize)

    for i in range(start_idx, len(test_data)):
        ex = test_data[i]
        q = ex["Question"].strip()
        ds = truncate_discharge_summary(ex["discharge_summary"])
        user_prompt = template.format(q=q, ds_excerpt=ds)

        try:
            t0 = time.time()
            pred = call_huggingface(user_prompt, tokenizer, model, SYSTEM_PROMPT)
            elapsed = time.time() - t0
        except Exception as e:
            print(f"  Error on example {i}: {e}")
            pred = ""
            elapsed = 0.0
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()

        predictions.append({
            "index": i, "note_id": ex.get("note_id", ""),
            "question": q, "reference": ex["Answer"].strip(),
            "prediction": pred, "original": ex["discharge_summary"],
            "inference_time_s": elapsed,
        })
        inference_times.append(elapsed)

        if (i + 1) % 5 == 0:
            checkpoint[checkpoint_key] = {"predictions": predictions, "inference_times": inference_times}
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False)
            avg_t = sum(inference_times) / len(inference_times)
            print(f"  [{i+1}/{len(test_data)}] avg_time={avg_t:.2f}s  [checkpoint saved]")

    del model, tokenizer
    torch.cuda.empty_cache()
    return predictions, inference_times


# ============================================================================
# MAIN
# ============================================================================

def run_evaluation(data_path: str, output_dir: str, n_dev: int = 0, n_test: int = 200,
                   use_llm_judge: bool = False, resume_from: str = None,
                   approaches: List[str] = None):
    """
    Run all four approaches on the same 200 test samples.

    approaches: list of approach names to run. Default: all four.
        Options: "gpt5", "qwen32b_zero", "qwen32b_one", "agentic_notools", "agentic_tools"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    all_approaches = ["gpt5", "qwen32b_zero", "qwen32b_one", "qwen7b_zero", "phi35_zero"]
    if approaches is None:
        approaches = all_approaches

    # ── Load data ───────────────────────────────────────────────────────────
    test_data, exemplar = load_test_set(data_path, n_dev, n_test)
    exemplar_text = build_one_shot_exemplar_text(exemplar) if exemplar else ""

    # Build templates
    zero_shot_tmpl = ZERO_SHOT_TEMPLATE
    one_shot_tmpl = ONE_SHOT_TEMPLATE_PREFIX.replace("{exemplar}", exemplar_text)

    # ── Load checkpoint ────────────────────────────────────────────────────
    checkpoint = {}
    if resume_from and os.path.exists(resume_from):
        with open(resume_from, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        print(f"Resumed from checkpoint: {resume_from}")

    checkpoint_path = os.path.join(output_dir, "evaluate_200_checkpoint.json")

    # ── OpenAI client ──────────────────────────────────────────────────────
    openai_client = None
    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # ── Evaluation models (loaded once) ────────────────────────────────────
    print("Loading evaluation models...")
    nli_pipe = hf_pipeline(
        "text-classification", model="microsoft/deberta-large-mnli", device=-1
    )
    sbert = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    print("Evaluation models loaded.")

    all_results = {}

    # ────────────────────────────────────────────────────────────────────────
    # 1. GPT-5 (zero-shot)
    # ────────────────────────────────────────────────────────────────────────
    if "gpt5" in approaches:
        print(f"\n{'='*60}")
        print("APPROACH 1: GPT-5 (zero-shot)")
        print(f"{'='*60}")
        if not openai_client:
            print("  SKIPPED: No OPENAI_API_KEY set.")
        else:
            predictions, times = run_gpt5(
                test_data, openai_client, checkpoint, "gpt5", checkpoint_path, zero_shot_tmpl
            )

            preds = [p["prediction"] for p in predictions]
            refs = [p["reference"] for p in predictions]
            origs = [p["original"] for p in predictions]

            print("\nComputing metrics for GPT-5...")
            metrics, judge_scores = compute_all_metrics(
                preds, refs, origs, nli_pipe, sbert,
                openai_client if use_llm_judge else None
            )
            if judge_scores:
                for p, s in zip(predictions, judge_scores):
                    p["llm_judge_score"] = s
            metrics["avg_inference_time_s"] = sum(times) / len(times) if times else 0.0
            metrics["total_inference_time_s"] = sum(times)
            all_results["GPT-5"] = {"predictions": predictions, "metrics": metrics}
            print(f"  Results: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}")

    # ────────────────────────────────────────────────────────────────────────
    # 2. Qwen2.5-32B-Instruct (zero-shot)
    # ────────────────────────────────────────────────────────────────────────
    if "qwen32b_zero" in approaches:
        print(f"\n{'='*60}")
        print("APPROACH 2: Qwen2.5-32B-Instruct (zero-shot)")
        print(f"{'='*60}")
        predictions, times = run_hf_model(
            test_data, "Qwen/Qwen2.5-32B-Instruct", zero_shot_tmpl,
            checkpoint, "qwen32b_zero", checkpoint_path
        )

        preds = [p["prediction"] for p in predictions]
        refs = [p["reference"] for p in predictions]
        origs = [p["original"] for p in predictions]

        print("\nComputing metrics for Qwen2.5-32B (zero-shot)...")
        metrics, judge_scores = compute_all_metrics(
            preds, refs, origs, nli_pipe, sbert,
            openai_client if use_llm_judge else None
        )
        if judge_scores:
            for p, s in zip(predictions, judge_scores):
                p["llm_judge_score"] = s
        metrics["avg_inference_time_s"] = sum(times) / len(times) if times else 0.0
        metrics["total_inference_time_s"] = sum(times)
        all_results["Qwen2.5-32B (zero-shot)"] = {"predictions": predictions, "metrics": metrics}
        print(f"  Results: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}")

    # ────────────────────────────────────────────────────────────────────────
    # 3. Qwen2.5-32B-Instruct (one-shot)
    # ────────────────────────────────────────────────────────────────────────
    if "qwen32b_one" in approaches:
        print(f"\n{'='*60}")
        print("APPROACH 3: Qwen2.5-32B-Instruct (one-shot)")
        print(f"{'='*60}")
        predictions, times = run_hf_model(
            test_data, "Qwen/Qwen2.5-32B-Instruct", one_shot_tmpl,
            checkpoint, "qwen32b_one", checkpoint_path
        )

        preds = [p["prediction"] for p in predictions]
        refs = [p["reference"] for p in predictions]
        origs = [p["original"] for p in predictions]

        print("\nComputing metrics for Qwen2.5-32B (one-shot)...")
        metrics, judge_scores = compute_all_metrics(
            preds, refs, origs, nli_pipe, sbert,
            openai_client if use_llm_judge else None
        )
        if judge_scores:
            for p, s in zip(predictions, judge_scores):
                p["llm_judge_score"] = s
        metrics["avg_inference_time_s"] = sum(times) / len(times) if times else 0.0
        metrics["total_inference_time_s"] = sum(times)
        all_results["Qwen2.5-32B (one-shot)"] = {"predictions": predictions, "metrics": metrics}
        print(f"  Results: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}")

    # ────────────────────────────────────────────────────────────────────────
    # 4. Qwen2.5-7B-Instruct (zero-shot) — single-model ablation
    # ────────────────────────────────────────────────────────────────────────
    if "qwen7b_zero" in approaches:
        print(f"\n{'='*60}")
        print("APPROACH 4: Qwen2.5-7B-Instruct (zero-shot)")
        print(f"{'='*60}")
        predictions, times = run_hf_model(
            test_data, "Qwen/Qwen2.5-7B-Instruct", zero_shot_tmpl,
            checkpoint, "qwen7b_zero", checkpoint_path,
            quantize=False,
        )

        preds = [p["prediction"] for p in predictions]
        refs = [p["reference"] for p in predictions]
        origs = [p["original"] for p in predictions]

        print("\nComputing metrics for Qwen2.5-7B (zero-shot)...")
        metrics, judge_scores = compute_all_metrics(
            preds, refs, origs, nli_pipe, sbert,
            openai_client if use_llm_judge else None
        )
        if judge_scores:
            for p, s in zip(predictions, judge_scores):
                p["llm_judge_score"] = s
        metrics["avg_inference_time_s"] = sum(times) / len(times) if times else 0.0
        metrics["total_inference_time_s"] = sum(times)
        all_results["Qwen2.5-7B (zero-shot)"] = {"predictions": predictions, "metrics": metrics}
        print(f"  Results: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}")

    # ────────────────────────────────────────────────────────────────────────
    # 5. Phi-3.5-mini-instruct (zero-shot) — single-model ablation
    # ────────────────────────────────────────────────────────────────────────
    if "phi35_zero" in approaches:
        print(f"\n{'='*60}")
        print("APPROACH 5: Phi-3.5-mini-instruct (zero-shot)")
        print(f"{'='*60}")
        predictions, times = run_hf_model(
            test_data, "microsoft/Phi-3.5-mini-instruct", zero_shot_tmpl,
            checkpoint, "phi35_zero", checkpoint_path,
            quantize=False,
        )

        preds = [p["prediction"] for p in predictions]
        refs = [p["reference"] for p in predictions]
        origs = [p["original"] for p in predictions]

        print("\nComputing metrics for Phi-3.5-mini (zero-shot)...")
        metrics, judge_scores = compute_all_metrics(
            preds, refs, origs, nli_pipe, sbert,
            openai_client if use_llm_judge else None
        )
        if judge_scores:
            for p, s in zip(predictions, judge_scores):
                p["llm_judge_score"] = s
        metrics["avg_inference_time_s"] = sum(times) / len(times) if times else 0.0
        metrics["total_inference_time_s"] = sum(times)
        all_results["Phi-3.5-mini (zero-shot)"] = {"predictions": predictions, "metrics": metrics}
        print(f"  Results: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}")

    # ────────────────────────────────────────────────────────────────────────
    # SAVE RESULTS
    # ────────────────────────────────────────────────────────────────────────
    # Save predictions
    pred_path = os.path.join(output_dir, f"evaluate_200_predictions_{timestamp}.json")
    save_preds = {}
    for name, data in all_results.items():
        save_preds[name] = data["predictions"]
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(save_preds, f, indent=2, ensure_ascii=False)
    print(f"\nPredictions saved to {pred_path}")

    # Save metrics JSON
    metrics_json_path = os.path.join(output_dir, f"evaluate_200_metrics_{timestamp}.json")
    save_metrics = {name: data["metrics"] for name, data in all_results.items()}
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"Metrics JSON saved to {metrics_json_path}")

    # Save metrics CSV
    if all_results:
        metrics_csv_path = os.path.join(output_dir, f"evaluate_200_metrics_{timestamp}.csv")
        metric_names = list(next(iter(all_results.values()))["metrics"].keys())
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Approach"] + metric_names)
            for name, data in all_results.items():
                row = [name] + [f"{data['metrics'].get(m, 0):.4f}" for m in metric_names]
                writer.writerow(row)
        print(f"Metrics CSV saved to {metrics_csv_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE (200 test samples)")
    print(f"{'='*80}")
    if all_results:
        metric_names = list(next(iter(all_results.values()))["metrics"].keys())
        header = f"{'Approach':<30}" + "".join(f"{m:<14}" for m in metric_names)
        print(header)
        print("-" * len(header))
        for name, data in all_results.items():
            row = f"{name:<30}" + "".join(f"{data['metrics'].get(m, 0):<14.4f}" for m in metric_names)
            print(row)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified 200-report evaluation")
    parser.add_argument("--data", default="data/MeDiSumQA.jsonl", help="Input JSONL file")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--n-dev", type=int, default=0, help="Number of dev samples to skip")
    parser.add_argument("--n-test", type=int, default=200, help="Number of test samples")
    parser.add_argument("--llm-judge", action="store_true", help="Enable GPT-4 LLM-as-judge")
    parser.add_argument("--resume", default=None, help="Path to checkpoint JSON to resume")
    parser.add_argument(
        "--approaches", nargs="+", default=None,
        choices=["gpt5", "qwen32b_zero", "qwen32b_one", "qwen7b_zero", "phi35_zero"],
        help="Which approaches to run (default: all)"
    )

    args = parser.parse_args()
    run_evaluation(
        data_path=args.data,
        output_dir=args.output,
        n_dev=args.n_dev,
        n_test=args.n_test,
        use_llm_judge=args.llm_judge,
        resume_from=args.resume,
        approaches=args.approaches,
    )
