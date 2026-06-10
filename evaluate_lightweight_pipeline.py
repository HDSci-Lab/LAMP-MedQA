"""
Lightweight Dual-Model Pipeline for Medical Text Simplification
================================================================
Tests whether two smaller models can match Qwen2.5-32B performance
via multi-step reasoning and metric-based validation.

Pipeline:
  1. EXTRACTOR  (Model A – Qwen3.5-9B)
       Extract relevant info from discharge summary for the patient question.
  2. EXTRACTION VALIDATOR  (automated metrics)
       SARI >45 %  |  Token F1 >0.30  |  SBERT >0.65
       On fail → Model B rewrites feedback → Model A retries (up to N).
  3. SIMPLIFIER  (Model A – same model, readability-focused prompt)
       Rewrite the extraction in ≤8th-grade English.
       Optional: --use-tools enables a MedlinePlus glossary tool that provides
       plain-English definitions of medical terms to help the simplifier.
  4. READABILITY VALIDATOR  (automated metric)
       FKGL < 10
       On fail → Model B provides readability feedback → Model A retries.
  5. Final answer collected; all metrics computed on held-out test set.

Models:
  Model A (extractor / simplifier) : Qwen/Qwen3.5-9B
  Model B (validator / feedback)   : microsoft/Phi-3.5-mini-instruct

No LLM-as-a-judge is used for internal validation – only automated metrics.

Tool augmentation (optional):
  The simplifier can be augmented with an offline MedlinePlus glossary
  (pre-downloaded via download_medlineplus_glossary.py). Model B extracts
  medical jargon from the extraction, looks up plain-English definitions,
  and provides them to the simplifier as context. Run with --use-tools flag.

Data splits (from 415 MeDiSumQA records after excluding one-shot exemplar, seed=42):
  Dev  : 20 reports  (pipeline tuning)
  Test : 200 reports (held-out final evaluation)

Usage:
  # Without tools
  python evaluate_lightweight_pipeline.py --split dev   --output results
  python evaluate_lightweight_pipeline.py --split test  --output results

  # With MedlinePlus glossary tool
  python evaluate_lightweight_pipeline.py --split dev  --output results --use-tools
  python evaluate_lightweight_pipeline.py --split test --output results --use-tools
"""

import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import csv
import random
import argparse
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import Counter
from typing import List, Dict, Optional, Tuple

import torch
import nltk
import textstat
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline as hf_pipeline,
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from bert_score import score as bert_score_fn
from openai import OpenAI

# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# ============================================================================
# FIX: DynamicCache / SlidingWindowCache compatibility across transformers versions
# ============================================================================
from transformers import DynamicCache
if not hasattr(DynamicCache, 'seen_tokens'):
    def _get_seen_tokens(self):
        for attr in ('_seen_tokens', 'seen_tokens', '_seq_len_cached'):
            if hasattr(self, attr):
                return getattr(self, attr)
        if hasattr(self, 'key_cache') and self.key_cache:
            return self.key_cache[0].shape[-2]
        return 0
    DynamicCache.seen_tokens = property(_get_seen_tokens)

import transformers.cache_utils as _cache_utils
if not hasattr(_cache_utils, 'SlidingWindowCache'):
    class _SlidingWindowCache(DynamicCache):
        pass
    _cache_utils.SlidingWindowCache = _SlidingWindowCache
if not hasattr(_cache_utils, 'StaticCache'):
    class _StaticCache(DynamicCache):
        pass
    _cache_utils.StaticCache = _StaticCache

# FIX: transformers 5.x calls _init_weights on int8 quantized tensors during
# _initialize_missing_keys, which crashes with "normal_kernel_cuda not
# implemented for 'Char'".  Silently skip – weights are already loaded.
import transformers.modeling_utils as _modeling_utils
_orig__initialize_weights = _modeling_utils.PreTrainedModel._initialize_weights
def _safe__initialize_weights(self, module, *args, **kwargs):
    try:
        _orig__initialize_weights(self, module, *args, **kwargs)
    except (NotImplementedError, TypeError):
        pass
_modeling_utils.PreTrainedModel._initialize_weights = _safe__initialize_weights

# ============================================================================
# ENVIRONMENT & REPRODUCIBILITY
# ============================================================================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

HF_TOKEN = os.environ.get("HF_TOKEN", None)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

# ============================================================================
# GENERATION SETTINGS
# ============================================================================
MAX_NEW_TOKENS = 512
MAX_INPUT_LENGTH = 42000
MAX_RETRIES = 3  # max metric-gate retries per stage

# ============================================================================
# METRIC THRESHOLDS  (internal validation gates)
# ============================================================================
EXTRACTION_THRESHOLDS = {
    # Reference-free: SBERT cosine between extraction and the patient question.
    # Tuned on the 20-sample dev split; see results/dev_threshold_tune.json.
    "sbert_eq": 0.50,
}
READABILITY_THRESHOLDS = {
    "fkgl": 10.0,       # FKGL < 10
}

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================
MODEL_A_ID = "Qwen/Qwen2.5-7B-Instruct"          # extractor + simplifier
MODEL_B_ID = "microsoft/Phi-3.5-mini-instruct"  # validator / feedback

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

# ============================================================================
# PROMPT TEMPLATES  (LangChain PromptTemplate objects)
# ============================================================================

# --- Stage 1: Extraction ---------------------------------------------------
EXTRACT_SYSTEM = (
    "You are a medical text summariser/simplifier. A patient asks the following "
    "question about their discharge summary. Please think step by step, first "
    "review what the key information that patient is asking, and then extract "
    "the relevant information from the discharge summary and produce a short, "
    "clear, patient-facing answer in plain English (1-2 sentences). "
    "Do NOT add medical jargon."
)

EXTRACT_PROMPT = PromptTemplate(
    input_variables=["question", "discharge_summary"],
    template=(
        "Question:\n{question}\n\n"
        "Discharge summary:\n{discharge_summary}\n\n"
        "Answer:"
    ),
)

# --- Stage 1 retry: extraction with feedback --------------------------------
EXTRACT_RETRY_PROMPT = PromptTemplate(
    input_variables=["question", "discharge_summary", "previous_answer", "feedback"],
    template=(
        "Question:\n{question}\n\n"
        "Discharge summary:\n{discharge_summary}\n\n"
        "Your previous answer was:\n{previous_answer}\n\n"
        "Feedback on your answer:\n{feedback}\n\n"
        "Please produce an improved short, clear, patient-facing answer in plain "
        "English (1-2 sentences). Do NOT add medical jargon.\n\n"
        "Improved Answer:"
    ),
)

# --- Stage 2: Readability simplification ------------------------------------
SIMPLIFY_SYSTEM = (
    "You are a plain-language editor. Rewrite the following medical answer so "
    "that a patient with no medical background can understand it. Use short "
    "sentences, everyday words, and keep it to 1-2 sentences. "
    "Do NOT add information that is not in the original answer."
)

SIMPLIFY_PROMPT = PromptTemplate(
    input_variables=["extracted_answer"],
    template=(
        "Medical answer to simplify:\n{extracted_answer}\n\n"
        "Plain-English rewrite:"
    ),
)

# --- Stage 2 retry: simplification with feedback ----------------------------
SIMPLIFY_RETRY_PROMPT = PromptTemplate(
    input_variables=["extracted_answer", "previous_simplification", "feedback"],
    template=(
        "Medical answer to simplify:\n{extracted_answer}\n\n"
        "Your previous simplification was:\n{previous_simplification}\n\n"
        "Feedback:\n{feedback}\n\n"
        "Please rewrite in simpler language (target: 8th-grade reading level). "
        "Use short sentences and everyday words. Keep it to 1-2 sentences.\n\n"
        "Improved plain-English rewrite:"
    ),
)

# --- Validator feedback prompts (Model B) -----------------------------------
# IMPORTANT: the reference (gold) answer is NEVER passed to Model B.  Feedback
# is grounded in the discharge summary and the patient's question only — no
# inference-time oracle access.
EXTRACTION_FEEDBACK_SYSTEM = (
    "You are a quality-control reviewer for medical text simplification. "
    "The extraction below did not appear to address the patient's question. "
    "Provide brief, actionable feedback to help the writer improve."
)

EXTRACTION_FEEDBACK_PROMPT = PromptTemplate(
    input_variables=["question", "discharge_summary", "current_answer", "metrics_report"],
    template=(
        "Patient question: {question}\n\n"
        "Discharge summary (source of truth):\n{discharge_summary}\n\n"
        "Current answer: {current_answer}\n\n"
        "Quality check:\n{metrics_report}\n\n"
        "Provide 1-2 sentences of specific, actionable feedback so the answer "
        "more directly addresses the patient's question while staying grounded "
        "in the discharge summary. Do NOT invent facts.\n\n"
        "Feedback:"
    ),
)

# --- Extraction LLM-classifier prompt (Model B) -----------------------------
# Used when --gate-mode=llm.  Model B classifies the extraction in a single
# call and, if rejected, returns its own actionable feedback string.  Still
# reference-free: the gold answer is never passed in.
EXTRACTION_CLASSIFIER_SYSTEM = (
    "You are a strict quality-control reviewer for medical question answering. "
    "You will be shown a patient question, the source discharge summary, and a "
    "candidate answer. Your job is to decide whether the candidate answer "
    "directly addresses the patient's question and is grounded in the "
    "discharge summary. Reply with a single JSON object only."
)

EXTRACTION_CLASSIFIER_PROMPT = PromptTemplate(
    input_variables=["question", "discharge_summary", "current_answer"],
    template=(
        "Patient question:\n{question}\n\n"
        "Discharge summary (source of truth):\n{discharge_summary}\n\n"
        "Candidate answer:\n{current_answer}\n\n"
        "Decide:\n"
        "  - Classification_result: true if the candidate answer adequately "
        "addresses the patient's question with content supported by the "
        "discharge summary; false otherwise.\n"
        "  - False_evidence: if Classification_result is false, give 1-2 "
        "sentences of specific, actionable feedback identifying the content "
        "gap or grounding issue. If Classification_result is true, return an "
        "empty string.\n\n"
        "Reply with ONLY a single JSON object on one line, no markdown, no "
        "prose before or after:\n"
        '{{"Classification_result": <true|false>, "False_evidence": "<text>"}}'
    ),
)

READABILITY_FEEDBACK_SYSTEM = (
    "You are a readability reviewer. The simplified text below has a "
    "Flesch-Kincaid Grade Level that is too high. "
    "Provide brief, actionable feedback to make it simpler."
)

# --- Tool-augmented simplification prompts --------------------------------
SIMPLIFY_WITH_GLOSSARY_SYSTEM = (
    "You are a plain-language editor. Rewrite the following medical answer so "
    "that a patient with no medical background can understand it. "
    "You have been given a glossary of plain-English definitions for medical "
    "terms found in the answer — use these definitions to replace jargon. "
    "Use short sentences, everyday words, and keep it to 1-2 sentences. "
    "Do NOT add information that is not in the original answer."
)

SIMPLIFY_WITH_GLOSSARY_PROMPT = PromptTemplate(
    input_variables=["extracted_answer", "glossary"],
    template=(
        "Medical answer to simplify:\n{extracted_answer}\n\n"
        "Glossary of plain-English replacements:\n{glossary}\n\n"
        "Plain-English rewrite (use the glossary above to replace medical terms):"
    ),
)

SIMPLIFY_WITH_GLOSSARY_RETRY_PROMPT = PromptTemplate(
    input_variables=["extracted_answer", "glossary", "previous_simplification", "feedback"],
    template=(
        "Medical answer to simplify:\n{extracted_answer}\n\n"
        "Glossary of plain-English replacements:\n{glossary}\n\n"
        "Your previous simplification was:\n{previous_simplification}\n\n"
        "Feedback:\n{feedback}\n\n"
        "Please rewrite in simpler language (target: 8th-grade reading level). "
        "Use the glossary above. Keep it to 1-2 sentences.\n\n"
        "Improved plain-English rewrite:"
    ),
)

READABILITY_FEEDBACK_PROMPT = PromptTemplate(
    input_variables=["current_text", "fkgl_score"],
    template=(
        "Current text:\n{current_text}\n\n"
        "Current FKGL score: {fkgl_score} (target: below 10)\n\n"
        "Provide 1-2 sentences of specific feedback on which words or "
        "phrases should be simplified and how.\n\n"
        "Feedback:"
    ),
)


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_id: str):
    """Load a HuggingFace model and tokenizer."""
    print(f"  Loading {model_id} ...")
    token = HF_TOKEN if "Qwen" in model_id else None

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=token, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        device_map="auto",
        torch_dtype=torch.float16,
        token=token,
        trust_remote_code=True,
    )

    # Phi models may need eager attention
    if "Phi-3" in model_id or "phi-3" in model_id or "Phi-4" in model_id or "phi-4" in model_id:
        load_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    print(f"  {model_id} loaded successfully.")
    return tokenizer, model


def format_chat_prompt(tokenizer, user_text: str, system_text: str = None) -> str:
    """Format prompt using tokenizer's chat template."""
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            combined = f"{system_text}\n\n{user_text}" if system_text else user_text
            messages = [{"role": "user", "content": combined}]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
    return (system_text + "\n\n" + user_text) if system_text else user_text


def generate(tokenizer, model, user_text: str, system_text: str = None,
             max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Run a single generation pass."""
    prompt = format_chat_prompt(tokenizer, user_text, system_text)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response.strip()


# ============================================================================
# MEDLINEPLUS OFFLINE GLOSSARY  (tool-augmented simplifier)
# ============================================================================

_glossary_data: Dict[str, str] = {}


def load_glossary(glossary_path: str = "data/medlineplus_glossary.json"):
    """Load the pre-downloaded MedlinePlus glossary from JSON."""
    global _glossary_data
    if _glossary_data:
        return  # already loaded
    if not os.path.exists(glossary_path):
        print(f"  WARNING: Glossary not found at {glossary_path} — "
              "tool-augmented mode will use empty glossary.")
        return
    with open(glossary_path, "r", encoding="utf-8") as f:
        _glossary_data = json.load(f)
    print(f"  Loaded MedlinePlus glossary: {len(_glossary_data)} terms.")


# Common terms to skip (too generic)
_SKIP_TERMS = {
    "patient", "doctor", "hospital", "treatment", "medication", "medicine",
    "diagnosis", "symptom", "condition", "disease", "surgery", "procedure",
    "blood", "heart", "pain", "test", "dose", "drug", "care", "health",
}


def extract_medical_terms(text: str, tokenizer=None, model=None,
                          system_prompt: str = None) -> List[str]:
    """
    Extract medical/clinical terms from text that a layperson might not know.
    Uses Model B (lightweight) for extraction when available, falls back to
    matching against the glossary keys.
    """
    if model is not None and tokenizer is not None:
        prompt = (
            "List only the medical or clinical terms in the text below that "
            "an average patient would NOT understand. Return one term per line, "
            "nothing else. If there are no such terms, return NONE.\n\n"
            f"Text: {text}\n\nMedical terms:"
        )
        raw = generate(tokenizer, model, prompt, system_text=system_prompt,
                       max_new_tokens=150)
        terms = []
        for line in raw.strip().split("\n"):
            term = re.sub(r"^[\d\.\-\*\s]+", "", line).strip().lower()
            term = re.sub(r"[^a-z\s\-]", "", term).strip()
            if term and term != "none" and len(term) > 2 and term not in _SKIP_TERMS:
                terms.append(term)
        return terms[:10]  # cap at 10

    # Fallback: match words in text against glossary keys
    text_lower = text.lower()
    found = []
    for term in _glossary_data:
        if term in text_lower and term not in _SKIP_TERMS:
            found.append(term)
    return sorted(found, key=len, reverse=True)[:10]


def build_glossary_for_text(text: str, tokenizer=None, model=None,
                            system_prompt: str = None) -> str:
    """
    Extract medical terms from text and look them up in the offline glossary.
    Returns a formatted glossary string for prompt injection.
    """
    terms = extract_medical_terms(text, tokenizer, model, system_prompt)
    entries = []
    for term in terms:
        # Try exact match, then fuzzy substring match against glossary keys
        definition = _glossary_data.get(term)
        if not definition:
            # Try to find a glossary key that contains this term
            for gkey, gdef in _glossary_data.items():
                if term in gkey or gkey in term:
                    definition = gdef
                    break
        if definition:
            entries.append(f"- {term}: {definition}")

    if not entries:
        return "(No additional definitions available)"
    return "\n".join(entries)


# ============================================================================
# LANGCHAIN WRAPPERS
# ============================================================================

class LangChainLocalLLM:
    """
    Thin LangChain-compatible wrapper around a local HuggingFace model.
    Exposes an `invoke(prompt_str) -> str` interface for use in LCEL chains.
    """

    def __init__(self, tokenizer, model, system_prompt: str = None,
                 max_new_tokens: int = MAX_NEW_TOKENS):
        self.tokenizer = tokenizer
        self.model = model
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens

    def invoke(self, text: str) -> str:
        if isinstance(text, dict):
            text = str(text)
        return generate(
            self.tokenizer, self.model, text,
            system_text=self.system_prompt,
            max_new_tokens=self.max_new_tokens,
        )


def build_chain(prompt_template: PromptTemplate, llm: LangChainLocalLLM):
    """Build a LangChain LCEL chain: prompt → LLM → strip output."""
    return prompt_template | RunnableLambda(lambda x: llm.invoke(x.text))


# ============================================================================
# PER-SAMPLE METRIC FUNCTIONS  (used inside the validation gates)
# ============================================================================

def _ngrams(sentence: str, n: int) -> List[tuple]:
    tokens = re.findall(r"\w+", sentence.lower())
    return [tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]


def sample_sari(pred: str, ref: str, orig: str) -> float:
    """Single-sample SARI (0-100)."""
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


def sample_token_f1(pred: str, ref: str) -> float:
    ta = re.findall(r"\w+", pred.lower())
    tb = re.findall(r"\w+", ref.lower())
    common = Counter(ta) & Counter(tb)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(ta) if ta else 0.0
    r = num_same / len(tb) if tb else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def sample_fkgl(text: str) -> float:
    return textstat.flesch_kincaid_grade(text)


# SBERT loaded once at module level
print("Loading evaluation models ...")
sbert_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
print("Evaluation models loaded.")


def sample_sbert(pred: str, ref: str) -> float:
    emb_p = sbert_model.encode([pred], convert_to_numpy=True)
    emb_r = sbert_model.encode([ref], convert_to_numpy=True)
    return float(cosine_similarity(emb_p, emb_r)[0, 0])


# ============================================================================
# VALIDATION GATES
# ============================================================================

def check_extraction(extraction: str, question: str) -> Tuple[bool, Dict[str, float], str]:
    """
    Reference-free extraction gate (gate-mode = sbert): the extraction must
    be relevant to the patient's question.  Score is SBERT cosine similarity
    between the extraction and the question; threshold is tuned on the dev
    split.  Returns (passed, metrics_dict, human_readable_report).
    """
    sbert_eq = sample_sbert(extraction, question)
    metrics = {"sbert_eq": sbert_eq}

    if sbert_eq <= EXTRACTION_THRESHOLDS["sbert_eq"]:
        report = (
            f"SBERT(extraction, question)={sbert_eq:.2f} "
            f"(need >{EXTRACTION_THRESHOLDS['sbert_eq']}); "
            f"the extraction does not appear to address the asked question."
        )
        return False, metrics, report

    return True, metrics, "Extraction relevance check passed."


def _parse_classifier_output(raw: str) -> Tuple[bool, str]:
    """
    Parse Model B's classifier reply into (passed, feedback).

    Tolerant to:
      - extra prose around the JSON object
      - True/true/yes/passed string variants for Classification_result
      - missing False_evidence field
      - completely malformed output (treated as 'failed' with the raw text
        truncated as the feedback, so the retry loop still gets a hint)
    """
    if not raw:
        return False, "Classifier returned empty output."

    text = raw.strip()
    # Pull out the first {...} block, allowing newlines inside.
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            cr_raw = obj.get("Classification_result",
                             obj.get("classification_result"))
            fe_raw = obj.get("False_evidence",
                             obj.get("false_evidence", "")) or ""
            if isinstance(cr_raw, str):
                cr = cr_raw.strip().lower() in ("true", "yes", "pass", "passed", "1")
            else:
                cr = bool(cr_raw)
            return cr, str(fe_raw).strip()
        except Exception:
            pass

    # Fallback: keyword sniff in the first short window.
    lo = text.lower()[:80]
    if ('"classification_result": true' in text.lower()
            or "classification_result: true" in text.lower()
            or lo.startswith(("true", "yes", "passed"))):
        return True, ""
    return False, text[:300]


def check_extraction_llm(extraction: str, question: str, ds: str,
                         classifier_chain) -> Tuple[bool, Dict[str, str], str]:
    """
    Reference-free extraction gate (gate-mode = llm): Model B classifies the
    candidate answer as True (passes) or False (fails) given the question
    and the discharge summary.  When False, Model B's own
    ``False_evidence`` field is used as the actionable retry feedback — so
    the classifier replaces both the metric gate AND the separate feedback
    chain.  Returns (passed, metrics_dict, human_readable_report).
    """
    raw = classifier_chain.invoke({
        "question": question,
        "discharge_summary": ds,
        "current_answer": extraction,
    })
    passed, feedback = _parse_classifier_output(raw)
    metrics = {
        "classifier_passed": bool(passed),
        "classifier_raw": raw[:500] if raw else "",
        "classifier_feedback": feedback[:500] if feedback else "",
    }
    if passed:
        report = "LLM classifier returned True."
    else:
        report = feedback or "LLM classifier returned False (no feedback parsed)."
    return passed, metrics, report


def check_readability(text: str) -> Tuple[bool, float, str]:
    """
    Check readability against FKGL threshold.
    Returns (passed, fkgl_value, report).
    """
    fkgl = sample_fkgl(text)
    passed = fkgl < READABILITY_THRESHOLDS["fkgl"]
    report = (
        f"FKGL={fkgl:.1f} — passed (< {READABILITY_THRESHOLDS['fkgl']})"
        if passed
        else f"FKGL={fkgl:.1f} — FAILED (need < {READABILITY_THRESHOLDS['fkgl']})"
    )
    return passed, fkgl, report


# ============================================================================
# DATA LOADING & SPLITTING
# ============================================================================

ONE_SHOT_EXEMPLAR_NOTE_ID = "14171200-DS-21"

def load_and_split(data_path: str, n_dev: int = 20, n_test: int = 200):
    """
    Load MeDiSumQA and create reproducible dev / test splits.
    Excludes the one-shot exemplar (note_id=14171200-DS-21) used in
    the baseline one-shot evaluation to prevent train/test leakage.
    Returns (dev_data, test_data).
    """
    data = []
    with open(data_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                # Exclude the one-shot exemplar used in baseline evaluation
                if record.get("note_id") == ONE_SHOT_EXEMPLAR_NOTE_ID:
                    continue
                data.append(record)

    print(f"Loaded {len(data)} records (excluded one-shot exemplar {ONE_SHOT_EXEMPLAR_NOTE_ID})")

    random.seed(SEED)
    shuffled = random.sample(data, len(data))

    dev_data = shuffled[:n_dev]
    test_data = shuffled[n_dev: n_dev + n_test]
    remaining = shuffled[n_dev + n_test:]

    print(f"Data split — Dev: {len(dev_data)}, Test: {len(test_data)}, "
          f"Remaining: {len(remaining)}  (total: {len(data)})")
    return dev_data, test_data


# ============================================================================
# SINGLE-SAMPLE PIPELINE
# ============================================================================

_DUMMY_EXTRACTION_FEEDBACK = (
    "Please make the answer more concise and accurate, focusing only on "
    "information directly relevant to the patient's question."
)
_DUMMY_READABILITY_FEEDBACK = (
    "Please use simpler, shorter words and sentences so the answer is easier "
    "for a patient to understand."
)


def run_sample_pipeline(
    example: Dict,
    extractor_chain,
    simplifier_chain,
    extract_retry_chain,
    simplify_retry_chain,
    feedback_extraction_chain,
    feedback_readability_chain,
    verbose: bool = False,
    use_tools: bool = False,
    simplifier_glossary_chain=None,
    simplify_glossary_retry_chain=None,
    tok_b=None,
    model_b=None,
    classifier_chain=None,
    gate_mode: str = "sbert",
    # ── Ablation flags ──────────────────────────────────────────────────────
    no_extraction: bool = False,
    no_simplification: bool = False,
    no_extraction_validation: bool = False,
    no_readability_validation: bool = False,
    max_retries: int = MAX_RETRIES,
    dummy_feedback: bool = False,
) -> Dict:
    """
    Run the full extract → validate → simplify → validate pipeline
    for a single example.

    gate_mode (extraction validation):
        "sbert" — SBERT(extraction, question) > tau (default; tau tuned on dev).
        "llm"   — Model B classifier returns Classification_result + False_evidence
                  in one call; replaces both metric gate and feedback chain.

    Ablation flags:
        no_extraction             — skip Stage 1; pass raw discharge summary to simplifier
        no_simplification         — skip Stage 3; use extraction output as final answer
        no_extraction_validation  — skip Stage 2 accuracy gate (no retry loop)
        no_readability_validation — skip Stage 4 readability gate (no retry loop)
        max_retries               — override the number of allowed retries per gate
        dummy_feedback   — replace Model B feedback with a fixed generic string
    """
    question = example.get("Question", "").strip()
    ds = example.get("discharge_summary", "")
    ds_trunc = ds[:42000] + "..." if len(ds) > 42000 else ds
    reference = example.get("Answer", "").strip()
    note_id = example.get("note_id", "")

    trace = {
        "note_id": note_id,
        "question": question,
        "reference": reference,
        "original": ds,
        "extraction_attempts": [],
        "simplification_attempts": [],
    }

    # ── STAGE 1: Extraction ─────────────────────────────────────────────────
    if no_extraction:
        extraction = ds_trunc
        trace["extraction_skipped"] = True
        if verbose:
            print(f"    [Extract] SKIPPED — using raw discharge summary")
    else:
        extraction = extractor_chain.invoke({
            "question": question,
            "discharge_summary": ds_trunc,
        })
        trace["extraction_attempts"].append(extraction)
        if verbose:
            print(f"    [Extract] {extraction[:100]}...")

    # ── STAGE 2: Extraction validation + retry loop ─────────────────────────
    if no_extraction or no_extraction_validation:
        if verbose and not no_extraction:
            print(f"    [ExtVal] SKIPPED (no_extraction_validation=True)")
    else:
        if gate_mode not in ("sbert", "llm"):
            raise ValueError(f"Unknown gate_mode={gate_mode!r}; expected 'sbert' or 'llm'")
        if gate_mode == "llm" and classifier_chain is None:
            raise ValueError("gate_mode='llm' requires classifier_chain")

        for retry in range(max_retries):
            # Reference-free gate: the gold answer is NOT used at inference time.
            if gate_mode == "sbert":
                passed, ext_metrics, report = check_extraction(extraction, question)
            else:  # gate_mode == "llm"
                passed, ext_metrics, report = check_extraction_llm(
                    extraction, question, ds_trunc, classifier_chain
                )
            trace[f"extraction_metrics_attempt_{retry}"] = ext_metrics

            if passed:
                if verbose:
                    if gate_mode == "sbert":
                        print(f"    [ExtVal/sbert] PASSED "
                              f"(sbert_eq={ext_metrics['sbert_eq']:.2f}) "
                              f"on attempt {retry + 1}")
                    else:
                        print(f"    [ExtVal/llm] PASSED on attempt {retry + 1}")
                break

            if retry == max_retries - 1:
                if verbose:
                    print(f"    [ExtVal/{gate_mode}] FAILED after {max_retries} "
                          f"attempts — using last")
                break

            # Decide the feedback string for the retry.
            #   sbert mode: route the failing-metric report through Model B's
            #               feedback chain to phrase it as actionable advice.
            #   llm mode  : the classifier already returned its own feedback in
            #               `report`, so we just reuse it (no separate chain
            #               call).
            # Reference is NEVER passed in either path.
            if dummy_feedback:
                feedback = _DUMMY_EXTRACTION_FEEDBACK
            elif gate_mode == "llm":
                feedback = report
            else:
                feedback = feedback_extraction_chain.invoke({
                    "question": question,
                    "discharge_summary": ds_trunc,
                    "current_answer": extraction,
                    "metrics_report": report,
                })

            if verbose:
                print(f"    [ExtFeedback] {feedback[:80]}...")

            # Retry extraction with feedback
            extraction = extract_retry_chain.invoke({
                "question": question,
                "discharge_summary": ds_trunc,
                "previous_answer": extraction,
                "feedback": feedback,
            })
            trace["extraction_attempts"].append(extraction)

            if verbose:
                print(f"    [ExtRetry-{retry + 1}] {extraction[:100]}...")

    # ── STAGE 3: Readability simplification ─────────────────────────────────
    if no_simplification:
        simplified = extraction
        trace["simplification_skipped"] = True
        if verbose:
            print(f"    [Simplify] SKIPPED — using extraction as final answer")
    else:
        # If tool-augmented mode, build a glossary for medical terms in extraction
        glossary_text = ""
        if use_tools:
            glossary_text = build_glossary_for_text(
                extraction, tokenizer=tok_b, model=model_b,
                system_prompt="You identify medical jargon in text."
            )
            trace["glossary_used"] = glossary_text
            if verbose:
                n_entries = glossary_text.count("\n- ") + (1 if glossary_text.startswith("- ") else 0)
                print(f"    [Glossary] {n_entries} MedlinePlus definitions retrieved")

        if use_tools and simplifier_glossary_chain is not None:
            simplified = simplifier_glossary_chain.invoke({
                "extracted_answer": extraction,
                "glossary": glossary_text,
            })
        else:
            simplified = simplifier_chain.invoke({
                "extracted_answer": extraction,
            })
        trace["simplification_attempts"].append(simplified)

        if verbose:
            print(f"    [Simplify] {simplified[:100]}...")

    # ── STAGE 4: Readability validation + retry loop ────────────────────────
    if no_simplification or no_readability_validation:
        if verbose and not no_simplification:
            print(f"    [ReadVal] SKIPPED (no_readability_validation=True)")
    else:
        for retry in range(max_retries):
            passed, fkgl, report = check_readability(simplified)
            trace[f"readability_fkgl_attempt_{retry}"] = fkgl

            if passed:
                if verbose:
                    print(f"    [ReadVal] PASSED (FKGL={fkgl:.1f}) on attempt {retry + 1}")
                break

            if retry == max_retries - 1:
                if verbose:
                    print(f"    [ReadVal] FAILED after {max_retries} attempts — using last")
                break

            # Get readability feedback from Model B (or use dummy)
            if dummy_feedback:
                feedback = _DUMMY_READABILITY_FEEDBACK
            else:
                feedback = feedback_readability_chain.invoke({
                    "current_text": simplified,
                    "fkgl_score": f"{fkgl:.1f}",
                })

            if verbose:
                print(f"    [ReadFeedback] {feedback[:80]}...")

            # Retry simplification with feedback (with or without glossary)
            if use_tools and simplify_glossary_retry_chain is not None:
                simplified = simplify_glossary_retry_chain.invoke({
                    "extracted_answer": extraction,
                    "glossary": glossary_text,
                    "previous_simplification": simplified,
                    "feedback": feedback,
                })
            else:
                simplified = simplify_retry_chain.invoke({
                    "extracted_answer": extraction,
                    "previous_simplification": simplified,
                    "feedback": feedback,
                })
            trace["simplification_attempts"].append(simplified)

            if verbose:
                print(f"    [SimplifyRetry-{retry + 1}] {simplified[:100]}...")

    trace["final_answer"] = simplified
    return trace


# ============================================================================
# CORPUS-LEVEL METRICS  (same as evaluate_all.py for final comparison)
# ============================================================================

nli_pipeline_model = None  # lazy-loaded


def _get_nli_pipeline():
    global nli_pipeline_model
    if nli_pipeline_model is None:
        print("  Loading NLI model ...")
        nli_pipeline_model = hf_pipeline(
            "text-classification",
            model="microsoft/deberta-large-mnli",
            device=-1,
        )
    return nli_pipeline_model


def corpus_sari(preds, refs, originals):
    if not preds:
        return 0.0
    scores = [sample_sari(p, r, o) for p, r, o in zip(preds, refs, originals)]
    return sum(scores) / len(scores)


def corpus_fkgl(preds):
    if not preds:
        return 0.0
    return sum(sample_fkgl(p) for p in preds) / len(preds)


def corpus_nli(preds, refs):
    nli_pipe = _get_nli_pipeline()
    scores = []
    for ref, pred in zip(refs, preds):
        result = nli_pipe(f"{ref} [SEP] {pred}", truncation=True, max_length=512)
        label = result[0]["label"].upper()
        if label == "ENTAILMENT":
            scores.append(1.0)
        elif label == "NEUTRAL":
            scores.append(0.5)
        else:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def corpus_sbert(preds, refs):
    emb_p = sbert_model.encode(preds, convert_to_numpy=True)
    emb_r = sbert_model.encode(refs, convert_to_numpy=True)
    cos = [float(cosine_similarity([a], [b])[0, 0]) for a, b in zip(emb_p, emb_r)]
    return sum(cos) / len(cos) if cos else 0.0


def corpus_token_f1(preds, refs):
    scores = [sample_token_f1(p, r) for p, r in zip(preds, refs)]
    return sum(scores) / len(scores) if scores else 0.0


def corpus_bleu(preds, refs):
    bleu = BLEU()
    return bleu.corpus_score(preds, [[r for r in refs]]).score / 100.0


def corpus_rouge_l(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    return sum(scores) / len(scores) if scores else 0.0


def corpus_bert_score(preds, refs):
    try:
        P, R, F1 = bert_score_fn(
            preds, refs, lang="en", model_type="distilbert-base-uncased", device="cpu"
        )
        return F1.mean().item()
    except Exception as e:
        print(f"  BERTScore error: {e}")
        return 0.0


def corpus_llm_judge(preds, refs, client):
    """GPT-4 LLM-as-judge semantic similarity (Likert 1-5)."""
    scores = []
    for i, (pred, ref) in enumerate(zip(preds, refs)):
        if not pred or not ref:
            scores.append(1.0)
            continue
        try:
            judge_prompt = f"""You are an expert medical QA judge. Rate the semantic similarity of the predicted answer and the reference answer.

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
                messages=[{"role": "user", "content": judge_prompt}],
            )
            result = json.loads(response.choices[0].message.content.strip())
            score = result.get("similarity", 1)
            scores.append(float(max(1, min(5, score))))
        except Exception as e:
            print(f"    LLM judge error on example {i}: {e}")
            scores.append(1.0)

        if (i + 1) % 10 == 0:
            print(f"    LLM judge progress: {i+1}/{len(preds)} (avg: {sum(scores)/len(scores):.4f})")

    return scores


def compute_all_metrics(preds, refs, originals, openai_client=None):
    """Compute all corpus-level metrics, optionally including LLM judge."""
    print("  Computing SARI ...")
    sari = corpus_sari(preds, refs, originals)
    print("  Computing FKGL ...")
    fkgl = corpus_fkgl(preds)
    print("  Computing NLI ...")
    nli = corpus_nli(preds, refs)
    print("  Computing SBERT ...")
    sbert = corpus_sbert(preds, refs)
    print("  Computing Token F1 ...")
    tf1 = corpus_token_f1(preds, refs)
    print("  Computing BLEU ...")
    bleu = corpus_bleu(preds, refs)
    print("  Computing ROUGE-L ...")
    rouge = corpus_rouge_l(preds, refs)
    print("  Computing BERTScore ...")
    bert = corpus_bert_score(preds, refs)

    word_counts = [len(re.findall(r"\w+", p)) for p in preds]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0.0

    metrics = {
        "sari_score": sari,
        "fkgl_score": fkgl,
        "nli_score": nli,
        "sbert_cos_mean": sbert,
        "token_f1": tf1,
        "bleu_score": bleu,
        "rouge_l_score": rouge,
        "bert_score": bert,
        "avg_output_words": avg_words,
    }

    per_sample_judge = None
    if openai_client:
        print("  Computing LLM Judge ...")
        per_sample_judge = corpus_llm_judge(preds, refs, openai_client)
        metrics["llm_judge_score"] = sum(per_sample_judge) / len(per_sample_judge) if per_sample_judge else 0.0

    return metrics, per_sample_judge


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def _ablation_suffix(no_extraction, no_simplification, no_extraction_validation,
                     no_readability_validation, dummy_feedback, max_retries):
    """Build a short filename suffix describing active ablation flags."""
    parts = []
    if no_extraction:
        parts.append("noext")
    if no_simplification:
        parts.append("nosimp")
    if no_extraction_validation:
        parts.append("noaccval")
    if no_readability_validation:
        parts.append("noreadval")
    if dummy_feedback:
        parts.append("dummyfb")
    if max_retries != MAX_RETRIES:
        parts.append(f"retry{max_retries}")
    return ("_abl_" + "_".join(parts)) if parts else ""


def run_pipeline(data_path: str, output_dir: str, split: str = "both",
                 n_dev: int = 20, n_test: int = 200, verbose: bool = False,
                 use_tools: bool = False, glossary_path: str = "data/medlineplus_glossary.json",
                 use_llm_judge: bool = False, resume_from: str = None,
                 gate_mode: str = "sbert",
                 # ── Ablation flags ──────────────────────────────────────────
                 no_extraction: bool = False,
                 no_simplification: bool = False,
                 no_extraction_validation: bool = False,
                 no_readability_validation: bool = False,
                 max_retries: int = MAX_RETRIES,
                 dummy_feedback: bool = False):
    """
    Run the full lightweight dual-model pipeline.

    Args:
        data_path: Path to MeDiSumQA.jsonl
        output_dir: Where to save results
        split: "dev", "test", or "both"
        n_dev: Number of dev samples
        n_test: Number of held-out test samples
        verbose: Print per-sample progress
        use_tools: If True, augment simplifier with MedlinePlus glossary
        glossary_path: Path to pre-downloaded glossary JSON
        use_llm_judge: If True, run GPT-4 LLM-as-judge scoring
        resume_from: Path to checkpoint JSON to resume from
        no_extraction: Ablation — skip Stage 1, pass raw DS to simplifier
        no_simplification: Ablation — skip Stage 3, use extraction as final answer
        no_extraction_validation: Ablation — skip Stage 2 accuracy gate
        no_readability_validation: Ablation — skip Stage 4 readability gate
        max_retries: Ablation — override retry count per gate
        dummy_feedback: Ablation — replace Model B feedback with fixed strings
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────────────────
    dev_data, test_data = load_and_split(data_path, n_dev, n_test)

    # ── Initialize OpenAI client for LLM judge ──────────────────────────────
    openai_client = None
    if use_llm_judge:
        if OPENAI_API_KEY:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            print("OpenAI client initialized for LLM-as-judge scoring.")
        else:
            print("WARNING: --llm-judge requested but OPENAI_API_KEY not set. Skipping judge.")

    # ── Derive pipeline name (used for checkpoints + output filenames) ────────
    if gate_mode not in ("sbert", "llm"):
        raise ValueError(f"Unknown gate_mode={gate_mode!r}; expected 'sbert' or 'llm'")
    abl_sfx = _ablation_suffix(no_extraction, no_simplification,
                                no_extraction_validation, no_readability_validation,
                                dummy_feedback, max_retries)
    base_name = "lightweight_pipeline_tools" if use_tools else "lightweight_pipeline"
    # Suffix filenames with the gate mode so LLM-gate runs don't collide with
    # SBERT-gate runs in the same output directory.
    gate_sfx = "" if gate_mode == "sbert" else "_llmgate"
    pipeline_name = base_name + gate_sfx + abl_sfx

    # ── Load models ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading models")
    print("=" * 60)

    tok_a, model_a = load_model(MODEL_A_ID)

    # Model B is only needed when validation gates are active (feedback loops)
    # or when tool-augmented mode uses it to extract medical terms.
    # In gate_mode='llm' Model B IS the classifier, so it's required even when
    # the readability gate is off (unless the extraction gate is also off).
    need_model_b_for_extval = (not no_extraction_validation) and (
        gate_mode == "llm" or not dummy_feedback
    )
    need_model_b_for_readval = (not no_readability_validation) and not dummy_feedback
    need_model_b = need_model_b_for_extval or need_model_b_for_readval or use_tools
    tok_b, model_b = (load_model(MODEL_B_ID)
                      if need_model_b else (None, None))
    if not need_model_b:
        print(f"  Model B ({MODEL_B_ID}) not loaded — not needed for this ablation.")

    # ── Load glossary if tool-augmented mode ─────────────────────────────────
    if use_tools:
        print(f"\nTool-augmented mode enabled — loading glossary from {glossary_path}")
        load_glossary(glossary_path)

    # ── Build LangChain chains ──────────────────────────────────────────────
    print("\nBuilding LangChain chains ...")

    # Model A chains (extractor + simplifier)
    llm_a_extract = LangChainLocalLLM(tok_a, model_a, system_prompt=EXTRACT_SYSTEM)
    llm_a_simplify = LangChainLocalLLM(tok_a, model_a, system_prompt=SIMPLIFY_SYSTEM)

    extractor_chain = build_chain(EXTRACT_PROMPT, llm_a_extract)
    extract_retry_chain = build_chain(EXTRACT_RETRY_PROMPT, llm_a_extract)
    simplifier_chain = build_chain(SIMPLIFY_PROMPT, llm_a_simplify)
    simplify_retry_chain = build_chain(SIMPLIFY_RETRY_PROMPT, llm_a_simplify)

    # Tool-augmented simplifier chains (glossary-aware prompts)
    simplifier_glossary_chain = None
    simplify_glossary_retry_chain = None
    if use_tools:
        llm_a_simplify_glossary = LangChainLocalLLM(
            tok_a, model_a, system_prompt=SIMPLIFY_WITH_GLOSSARY_SYSTEM
        )
        simplifier_glossary_chain = build_chain(
            SIMPLIFY_WITH_GLOSSARY_PROMPT, llm_a_simplify_glossary
        )
        simplify_glossary_retry_chain = build_chain(
            SIMPLIFY_WITH_GLOSSARY_RETRY_PROMPT, llm_a_simplify_glossary
        )

    # Model B chains (validator feedback / classifier) — only built when
    # Model B is loaded.
    feedback_extraction_chain = None
    feedback_readability_chain = None
    classifier_chain = None
    if need_model_b:
        # Feedback chain only needed for the SBERT gate (the LLM gate
        # produces its own feedback inline as part of the classifier output).
        if gate_mode == "sbert" and not no_extraction_validation:
            llm_b_ext_fb = LangChainLocalLLM(
                tok_b, model_b,
                system_prompt=EXTRACTION_FEEDBACK_SYSTEM, max_new_tokens=150,
            )
            feedback_extraction_chain = build_chain(EXTRACTION_FEEDBACK_PROMPT, llm_b_ext_fb)

        if gate_mode == "llm" and not no_extraction_validation:
            llm_b_classify = LangChainLocalLLM(
                tok_b, model_b,
                system_prompt=EXTRACTION_CLASSIFIER_SYSTEM, max_new_tokens=200,
            )
            classifier_chain = build_chain(EXTRACTION_CLASSIFIER_PROMPT, llm_b_classify)

        if not no_readability_validation:
            llm_b_read_fb = LangChainLocalLLM(
                tok_b, model_b,
                system_prompt=READABILITY_FEEDBACK_SYSTEM, max_new_tokens=150,
            )
            feedback_readability_chain = build_chain(READABILITY_FEEDBACK_PROMPT, llm_b_read_fb)

    print("Chains ready.\n")

    # ── Determine which splits to run ───────────────────────────────────────
    splits_to_run = {}
    if split in ("dev", "both"):
        splits_to_run["dev"] = dev_data
    if split in ("test", "both"):
        splits_to_run["test"] = test_data

    all_results = {}

    # ── Load checkpoint if resuming ────────────────────────────────────────
    checkpoint = {}
    if resume_from and os.path.exists(resume_from):
        with open(resume_from, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        print(f"Loaded checkpoint from {resume_from}")

    checkpoint_path = os.path.join(output_dir, f"{pipeline_name}_checkpoint.json")

    for split_name, split_data in splits_to_run.items():
        print(f"\n{'=' * 60}")
        print(f"Running pipeline on {split_name.upper()} split ({len(split_data)} samples)")
        print(f"{'=' * 60}")

        # Resume from checkpoint if available for this split
        traces = []
        inference_times = []
        start_idx = 0
        if split_name in checkpoint:
            traces = checkpoint[split_name].get("traces", [])
            inference_times = [t.get("inference_time_s", 0) for t in traces]
            start_idx = len(traces)
            print(f"  Resuming from sample {start_idx + 1}/{len(split_data)} "
                  f"({start_idx} already completed)")

        for i, example in enumerate(split_data):
            if i < start_idx:
                continue

            print(f"\n  [{i + 1}/{len(split_data)}] "
                  f"Q: {example.get('Question', '')[:60]}...")

            t0 = time.time()
            try:
                trace = run_sample_pipeline(
                    example,
                    extractor_chain,
                    simplifier_chain,
                    extract_retry_chain,
                    simplify_retry_chain,
                    feedback_extraction_chain,
                    feedback_readability_chain,
                    verbose=verbose,
                    use_tools=use_tools,
                    simplifier_glossary_chain=simplifier_glossary_chain,
                    simplify_glossary_retry_chain=simplify_glossary_retry_chain,
                    tok_b=tok_b,
                    model_b=model_b,
                    classifier_chain=classifier_chain,
                    gate_mode=gate_mode,
                    no_extraction=no_extraction,
                    no_simplification=no_simplification,
                    no_extraction_validation=no_extraction_validation,
                    no_readability_validation=no_readability_validation,
                    max_retries=max_retries,
                    dummy_feedback=dummy_feedback,
                )
            except Exception as e:
                print(f"    ERROR: {e}")
                trace = {
                    "question": example.get("Question", ""),
                    "reference": example.get("Answer", ""),
                    "original": example.get("discharge_summary", ""),
                    "final_answer": "",
                    "error": str(e),
                }
            elapsed = time.time() - t0
            inference_times.append(elapsed)

            trace["inference_time_s"] = elapsed
            traces.append(trace)

            print(f"    Final: {trace.get('final_answer', '')[:100]}...")
            print(f"    Time: {elapsed:.1f}s")

            torch.cuda.empty_cache()

            # ── Save checkpoint every 5 samples ────────────────────────────
            if (i + 1) % 5 == 0 or (i + 1) == len(split_data):
                checkpoint[split_name] = {"traces": traces}
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(checkpoint, f, ensure_ascii=False)
                print(f"    [Checkpoint saved: {i + 1}/{len(split_data)}]")

        # ── Compute corpus metrics ──────────────────────────────────────────
        preds = [t["final_answer"] for t in traces]
        refs = [t["reference"] for t in traces]
        originals = [t["original"] for t in traces]

        print(f"\nComputing corpus metrics for {split_name} ...")
        metrics, judge_scores = compute_all_metrics(preds, refs, originals, openai_client)
        if judge_scores:
            for t, s in zip(traces, judge_scores):
                t["llm_judge_score"] = s
        metrics["avg_inference_time_s"] = (
            sum(inference_times) / len(inference_times) if inference_times else 0.0
        )
        metrics["total_inference_time_s"] = sum(inference_times)

        # ── Print results ───────────────────────────────────────────────────
        print(f"\n{'─' * 50}")
        print(f"Results for {split_name.upper()} split "
              f"(pipeline: {MODEL_A_ID} + {MODEL_B_ID})")
        print(f"{'─' * 50}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        all_results[split_name] = {
            "metrics": metrics,
            "traces": traces,
            "model_a": MODEL_A_ID,
            "model_b": MODEL_B_ID if need_model_b else None,
            "use_tools": use_tools,
            "gate_mode": gate_mode,
            "n_samples": len(split_data),
            "thresholds": {
                "extraction": EXTRACTION_THRESHOLDS,
                "readability": READABILITY_THRESHOLDS,
            },
            "ablation": {
                "no_extraction": no_extraction,
                "no_simplification": no_simplification,
                "no_extraction_validation": no_extraction_validation,
                "no_readability_validation": no_readability_validation,
                "max_retries": max_retries,
                "dummy_feedback": dummy_feedback,
            },
        }

    # ── Save results ────────────────────────────────────────────────────────

    # Save full traces (predictions + per-sample data)
    traces_path = os.path.join(output_dir, f"{pipeline_name}_traces_{timestamp}.json")
    # Convert traces for JSON serialization
    save_data = {}
    for split_name, result in all_results.items():
        save_data[split_name] = {
            "metrics": result["metrics"],
            "model_a": result["model_a"],
            "model_b": result["model_b"],
            "gate_mode": result.get("gate_mode", "sbert"),
            "n_samples": result["n_samples"],
            "thresholds": result["thresholds"],
            "ablation": result["ablation"],
            "predictions": [
                {
                    "index": i,
                    "note_id": t.get("note_id", ""),
                    "question": t["question"],
                    "reference": t["reference"],
                    "prediction": t["final_answer"],
                    "original": t["original"],
                    "extraction_attempts": t.get("extraction_attempts", []),
                    "simplification_attempts": t.get("simplification_attempts", []),
                    "inference_time_s": t.get("inference_time_s", 0),
                    **({"llm_judge_score": t["llm_judge_score"]} if "llm_judge_score" in t else {}),
                }
                for i, t in enumerate(result["traces"])
            ],
        }
    with open(traces_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nFull traces saved to {traces_path}")

    # Save metrics summary CSV
    metrics_csv_path = os.path.join(output_dir, f"{pipeline_name}_metrics_{timestamp}.csv")
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        metric_names = list(next(iter(all_results.values()))["metrics"].keys())
        writer.writerow(["Split", "Model_A", "Model_B"] + metric_names)
        for split_name, result in all_results.items():
            row = [split_name, result["model_a"], result["model_b"]]
            row += [f"{result['metrics'].get(m, 0):.4f}" for m in metric_names]
            writer.writerow(row)
    print(f"Metrics CSV saved to {metrics_csv_path}")

    # ── Print comparison table ──────────────────────────────────────────────
    tool_label = " + MedlinePlus glossary" if use_tools else ""
    abl_label = f" [{abl_sfx.lstrip('_')}]" if abl_sfx else ""
    print(f"\n{'=' * 80}")
    print(f"FINAL COMPARISON — Lightweight Pipeline{tool_label}{abl_label}")
    print(f"  Model A (Extractor/Simplifier): {MODEL_A_ID}")
    if need_model_b:
        print(f"  Model B (Validator/Feedback):    {MODEL_B_ID}")
    if use_tools:
        print(f"  Tool augmentation:               MedlinePlus offline glossary")
    if abl_sfx:
        print(f"  Ablation:                        {abl_sfx.lstrip('_abl_')}")
    print(f"{'=' * 80}")

    for split_name, result in all_results.items():
        print(f"\n  {split_name.upper()} ({result['n_samples']} samples):")
        for k, v in result["metrics"].items():
            print(f"    {k:<25} {v:.4f}")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    to_delete = [model_a, tok_a]
    if need_model_b:
        to_delete += [model_b, tok_b]
    for obj in to_delete:
        del obj
    torch.cuda.empty_cache()
    print("\nModels unloaded, GPU memory freed.")

    return all_results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lightweight Dual-Model Pipeline Evaluation"
    )
    parser.add_argument(
        "--data", default="data/MeDiSumQA.jsonl", help="Input JSONL file"
    )
    parser.add_argument(
        "--output", default="results", help="Output directory"
    )
    parser.add_argument(
        "--split", default="both", choices=["dev", "test", "both"],
        help="Which data split to evaluate"
    )
    parser.add_argument(
        "--n-dev", type=int, default=20, help="Number of dev samples"
    )
    parser.add_argument(
        "--n-test", type=int, default=200, help="Number of test samples"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-sample details"
    )
    parser.add_argument(
        "--use-tools", action="store_true",
        help="Enable MedlinePlus glossary tool for simplifier"
    )
    parser.add_argument(
        "--glossary", default="data/medlineplus_glossary.json",
        help="Path to pre-downloaded MedlinePlus glossary JSON"
    )
    parser.add_argument(
        "--llm-judge", action="store_true",
        help="Enable GPT-4 LLM-as-judge scoring (requires OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to checkpoint JSON to resume from"
    )
    parser.add_argument(
        "--gate-mode", default="sbert", choices=["sbert", "llm"],
        help="Extraction-validation gate signal: 'sbert' = SBERT(extraction, "
             "question) > tau (default; tau tuned on dev). 'llm' = Model B "
             "classifies extraction and returns inline feedback."
    )
    # ── Ablation flags ────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-extraction", action="store_true",
        help="Ablation: skip Stage 1 — pass raw discharge summary to simplifier"
    )
    parser.add_argument(
        "--no-simplification", action="store_true",
        help="Ablation: skip Stage 3 — use extraction output as final answer"
    )
    parser.add_argument(
        "--no-extraction-validation", action="store_true",
        help="Ablation: skip Stage 2 accuracy gate (no extraction retry loop)"
    )
    parser.add_argument(
        "--no-readability-validation", action="store_true",
        help="Ablation: skip Stage 4 readability gate (no simplification retry loop)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=MAX_RETRIES,
        help=f"Ablation: max retries per validation gate (default: {MAX_RETRIES})"
    )
    parser.add_argument(
        "--dummy-feedback", action="store_true",
        help="Ablation: replace Model B feedback with fixed generic strings"
    )

    args = parser.parse_args()

    run_pipeline(
        data_path=args.data,
        output_dir=args.output,
        split=args.split,
        n_dev=args.n_dev,
        n_test=args.n_test,
        verbose=args.verbose,
        use_tools=args.use_tools,
        glossary_path=args.glossary,
        use_llm_judge=args.llm_judge,
        resume_from=args.resume,
        gate_mode=args.gate_mode,
        no_extraction=args.no_extraction,
        no_simplification=args.no_simplification,
        no_extraction_validation=args.no_extraction_validation,
        no_readability_validation=args.no_readability_validation,
        max_retries=args.max_retries,
        dummy_feedback=args.dummy_feedback,
    )
