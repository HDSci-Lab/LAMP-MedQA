# Data

This directory holds the inputs the pipeline expects at runtime.

## `medlineplus_glossary.json` (included)

Offline glossary of 1,926 MedlinePlus health-topic entries, built from the
public MedlinePlus Health Topics XML feed. Used by the tool-augmented
variant of the simplifier. To rebuild from source:

```bash
python ../download_medlineplus_glossary.py
```

## `MeDiSumQA.jsonl` (you must obtain this)

The MeDiSumQA dataset is derived from MIMIC-IV discharge summaries and is
distributed under the PhysioNet Credentialed Health Data Use Agreement. It
is **not redistributable** and is therefore not included in this repository.

To obtain it:

1. Become a credentialed PhysioNet user
   (<https://physionet.org/settings/credentialing/>) and complete the
   required CITI Data or Specimens Only Research training.
2. Sign the MIMIC-IV data use agreement.
3. Download MeDiSumQA from PhysioNet
   (Dada et al., 2025; <https://physionet.org/content/medisumqa/>) and
   place the file at `data/MeDiSumQA.jsonl`.

The expected format is JSON Lines, one record per line, with at minimum a
patient question, the source discharge summary, and the
clinician-verified reference answer. The exact field names used by the
loaders are visible in `evaluate_lightweight_pipeline.py` and
`evaluate_200.py` (search for `MeDiSumQA` in the `load_data` /
`prepare_splits` functions).

The pipeline reproduces the paper's splits with `seed=42`: 20 reports for
the development set and 200 for the held-out test set, with the one-shot
exemplar excluded from both.

Before running any of the OpenAI-dependent steps (GPT-5 baseline, GPT-4
LLM-as-a-Judge), read the **Compliance** section in the [top-level
README](../README.md#compliance-using-mimic-data-with-third-party-llms).
Sending MeDiSumQA text through a default OpenAI API key does not satisfy
the PhysioNet Data Use Agreement.
