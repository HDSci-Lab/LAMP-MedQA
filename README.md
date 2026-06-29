# LAMP-MedQA

Code for the paper **"LAMP-MedQA: A Lightweight Multi-Agent System for
Patient-Oriented Medical Question Answering"** (Johnson, Banerjee, Crawford,
Welch, Davies, and Wang).

LAMP-MedQA decomposes patient-facing medical question answering into two
metric-gated stages, evidence extraction from a discharge summary, then
patient-friendly simplification, with iterative self-correction. The
generation agents use Qwen2.5-7B-Instruct and the reviewer/verifier agents
use Phi-3.5-Mini-Instruct.

Full paper link: https://aclanthology.org/2026.acl-srw.60/

## Setup

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

Tested with Python 3.11. A CUDA GPU is required for the pipeline (an H100 was
used in the paper); the OpenAI API is required for baselines and
LLM-as-a-Judge scoring.

Set the credentials the scripts expect:

```bash
export HF_TOKEN=...            # Hugging Face read token (gated Qwen weights)
export OPENAI_API_KEY=...      # GPT-5 baseline + LLM-Judge
```

## Data

The MeDiSumQA dataset is derived from MIMIC-IV discharge summaries and is
**not redistributable**. Access requires PhysioNet credentialing. See
[data/README.md](data/README.md) for the access path and expected file
layout.

The MedlinePlus glossary used by the tool-augmented variant is included at
`data/medlineplus_glossary.json` (built from public MedlinePlus XML). To
rebuild it from source:

```bash
python download_medlineplus_glossary.py
```

## Compliance: using MIMIC data with third-party LLMs

Several steps in this code base send text derived from MeDiSumQA (and
therefore from MIMIC-IV discharge summaries) to third-party LLM services:

- the GPT-5 baseline in `evaluate_200.py`
- the GPT-4 LLM-as-a-Judge scoring (`--llm-judge`, `llm_judge.py`,
  `llm_judge_ablations.py`, `rerun_llm_judge*.py`)

The PhysioNet Credentialed Data Use Agreement
([policy notice, 2025-09-24](https://physionet.org/news/post/gpt-responsible-use))
**prohibits sharing access to MIMIC data with third parties, including via
APIs or online platforms**, unless zero data retention is verifiably in
place. Key requirements as stated by PhysioNet:

- **Zero data retention.** MIMIC data must not be stored or retained by
  third-party LLM services.
- **No training use.** The data must not be used to train or improve any
  model.
- **No human review.** No human at the service may inspect the data.

The numbers in the paper were produced in a secure research workspace with
contractually verified zero data retention. **A default OpenAI / Anthropic
/ Google consumer API key does not meet these requirements.** If you intend
to run the OpenAI-dependent steps on real MeDiSumQA data, you must either:

1. Use a zero-data-retention enterprise endpoint that you have verified
   contractually (e.g. Azure OpenAI with ZDR enabled, an OpenAI Enterprise
   ZDR agreement, or your institution's secure data environment); or
2. Replace those steps with a locally deployed LLM. Skip `--llm-judge`
   and drop the GPT baselines if you cannot meet the requirement.

PhysioNet cannot verify the data practices of external services. The
researcher running this code is responsible for compliance.

Optional belt-and-braces: `transformers` / `huggingface_hub` send an
anonymous user-agent string on model-weight downloads (library version and
OS only — no MeDiSumQA content). To suppress it entirely, set
`HF_HUB_DISABLE_TELEMETRY=1` in the environment that runs the pipeline.

## Reproducing the paper

See [REPRODUCE.md](REPRODUCE.md) for the full ordered command list (threshold
tuning, baselines, single-model ablations, LAMP-MedQA, pipeline ablations,
and significance tests).

The pipeline uses a reference-free extraction gate: SBERT cosine similarity
between the candidate extraction and the patient question, with τ tuned on
the 20-sample dev split. The readability gate is FKGL < 10. Both loops allow
up to three revision attempts.

## Repository layout

```
evaluate_lightweight_pipeline.py     LAMP-MedQA pipeline (extractor, reviewer,
                                     simplifier, readability verifier)
evaluate_200.py                      GPT-5 / Qwen-32B zero-/one-shot baselines
                                     and single-model ablation baselines
tune_extraction_threshold.py         Dev-set τ tuner (balanced-accuracy
                                     maximisation)
statistical_tests.py                 Paired bootstrap + Wilcoxon for the main
statistical_tests_ablations.py       results and ablations
llm_judge.py                         GPT-4 LLM-as-a-Judge scoring
llm_judge_ablations.py
rerun_llm_judge.py                   Re-score existing prediction files
rerun_llm_judge_baselines.py
recalculate_metrics.py               Recompute SARI/FKGL/SBERT/etc. on a
                                     traces JSON without rerunning inference
download_medlineplus_glossary.py     Build data/medlineplus_glossary.json
data/medlineplus_glossary.json       Offline glossary (1,926 entries)
figures/pipeline_schematic.tex       TikZ source for the pipeline figure
REPRODUCE.md                         Ordered list of commands to reproduce
                                     the paper's numbers
```

## Citation

```bibtex
@inproceedings{johnson2026lampmedqa,
  title     = {LAMP-MedQA: A Lightweight Multi-Agent System for
               Patient-Oriented Medical Question Answering},
  author    = {Johnson, Jack and Banerjee, Meghali and Crawford, Joseph and
               Welch, James and Davies, Jim and Wang, Tingyan},
  year      = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
