# Business Entity Resolution Pipeline

## Overview

Matches business records across 3 noisy data sources for the Amazon ML Challenge.
Source 1 is the deduplicated reference; the pipeline finds matching records from
Sources 2 and 3.

**Architecture**: Union blocking (token + TF-IDF char n-gram) → LightGBM classifier
over ~30 hand-engineered similarity features → F₀.₅-tuned threshold.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Place Data

Place the competition data files in the project root:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

### 3. Train + Validate (on a held-out split)

```bash
python -m code.business_entity_resolution.src.pipeline --mode train
```

This will:
- Split training data 80/20 (S1 entity level split)
- Run blocking on both splits (reports recall ceiling)
- Compute pairwise features
- Train LightGBM + sweep thresholds against F₀.₅
- Report detailed validation metrics

### 4. Generate Test Submission

```bash
python -m code.business_entity_resolution.src.pipeline --mode both
```

This will:
- Retrain on full training data (with a small 10% holdout for threshold tuning)
- Run blocking + features + inference on test data
- Write `output/matching_results.tsv` and `output/candidate_pairs.tsv`
- Run validation checks

### 5. Just Inference (with pre-trained model)

```bash
python -m code.business_entity_resolution.src.pipeline --mode test
```

## Pipeline Architecture

```
Data Loading → Normalization → Blocking → Features → Matching → Output
                   │                │         │          │
                   ▼                ▼         ▼          ▼
          Legal suffix dict   Token+TF-IDF  ~30 sim   LightGBM
          Address abbrev dict   union       features   + threshold
          Unicode/case fold                             tuning (F₀.₅)
```

## Module Structure

| Module | Purpose |
|--------|---------|
| `src/normalization.py` | Legal suffix + address abbreviation dictionaries, Unicode normalization, tokenization |
| `src/blocking.py` | Token-based and TF-IDF character n-gram blocking with union strategy |
| `src/features.py` | ~30 pairwise similarity features (string, token, character n-gram metrics) |
| `src/matching.py` | LightGBM training, inference, threshold tuning, LLM tiebreaker placeholder |
| `src/scoring.py` | F₀.₅ implementation (per-entity + macro-average), threshold sweep |
| `src/inference.py` | Output file generation + validation checks |
| `src/pipeline.py` | Main pipeline orchestrator + CLI entry point |

## Key CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `train` | `train`, `test`, or `both` |
| `--val-fraction` | `0.2` | Validation split fraction |
| `--top-k` | `30` | Top-K for TF-IDF blocking |
| `--max-block-size` | `500` | Max block size for token blocking |
| `--log-file` | None | Optional log file path |

## Design Decisions

- **No hardcoded country values**: Pipeline treats country as an open set of string labels.
  Works for US, India, France, and any other country in future data.
- **No external API/database lookups**: All processing is local, using only the provided data.
- **Precision-biased**: F₀.₅ weights precision 2× over recall. The threshold is explicitly
  tuned against this metric. Default behavior for singletons (no match) is preserved.
- **LLM tiebreaker ready**: `matching.py` contains a placeholder for hybrid LLM mode.
  When a local LLM model is available, it can be enabled to resolve ambiguous pairs.
