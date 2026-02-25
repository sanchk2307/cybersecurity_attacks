# README Proposed Changes Report

**Date:** 2026-02-25
**Scope:** `docs/README.md` — full rewrite proposal

---

## Executive Summary

The current README is significantly outdated and incomplete. It references a stale directory
structure (e.g., `src/config.py` instead of `src/utilities/config.py`), missing files
(`randomforrest_model.pkl`, `data/.cache/`, `src/utilities/`), and contains an empty
"Description" section. The EDA section describes an old class-based approach that is no longer
the primary workflow. Several critical features — the Streamlit app, Random Forest model,
parallel processing, caching system, and feature engineering methodology — are entirely
undocumented.

This report lists every proposed change with rationale.

---

## 1. Title — Fix Typo

**Current:**
```
# Cybersecurity Attacks ML Analizer
```

**Proposed:**
```
# Cybersecurity Attacks ML Analyzer
```

**Rationale:** "Analizer" → "Analyzer" (spelling fix).

---

## 2. Add Project Description

**Current:** Empty `## Description` section.

**Proposed:**
```markdown
## Description

A machine learning classification system for predicting cyber attack types (DDoS,
Intrusion, Malware) from network traffic data. The project includes:

- **Exploratory Data Analysis** — IP geolocation, device fingerprinting, temporal
  decomposition, and statistical testing (MCC, Chi-Square)
- **Feature Engineering** — Bias-based target encoding with dynamic thresholds,
  continuous variable binning, and cross-tabulation features
- **ML Models** — Logistic Regression baseline and hyperparameter-tuned Random Forest
  (RandomizedSearchCV, 30 iterations, 3-fold stratified CV, F1-macro)
- **Interactive Web App** — Streamlit interface with CSV upload, row selection, and
  manual input modes for real-time attack prediction
- **Parallel Processing** — ThreadPoolExecutor for I/O-bound tasks (IP geolocation),
  ProcessPoolExecutor for CPU-bound tasks (UA parsing), with disk caching

### Dataset

- **Source:** `data/cybersecurity_attacks.csv`
- **Size:** 40,000 rows × 25 columns
- **Target:** `Attack Type` — 3 balanced classes (DDoS, Intrusion, Malware)
- **Features:** Timestamps, IP addresses, ports, protocols, packet metadata, payload
  data, anomaly scores, device user-agents, geolocation, firewall/IDS logs
```

**Rationale:** The project has no description at all. Users arriving at the repo have zero
context about what this project does, what models are used, or what the dataset looks like.

---

## 3. Update Directory Tree

**Current:** Shows flat `src/` structure, missing `src/utilities/` subdirectory, missing
`randomforrest_model.pkl`, missing `data/.cache/`, references non-existent `df.csv`.

**Proposed:**
```markdown
## Repo Organization

```
cybersecurity_attacks/
├── pipeline.py                    # Main EDA + training orchestrator
├── app.py                         # Streamlit web application
├── helpers.py                     # Utility functions for the Streamlit app
├── cybersecurity_eda.ipynb        # Jupyter notebook for EDA exploration
├── pixi.toml                      # Pixi package manager config
├── pixi.lock                      # Locked dependency versions
├── LICENSE
├── data/
│   ├── cybersecurity_attacks.csv  # Original dataset (40k rows × 25 cols)
│   ├── df.parquet                 # Processed data after EDA
│   ├── pre_model_df.parquet       # Pre-model checkpoint
│   ├── pre_model_crosstabs.pkl    # Crosstab cache
│   └── .cache/                    # Disk cache for IP geolocation & UA parsing
├── models/
│   ├── logit_model.pkl            # Trained Logistic Regression model
│   └── randomforrest_model.pkl    # Tuned Random Forest model
├── src/
│   ├── __init__.py
│   ├── eda_pipeline.py            # EDA pipeline orchestration
│   ├── modelling.py               # Model training & evaluation
│   ├── EDA.py                     # Legacy EDA class
│   ├── ports_pipeline.py          # Port analysis exploration
│   └── utilities/
│       ├── __init__.py
│       ├── config.py              # Configuration & data loading
│       ├── data_preparation.py    # IP geolocation, device parsing (parallel)
│       ├── feature_engineering.py # Binning, crosstabs, bias scoring
│       ├── visualization.py       # Geographic plots, strip plots
│       ├── statistical_analysis.py# MCC correlation & Chi-Square tests
│       ├── diagrams.py            # Sankey & parallel categories
│       ├── utils.py               # Categorical encoding helpers
│       ├── download_files.py      # External data download (stub)
│       ├── payload_analyzer.py    # Payload feature extraction
│       └── time_series.py         # Time-series analysis utilities
├── geolite2_db/
│   ├── GeoLite2-ASN.mmdb
│   ├── GeoLite2-Country.mmdb
│   └── readme.txt
└── docs/
    ├── README.md
    └── ML Python Labs Group Work Distribution.docx
```
```

**Rationale:** The tree must reflect the actual codebase. Key changes:
- `src/` modules moved into `src/utilities/` subdirectory
- `df.csv` → `df.parquet` (format changed)
- Added `randomforrest_model.pkl` (tuned RF model)
- Added `data/.cache/` directory
- Added `helpers.py` at root
- Added inline comments explaining each file's purpose

---

## 4. Replace EDA Section with Pipeline Documentation

**Current:** Describes an old class-based EDA approach with GeoNames city-state matching
(attributes like `india_df`, `admin_df`, `max_city_population`). This class (`EDA.py`) is
now legacy and not used in the main pipeline.

**Proposed:**
```markdown
## Pipeline

The main pipeline (`pipeline.py`) orchestrates the full workflow:

1. **Data Loading** — Reads raw CSV or loads cached parquet checkpoint
2. **EDA** (`src/eda_pipeline.py`):
   - Column renaming and cleaning (drops User Information, Payload Data)
   - Date feature decomposition (day, month-week, weekday, hour, minute buckets)
   - IP geolocation via GeoIP2 (parallel, cached to disk)
   - Device User-Agent parsing (parallel, cached to disk)
   - Categorical encoding and binary indicators
   - Cross-tabulations against Attack Type
3. **Visualizations** (parallel):
   - Geographic attack distribution maps (6-panel)
   - Sankey diagram (country-to-country IP flows)
   - Parallel categories diagram
   - MCC correlation analysis and Chi-Square tests
4. **Feature Engineering** (`src/utilities/feature_engineering.py`):
   - Continuous variable binning (ports, anomaly scores, packet length)
   - Bias-based target encoding with dynamic thresholds
   - Bias scoring on a 1–6 scale
5. **Model Training** (`src/modelling.py`):
   - Train/test split (90/10, stratified)
   - Logistic Regression (`lbfgs`, `max_iter=1000`)
   - Random Forest with `RandomizedSearchCV` (30 iterations, 3-fold CV, F1-macro)
6. **Evaluation** — Confusion matrix, ROC curves, precision-recall curves,
   mutual information feature importance
```

**Rationale:** The old EDA description is misleading. The actual pipeline is much more
sophisticated and should be documented accurately.

---

## 5. Add Quick Start / Installation Section

**Current:** No installation or setup instructions.

**Proposed:**
```markdown
## Getting Started

### Prerequisites

- [Pixi](https://pixi.sh/) package manager
- MaxMind GeoLite2 databases (included in `geolite2_db/`)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd cybersecurity_attacks

# Install dependencies via Pixi
pixi install
```

### Running the Pipeline

```bash
# Full pipeline with visualizations
pixi run run-pipeline

# Skip figures (faster, model metrics only)
pixi run run-pipeline --no-figures

# Model metrics only (skip EDA + visualizations)
pixi run run-pipeline --model-only

# Sequential mode (lower RAM usage — runs all stages one at a time)
pixi run run-pipeline --sequential

# Memory limit — auto-tunes concurrency to fit within budget
pixi run run-pipeline --mem-limit 8    # Fully sequential, 1 worker
pixi run run-pipeline --mem-limit 16   # Sequential stages, 2 workers
pixi run run-pipeline --mem-limit 32   # 2 concurrent stages, 4 workers
pixi run run-pipeline --mem-limit 64   # Full parallel, all CPU workers

# Live memory monitor TUI (shows RSS, peak, system usage per stage)
pixi run run-pipeline --monitor

# Flags can be combined
pixi run run-pipeline --mem-limit 16 --no-figures
```

### Running the Web App

```bash
pixi run run-app
```

The Streamlit app supports three prediction modes:
1. **CSV Upload** — Batch prediction from uploaded CSV files
2. **Pick a Row** — Select a row from the test set (shows known label for comparison)
3. **Manual Input** — Fill in all 25 fields manually with an interactive IP geolocation map
```

**Rationale:** A new user cannot run this project without knowing about Pixi, the available
commands, or the CLI flags (`--no-figures`, `--model-only`). The Streamlit app is a major
feature that is completely undocumented.

---

## 6. Add Models Section

**Current:** No mention of models, performance, or methodology.

**Proposed:**
```markdown
## Models

| Model | File | Approach |
|-------|------|----------|
| Logistic Regression | `models/logit_model.pkl` | Baseline classifier (`lbfgs`, `max_iter=1000`) |
| Random Forest | `models/randomforrest_model.pkl` | Tuned via `RandomizedSearchCV` (30 combos, 3-fold stratified CV, F1-macro) |

### Feature Engineering Methodology

The core innovation is **bias-based target encoding**:
1. Cross-tabulations compute the % distribution of each feature value across attack types
2. A **dynamic threshold** function adjusts the bias cutoff based on observation count
   (`threshold = a * nobs + b`, clamped to a floor value)
3. Feature values are assigned a **bias score (1–6)** based on percentage tiers:
   37.5%, 40%, 50%, 60%, 75%, 90%

### Evaluation Metrics

Both models are evaluated with:
- Confusion Matrix
- Multiclass ROC Curves (one-vs-rest, AUC per class)
- Precision-Recall Curves
- Classification Report (Precision, Recall, F1 per class)
- Feature Importance (Mutual Information)
```

**Rationale:** The ML methodology is the intellectual core of this project and deserves
prominent documentation.

---

## 7. Update Available Commands Section

**Current:** Only lists `run-pipeline` and `gen-dir-repr`. Missing `run-app`.

**Proposed:**
```markdown
### Available Commands

```bash
pixi run run-pipeline    # Run the full EDA + training pipeline
pixi run run-app         # Launch the Streamlit web application
pixi run gen-dir-repr    # Generate ASCII directory tree
```
```

**Rationale:** `run-app` is a key command that is missing.

---

## 8. Update Results Section

**Current:**
> So far the EDA gathers the latitude and longitude data for the Geolocation column for the
> 93.5% of the rows cybersecurity dataset.

**Proposed:**
```markdown
## Results

- **IP Geolocation Coverage:** 93.5% of IPs successfully geolocated via GeoIP2
- **Visualizations:** Geographic attack distribution maps, Sankey traffic flow diagrams,
  parallel categories, strip plots, and statistical test outputs
- **Model Outputs:** Trained models persisted in `models/` directory; evaluation plots
  (confusion matrix, ROC, precision-recall) generated during pipeline execution
```

**Rationale:** The results section should reflect the full pipeline output, not just
geolocation coverage.

---

## 9. Add Performance / Caching Section

**Current:** No mention of caching or performance optimizations.

**Proposed:**
```markdown
## Performance

The pipeline uses several optimization strategies:

- **Pre-model caching:** EDA results saved as `pre_model_df.parquet` +
  `pre_model_crosstabs.pkl` — subsequent runs skip EDA entirely
- **Disk caching:** IP geolocation and UA parsing results cached in `data/.cache/`
  (JSON files) — only uncached entries are processed on re-runs
- **Parallel processing:**
  - I/O-bound tasks (IP lookup): `ThreadPoolExecutor`
  - CPU-bound tasks (UA parsing): `ProcessPoolExecutor`
  - Visualization generation: parallel via `ThreadPoolExecutor`
  - Workers: `max(1, cpu_count - 2)`
```

**Rationale:** Significant engineering effort went into performance. Documenting it helps
contributors understand design decisions and avoid regressions.

---

## 10. Update Utilities Section

**Current:** Only mentions the PowerShell directory tree script.

**Proposed:**
```markdown
## Utilities

- `generate_ascii_dir_repr.ps1` — PowerShell script to generate an ASCII directory tree,
  with options to exclude dotfiles and `__pycache__`
- `src/utilities/` — Modular utility package:
  - `config.py` — Global settings, data paths, Django/GeoIP2 configuration
  - `data_preparation.py` — Parallel IP geolocation and device UA parsing with disk caching
  - `feature_engineering.py` — Continuous variable binning, crosstabs, bias scoring
  - `visualization.py` — Geographic plots, strip plots, interpretation histograms
  - `statistical_analysis.py` — MCC correlation and Chi-Square independence tests
  - `diagrams.py` — Sankey and parallel categories diagrams (Plotly)
  - `utils.py` — Categorical encoding and chart display helpers
```

**Rationale:** The `src/utilities/` package is the backbone of the project and should be
documented.

---

## 11. Add Team Composition Table (Minor Enhancement)

**Current:** Plain list of names.

**Proposed:** Keep as-is (names without roles), but move it to the end of the README so the
technical content appears first. Alternatively, format as a compact list:

```markdown
## Team

Eugenio La Cava | Otmane Qorchi | Janagam Vasantha | Elly Smagghe |
Kaloina Rakotobe | Sanchana Krishna Kumar | Siham Eldjouher
```

**Rationale:** The team section is important but not the first thing a reader needs to see.
Project description and setup instructions are higher priority.

---

## Summary of All Changes

| # | Section | Change Type | Priority |
|---|---------|-------------|----------|
| 1 | Title | Typo fix ("Analizer" → "Analyzer") | High |
| 2 | Description | New section (project overview + dataset) | High |
| 3 | Directory Tree | Full update (matches actual codebase) | High |
| 4 | EDA/Pipeline | Rewrite (legacy → current pipeline) | High |
| 5 | Getting Started | New section (installation + commands) | High |
| 6 | Models | New section (methodology + evaluation) | Medium |
| 7 | Commands | Add missing `run-app` command | Medium |
| 8 | Results | Expand beyond geolocation coverage | Medium |
| 9 | Performance | New section (caching + parallelism) | Medium |
| 10 | Utilities | Expand to cover `src/utilities/` package | Low |
| 11 | Team | Reposition to end of README | Low |

---

## Recommended Section Order for New README

1. Title (with fix)
2. Description (new)
3. Getting Started (new)
4. Pipeline (rewritten)
5. Models (new)
6. Results (expanded)
7. Performance (new)
8. Repo Organization (updated tree)
9. Utilities (expanded)
10. Package Management (keep, update commands)
11. Team (repositioned)
12. License
