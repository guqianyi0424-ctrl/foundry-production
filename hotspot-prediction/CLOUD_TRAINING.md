# Cloud Training Commands

These commands retrain the hotspot residue model with data1 as train/validation data and data2 as the independent test set.

```bash
git clone git@github.com:guqianyi0424-ctrl/foundry-production.git
cd foundry-production/hotspot-prediction

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the CUDA PyTorch build that matches your cloud image first if needed.
# Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

export PYTHONPATH="$PWD/src:$PWD"

# Default: keep all data1 train/validation samples and report overlap with data2.
# This keeps training usable with the current small data1/data2 pair.
python scripts/prepare_data.py --overlap-policy none
python scripts/generate_reports.py
python scripts/train.py --force-reload
python scripts/evaluate_test.py --threshold 0.5

# Add Top-K tables after evaluation has produced results/test_predictions.csv.
python scripts/generate_reports.py --predictions-csv results/test_predictions.csv
```

Expected outputs:

- normalized CSVs under `hotspot-prediction/processed/`
- feature caches under `hotspot-prediction/data/features/`
- fold checkpoints under `hotspot-prediction/models/`
- evaluation reports under `hotspot-prediction/results/`
- thesis tables under `hotspot-prediction/results/`

For a CPU-only smoke test, run only:

```bash
export PYTHONPATH="$PWD/src:$PWD"
python scripts/prepare_data.py
```

Full training requires `torch`, `dgl`, `transformers`, `biopython`, `pandas`, `openpyxl`, and enough disk space for the ESM-2 model and downloaded PDB files.

## Baseline-style overlap filtering

The PPI-hotspotID benchmark uses nonredundant proteins. With the two bundled source files, strict overlap filtering by UniProt or PDB-chain leaves only 7 data1 train/validation samples, so it is useful for an audit but too small for normal GAT retraining.

Run this command to generate the strict split and inspect the sample count:

```bash
python scripts/prepare_data.py --overlap-policy uniprot-or-pdb-chain
```

For publishable comparison, use a larger training set or cluster all sequences at the benchmark threshold, for example 60% identity, and split by cluster.

## Thesis ablation runs

The older `ablation_study.py` edits `config.py` and clears checkpoint/cache files, so avoid it for repeated cloud runs. Use environment-variable profiles instead. Each run writes to its own feature, model, and result directory.

```bash
cd foundry-production/hotspot-prediction
export PYTHONPATH="$PWD/src:$PWD"
python scripts/prepare_data.py --overlap-policy none
python scripts/generate_reports.py

# 1) ESM-2 only
HOTSPOT_PSSM_DIM=0 HOTSPOT_HMM_DIM=0 HOTSPOT_TRADITIONAL_DIM=0 \
HOTSPOT_FEATURES_DIR=data/features/esm2_only \
HOTSPOT_MODELS_DIR=models/esm2_only \
HOTSPOT_RESULTS_DIR=results/esm2_only \
python scripts/train.py --force-reload

HOTSPOT_PSSM_DIM=0 HOTSPOT_HMM_DIM=0 HOTSPOT_TRADITIONAL_DIM=0 \
HOTSPOT_FEATURES_DIR=data/features/esm2_only \
HOTSPOT_MODELS_DIR=models/esm2_only \
HOTSPOT_RESULTS_DIR=results/esm2_only \
python scripts/evaluate_test.py --threshold 0.5

python scripts/generate_reports.py \
  --predictions-csv results/esm2_only/test_predictions.csv \
  --output-dir results/esm2_only

# 2) ESM-2 + PSSM
HOTSPOT_PSSM_DIM=20 HOTSPOT_HMM_DIM=0 HOTSPOT_TRADITIONAL_DIM=0 \
HOTSPOT_FEATURES_DIR=data/features/esm2_pssm \
HOTSPOT_MODELS_DIR=models/esm2_pssm \
HOTSPOT_RESULTS_DIR=results/esm2_pssm \
python scripts/train.py --force-reload

HOTSPOT_PSSM_DIM=20 HOTSPOT_HMM_DIM=0 HOTSPOT_TRADITIONAL_DIM=0 \
HOTSPOT_FEATURES_DIR=data/features/esm2_pssm \
HOTSPOT_MODELS_DIR=models/esm2_pssm \
HOTSPOT_RESULTS_DIR=results/esm2_pssm \
python scripts/evaluate_test.py --threshold 0.5

python scripts/generate_reports.py \
  --predictions-csv results/esm2_pssm/test_predictions.csv \
  --output-dir results/esm2_pssm

# 3) ESM-2 + HMM
HOTSPOT_PSSM_DIM=0 HOTSPOT_HMM_DIM=30 HOTSPOT_TRADITIONAL_DIM=0 \
HOTSPOT_FEATURES_DIR=data/features/esm2_hmm \
HOTSPOT_MODELS_DIR=models/esm2_hmm \
HOTSPOT_RESULTS_DIR=results/esm2_hmm \
python scripts/train.py --force-reload

HOTSPOT_PSSM_DIM=0 HOTSPOT_HMM_DIM=30 HOTSPOT_TRADITIONAL_DIM=0 \
HOTSPOT_FEATURES_DIR=data/features/esm2_hmm \
HOTSPOT_MODELS_DIR=models/esm2_hmm \
HOTSPOT_RESULTS_DIR=results/esm2_hmm \
python scripts/evaluate_test.py --threshold 0.5

python scripts/generate_reports.py \
  --predictions-csv results/esm2_hmm/test_predictions.csv \
  --output-dir results/esm2_hmm

# 4) Main model: ESM-2 + PSSM + HMM
HOTSPOT_PSSM_DIM=20 HOTSPOT_HMM_DIM=30 HOTSPOT_TRADITIONAL_DIM=0 \
HOTSPOT_FEATURES_DIR=data/features/esm2_pssm_hmm \
HOTSPOT_MODELS_DIR=models/esm2_pssm_hmm \
HOTSPOT_RESULTS_DIR=results/esm2_pssm_hmm \
python scripts/train.py --force-reload

HOTSPOT_PSSM_DIM=20 HOTSPOT_HMM_DIM=30 HOTSPOT_TRADITIONAL_DIM=0 \
HOTSPOT_FEATURES_DIR=data/features/esm2_pssm_hmm \
HOTSPOT_MODELS_DIR=models/esm2_pssm_hmm \
HOTSPOT_RESULTS_DIR=results/esm2_pssm_hmm \
python scripts/evaluate_test.py --threshold 0.5

python scripts/generate_reports.py \
  --predictions-csv results/esm2_pssm_hmm/test_predictions.csv \
  --output-dir results/esm2_pssm_hmm

# Aggregate real completed ablation metrics into one table.
python scripts/generate_reports.py \
  --ablation-results-root results \
  --output-dir results
```

Report these as exploratory ablations for a small undergraduate dataset. The key presentation is the trend across feature groups plus the independent data2 result, not claiming a large-scale benchmark.

## Paper-safe evaluation rules

Do not use `--find-threshold` for the paper's main independent-test table. That option searches the test labels and is only acceptable as an oracle analysis in an appendix. For the main result, use a fixed threshold such as `0.5` or a threshold selected from validation folds and recorded before running `scripts/evaluate_test.py`.

Before writing the thesis tables, deduplicate or report data2 at the PDB-chain level. The raw data2 source contains repeated records; the paper should present independent-test metrics on the normalized PDB-chain samples and Top-K metrics from `results/test_predictions.csv`.

The current bundled data is small and overlapping: strict removal of data2-overlapping data1 proteins leaves too few training samples for normal GAT retraining. Treat strict non-overlap as an audit table, and present the normal data1-train/data2-test experiment as an undergraduate exploratory study rather than a publication-grade benchmark.
