# Hotspot Prediction Thesis Experiment Guide

This guide turns the current hotspot-prediction project into a thesis-ready experiment workflow. The goal is not to claim a large-scale benchmark win, but to present a reproducible undergraduate study: same source data as the PPI-hotspotID baseline, a deep-learning hotspot predictor, fair evaluation, ablation experiments, and clear limitations.

## 1. Paper Positioning

Use this thesis framing:

> A hotspot residue prediction method based on ESM-2 residue embeddings and graph attention networks is designed and evaluated on the same source dataset used by the PPI-hotspotID-style baseline. The study focuses on small-sample and imbalanced-data behavior, independent-test evaluation, and Top-K hotspot recommendation.

Avoid claiming that the model exceeds the original PPI-hotspotID paper unless the baseline is rerun on exactly the same processed samples and evaluation protocol.

## 2. Known Data Constraints

Current processed data summary:

| Dataset | Role | Scale | Notes |
|---|---|---:|---|
| data1 | mutation source | 690 records, 271 positive mutations | label by `ddG >= 2.0 kcal/mol` |
| data1_train_val | train/validation | 49 PDB chains, 134 hotspots | small training set |
| data2_test | independent test | 74 PDB chains, about 235 hotspots | UniProt positions mapped to PDB-relative positions |
| strict_non_overlap | audit only | 7 PDB chains, 18 hotspots | too small for normal retraining |

Important limitations to write explicitly:

- The dataset is small for deep learning.
- Positive and negative residues are highly imbalanced.
- data1 and data2 contain substantial UniProt/PDB-chain overlap, so strict non-overlap is only an audit.
- Raw data2 has repeated records; thesis tables should use normalized PDB-chain samples and report the preprocessing rule.

## 3. Required Main Experiments

Run these methods and put them in the main comparison table.

| Method | Purpose | Required output |
|---|---|---|
| Rule-based baseline | lower bound and engineering fallback | independent-test metrics and Top-K |
| PPI-hotspotID-style baseline | traditional ML comparison if feature extraction is available | independent-test metrics on same samples |
| ESM-2 + MLP | ablation for pretrained residue embeddings | CV and independent-test metrics |
| ESM-2 + GAT | main single deep-learning model | CV and independent-test metrics |
| ESM-2 + GAT ensemble | main reported model | independent-test metrics and Top-K |

If the full PPI-hotspotID pipeline cannot run because conservation, SASA, AmberTools, or FreeSASA features are missing, write it as an incomplete reproduced baseline and use the rule-based baseline plus ESM-2 MLP as the fair comparison.

## 4. Metrics To Report

Main residue-level metrics:

- ROC-AUC
- PR-AUC
- Precision
- Recall/Sensitivity
- Specificity
- F1
- MCC
- Confusion matrix: TP, FP, TN, FN

Application-oriented metrics:

- Top-3 hit rate per protein
- Top-5 hit rate per protein
- Top-10 hit rate per protein
- Top-K hotspot recall

Use the Top-K metrics as a key thesis result because hotspot prediction is often used to recommend candidate residues for validation.

## 5. Threshold Rule

For the independent test set, do not search the best threshold on data2.

Allowed for the main paper:

- Fixed threshold `0.5`, or
- Threshold selected from validation-fold predictions before evaluating data2.

Not allowed as main result:

- `scripts/evaluate_test.py --find-threshold`
- `fair_comparison.py` threshold search on data2 labels

If used, test-label threshold search must be labeled as an oracle upper-bound analysis.

## 6. Cloud Training Commands

Run on the cloud server from the repository root.

```bash
cd foundry-production/hotspot-prediction

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Choose the CUDA PyTorch build matching the cloud image.
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

export PYTHONPATH="$PWD/src:$PWD"
```

Prepare normalized data and report tables:

```bash
python scripts/prepare_data.py --overlap-policy none
python scripts/generate_reports.py
```

Train the main ESM-2 + GAT model:

```bash
python scripts/train.py --force-reload
```

Evaluate the 5-fold checkpoint ensemble on data2 with a paper-safe fixed threshold:

```bash
python scripts/evaluate_test.py --threshold 0.5
python scripts/generate_reports.py --predictions-csv results/test_predictions.csv
```

Expected key artifacts:

| Artifact | Purpose |
|---|---|
| `processed/data1_mutations.csv` | mutation-level training labels |
| `processed/train_val_samples.csv` | data1 train/validation PDB-chain samples |
| `processed/data2_test_samples.csv` | normalized data2 independent-test samples |
| `results/dataset_summary.csv` | thesis dataset table |
| `results/cross_validation_results.csv` | 5-fold CV table |
| `results/test_metrics.csv` | independent-test metrics |
| `results/test_predictions.csv` | residue-level probabilities for Top-K |
| `results/topk_metrics.csv` | Top-K recommendation metrics |
| `results/test_roc_curve.png` | ROC figure |
| `results/test_pr_curve.png` | PR figure |

## 7. Ablation Runs

Use isolated directories so one run does not overwrite another.

### 7.1 ESM-2 Only

```bash
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
```

### 7.2 Main Profile

```bash
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
```

Only report PSSM/HMM profiles as real ablations if the corresponding feature files are actually generated. If these dimensions are zero-filled placeholders, label the run as a configuration check instead of a biological-feature improvement.

Aggregate completed ablation metrics:

```bash
python scripts/generate_reports.py \
  --ablation-results-root results \
  --output-dir results
```

## 8. Baseline Plan

### 8.1 Rule-Based Baseline

Use simple residue-level scoring as a lower bound:

- aromatic or charged residue type gets higher score;
- surface-exposed residues get higher score if SASA is available;
- residues near the interface or protein center can be ranked if structure-derived distances are available.

Report it as an engineering fallback, not a scientific state-of-the-art method.

### 8.2 PPI-HotspotID-Style Baseline

Use `ppihotspotid-main` only if the same normalized data2 PDB-chain samples can be featurized. Required external tools may include AmberTools, FreeSASA, xssp/DSSP, and conservation features.

Fair comparison requirements:

- Same PDB chains.
- Same residue labels.
- Same metric code.
- Same threshold rule or probability ranking.
- No comparing against paper numbers from a different split.

## 9. Thesis Tables And Figures

Recommended thesis outputs:

| Section | Table/Figure |
|---|---|
| Dataset | data1/data2 summary, label definition, overlap audit |
| Method | architecture diagram: sequence/PDB to ESM-2, graph, GAT, residue score |
| CV results | 5-fold ROC-AUC, PR-AUC, F1, MCC mean/std |
| Independent test | residue-level metrics and confusion matrix |
| Top-K analysis | Top-3/5/10 hit rate and hotspot recall |
| Ablation | ESM-2 only vs GAT vs ensemble |
| Baseline comparison | rule baseline, PPI-hotspotID-style baseline if available, ours |
| Limitations | small sample, imbalance, overlap, feature availability |

## 10. Paper Wording For Weak Results

Use wording like this:

> Under a small and highly imbalanced hotspot residue dataset, the proposed ESM-2 + GAT model shows usable ranking ability but limited threshold-classification performance. Therefore, this study reports both residue-level classification metrics and per-protein Top-K recommendation metrics. The results indicate that pretrained protein language model embeddings and residue graph structure can support hotspot candidate ranking, while larger nonredundant datasets and complete traditional features are needed for stronger generalization conclusions.

Do not write:

- "The proposed model fully outperforms PPI-hotspotID."
- "The model has reached publication-level performance."
- "Balanced-sampling metrics prove real independent-test performance."

## 11. Final Cloud Checklist

Before writing the thesis result chapter, confirm these files exist:

```bash
ls -lh results/dataset_summary.csv
ls -lh results/cross_validation_results.csv
ls -lh results/test_metrics.csv
ls -lh results/test_predictions.csv
ls -lh results/topk_metrics.csv
ls -lh results/test_roc_curve.png
ls -lh results/test_pr_curve.png
```

Then copy the key numeric values into the thesis tables and describe the limitations honestly.
