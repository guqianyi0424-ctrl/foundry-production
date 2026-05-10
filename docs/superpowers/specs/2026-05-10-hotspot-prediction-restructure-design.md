# Hotspot Prediction Restructure Design

## Goal

Refactor the `hotspot-prediction` project into a clearer, reproducible pipeline while preserving the main modeling framework: ESM-2 residue embeddings plus a GAT residue graph classifier. The refactor will fix known data, training, evaluation, checkpoint, and inference consistency problems.

## Dataset Policy

Two source datasets are used with distinct roles.

`elife-96643-data1-v1.xlsx` is the training and validation source. It contains alanine or other mutation measurements with binding free energy change. A residue is labeled as a hotspot when `Binding free energy change (kcal/mol) > 2.0`; otherwise it is labeled as non-hotspot. This rule is the source of truth, even if the original `PPI-hot spots assignment` column disagrees.

`elife-96643-data2-v1.xlsx` is the independent test source only. It contains protein-level hotspot lists in UniProt numbering. These positions are converted to PDB residue numbering using the free protein A structure range, for example `5f18-A(374-620)`.

Training, validation, threshold selection, and model selection must not use data2 labels. Data2 is evaluated once with a threshold selected on validation data.

## Architecture

The existing `hotspot-prediction/` directory remains the project root. The implementation will move from large scripts into focused modules under `src/hotspot_prediction/`, with thin scripts or CLI entry points for common workflows.

Planned structure:

```text
hotspot-prediction/
  configs/
    default.yaml
  scripts/
    prepare_data.py
    train.py
    evaluate_test.py
  src/hotspot_prediction/
    cli.py
    config.py
    data/
      normalize.py
      pdb.py
      dataset.py
    features/
      esm.py
      graph.py
      cache.py
    models/
      gat.py
      losses.py
    training/
      train.py
      metrics.py
      thresholds.py
    evaluation/
      evaluate.py
    inference/
      predictor.py
  tests/
```

Compatibility wrappers may remain in the current top-level files (`dataset.py`, `model.py`, `train.py`, `evaluate_test.py`, `main.py`) so existing commands and backend imports do not break immediately.

## Data Flow

Data normalization produces explicit intermediate tables:

- `processed/data1_mutations.csv`: one row per mutation record from data1, with PDB id, chain id, UniProt id, PDB residue number, wild-type residue, mutant residue, ddG, and binary label from `ddG > 2.0`.
- `processed/train_val_samples.csv`: one row per PDB-chain training sample with residue-level labels aggregated from data1.
- `processed/data2_test_samples.csv`: one row per PDB-chain test sample with data2 hotspot lists converted from UniProt numbering to PDB-relative numbering.

Feature preparation downloads or locates PDB files, extracts residue sequences, C-alpha coordinates, residue numbers, ESM-2 embeddings, optional zero-filled PSSM/HMM feature blocks, and graph edges. Processed graph samples are cached with a metadata file containing feature dimensions, ESM model name, graph cutoff, dataset hash, and code version.

## Model

The model remains a node classifier over residue graphs:

- Node features: ESM-2 650M embeddings plus optional PSSM and HMM feature blocks, matching the current 1330-dimensional default.
- Graph edges: residue pairs within a configurable C-alpha distance cutoff, with the same graph construction used for training, evaluation, and backend inference.
- Classifier: GAT layers with residual connections and focal or weighted focal loss.

The existing edge feature calculation will either be removed from the model contract or wired into a graph layer that actually consumes edge features. The first implementation will keep plain GAT and remove misleading claims that distance/cosine edge features are used by the model.

## Training

Training uses data1 only. Cross-validation uses grouped splits by UniProt id, falling back to PDB-chain grouping if UniProt id is absent. This prevents the same protein family or exact chain from appearing in both training and validation folds.

Class imbalance is handled with weighted focal loss and optional sample weighting. Batch collation will be corrected so labels align with concatenated node features for any batch size, although the default can remain `batch_size=1` for memory safety.

Checkpoints are saved and loaded only with `torch.save` and `torch.load`, including model config, fold, feature metadata, selected validation threshold, and validation metrics.

## Evaluation

Validation folds report residue-level ROC-AUC, PR-AUC, precision, recall, specificity, F1, MCC, and confusion matrix counts. Fold thresholds are selected only from validation predictions.

Independent data2 evaluation loads trained fold checkpoints, ensembles their probabilities, applies a validation-derived threshold, and reports full residue-level metrics plus per-protein summary metrics. It must not search for a best test threshold.

Balanced/downsampled metrics may be generated as secondary analysis, but they must be labeled as secondary and must not replace full-dataset metrics.

## Inference And Backend Integration

The backend hotspot predictor will use the same feature and graph construction modules as training. It will cache the predictor instance so ESM-2 and checkpoint loading do not repeat for every request.

If the GAT stack cannot load, the predictor may fall back to rule-based or ESM-attention scoring, but the API response must expose `method`, `model_loaded`, and fallback reason clearly.

## Testing

Tests will cover:

- Data1 label generation from the `ddG > 2.0` rule.
- Data2 UniProt-to-PDB numbering conversion.
- PDB id and chain parsing.
- Batch collation alignment for multiple proteins with different lengths.
- Checkpoint load/save path uses `torch.load`, not `pickle.load`.
- Threshold selection does not inspect data2 labels during independent evaluation.
- Backend predictor reuses a cached predictor instance.

Heavy ESM-2 and DGL execution will be isolated behind interfaces so unit tests can use small fake arrays and lightweight fake models.

## Cloud Training

The refactor will provide commands for a cloud machine with Python, CUDA PyTorch, DGL, Transformers, and enough disk space for ESM-2 and PDB caches. The final commands will include:

- dependency installation,
- data preparation,
- training on data1,
- independent data2 evaluation,
- optional push of result artifacts.

Exact commands will be finalized after implementation so they match the CLI and file paths actually present in the repo.

## Non-Goals

This refactor will not replace ESM-2 with a different protein language model, rewrite the backend UI, or introduce a new deep learning framework. It will not use data2 for model selection.
