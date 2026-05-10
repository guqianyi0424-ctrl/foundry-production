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

python scripts/prepare_data.py
python scripts/train.py --force-reload
python scripts/evaluate_test.py
```

Expected outputs:

- normalized CSVs under `hotspot-prediction/processed/`
- feature caches under `hotspot-prediction/data/features/`
- fold checkpoints under `hotspot-prediction/models/`
- evaluation reports under `hotspot-prediction/results/`

For a CPU-only smoke test, run only:

```bash
export PYTHONPATH="$PWD/src:$PWD"
python scripts/prepare_data.py
```

Full training requires `torch`, `dgl`, `transformers`, `biopython`, `pandas`, `openpyxl`, and enough disk space for the ESM-2 model and downloaded PDB files.
