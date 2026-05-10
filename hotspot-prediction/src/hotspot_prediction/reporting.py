"""Reporting helpers for dataset and experiment tables."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Any


def _count_hotspots(rows: Iterable[dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        if "n_hotspots" in row and str(row["n_hotspots"]) != "":
            total += int(row["n_hotspots"])
        elif "label" in row:
            total += int(row["label"])
        elif "hotspot_list" in row:
            total += sum(1 for item in str(row["hotspot_list"]).split(",") if item.strip())
    return total


def build_dataset_summary(
    data1_mutations: list[dict[str, Any]],
    train_samples: list[dict[str, Any]],
    data2_rows: list[dict[str, Any]],
    test_samples: list[dict[str, Any]],
    strict_train_samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a thesis-friendly dataset summary table."""
    return [
        {
            "dataset": "data1",
            "role": "mutation source",
            "records": len(data1_mutations),
            "pdb_chains": "",
            "hotspots": _count_hotspots(data1_mutations),
            "definition": "ddG >= 2.0 kcal/mol",
        },
        {
            "dataset": "data1_train_val",
            "role": "train/validation",
            "records": "",
            "pdb_chains": len(train_samples),
            "hotspots": _count_hotspots(train_samples),
            "definition": "aggregated from data1",
        },
        {
            "dataset": "data2",
            "role": "raw independent test source",
            "records": len(data2_rows),
            "pdb_chains": "",
            "hotspots": "",
            "definition": "curated hotspot list",
        },
        {
            "dataset": "data2_test",
            "role": "independent test",
            "records": "",
            "pdb_chains": len(test_samples),
            "hotspots": _count_hotspots(test_samples),
            "definition": "UniProt numbering mapped to PDB-relative numbering",
        },
        {
            "dataset": "strict_non_overlap",
            "role": "audit only",
            "records": "",
            "pdb_chains": len(strict_train_samples),
            "hotspots": _count_hotspots(strict_train_samples),
            "definition": "data1 train after removing data2 UniProt/PDB-chain overlap",
        },
    ]


def build_experiment_matrix() -> list[dict[str, str]]:
    """Describe recommended thesis experiments without inventing metric values."""
    return [
        {
            "method": "Rule-based baseline",
            "input": "residue type + simple exposure/center heuristics",
            "purpose": "engineering fallback and lower bound",
            "status": "run on cloud or backend",
        },
        {
            "method": "Traditional ML / PPI-hotspotID-style",
            "input": "aa type, SASA, conservation, energy-like features",
            "purpose": "baseline inspired by PPI-hotspotID",
            "status": "optional if feature extraction is available",
        },
        {
            "method": "ESM-2 embedding + MLP",
            "input": "frozen ESM-2 residue embeddings",
            "purpose": "test value of pretrained sequence features",
            "status": "recommended ablation",
        },
        {
            "method": "ESM-2 + GAT",
            "input": "frozen ESM-2 plus residue distance graph",
            "purpose": "main single model",
            "status": "implemented training path",
        },
        {
            "method": "ESM-2 + GAT ensemble",
            "input": "5-fold GAT checkpoints",
            "purpose": "main reported model",
            "status": "implemented evaluation path",
        },
    ]


def build_ablation_plan() -> list[dict[str, str]]:
    """Return non-destructive cloud ablation profiles.

    Each profile writes features, checkpoints, and reports to an isolated
    directory, so one ablation run does not delete or overwrite another run.
    """
    profiles = [
        {
            "profile": "esm2_only",
            "features": "ESM-2",
            "pssm_dim": "0",
            "hmm_dim": "0",
            "traditional_dim": "0",
        },
        {
            "profile": "esm2_pssm",
            "features": "ESM-2 + PSSM",
            "pssm_dim": "20",
            "hmm_dim": "0",
            "traditional_dim": "0",
        },
        {
            "profile": "esm2_hmm",
            "features": "ESM-2 + HMM",
            "pssm_dim": "0",
            "hmm_dim": "30",
            "traditional_dim": "0",
        },
        {
            "profile": "esm2_pssm_hmm",
            "features": "ESM-2 + PSSM + HMM",
            "pssm_dim": "20",
            "hmm_dim": "30",
            "traditional_dim": "0",
        },
    ]

    rows: list[dict[str, str]] = []
    for profile in profiles:
        name = profile["profile"]
        env_parts = [
            f"HOTSPOT_PSSM_DIM={profile['pssm_dim']}",
            f"HOTSPOT_HMM_DIM={profile['hmm_dim']}",
            f"HOTSPOT_TRADITIONAL_DIM={profile['traditional_dim']}",
            f"HOTSPOT_FEATURES_DIR=data/features/{name}",
            f"HOTSPOT_MODELS_DIR=models/{name}",
            f"HOTSPOT_RESULTS_DIR=results/{name}",
        ]
        rows.append(
            {
                **profile,
                "model": "GAT",
                "cloud_env": " ".join(env_parts),
                "status": "planned",
            }
        )
    return rows


def _read_table(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", None):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _mean(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def _std(values: list[float]) -> float | str:
    if len(values) < 2:
        return ""
    avg = sum(values) / len(values)
    return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def collect_ablation_results(results_root: str | Path) -> list[dict[str, Any]]:
    """Collect completed ablation metrics from isolated result directories."""
    results_root = Path(results_root)
    rows: list[dict[str, Any]] = []

    for plan in build_ablation_plan():
        profile = plan["profile"]
        result_dir = results_root / profile
        summary: dict[str, Any] = {
            "profile": profile,
            "features": plan["features"],
            "status": "pending",
            "cv_roc_auc_mean": "",
            "cv_roc_auc_std": "",
            "cv_pr_auc_mean": "",
            "cv_pr_auc_std": "",
            "cv_f1_mean": "",
            "cv_f1_std": "",
            "cv_mcc_mean": "",
            "cv_mcc_std": "",
            "test_roc_auc": "",
            "test_pr_auc": "",
            "test_f1": "",
            "test_precision": "",
            "test_recall": "",
            "test_mcc": "",
            "top3_hit_rate": "",
            "top3_hotspot_recall": "",
            "top5_hit_rate": "",
            "top5_hotspot_recall": "",
            "top10_hit_rate": "",
            "top10_hotspot_recall": "",
        }

        cv_path = result_dir / "cross_validation_results.csv"
        if cv_path.exists():
            cv_rows = _read_table(cv_path)
            for source_key, output_prefix in [
                ("roc_auc", "cv_roc_auc"),
                ("pr_auc", "cv_pr_auc"),
                ("f1", "cv_f1"),
                ("mcc", "cv_mcc"),
            ]:
                values = _float_values(cv_rows, source_key)
                summary[f"{output_prefix}_mean"] = _mean(values)
                summary[f"{output_prefix}_std"] = _std(values)

        metrics_path = result_dir / "test_metrics.csv"
        if metrics_path.exists():
            metric_rows = _read_table(metrics_path)
            if metric_rows:
                metric = metric_rows[0]
                for key in ["roc_auc", "pr_auc", "f1", "precision", "recall", "mcc"]:
                    summary[f"test_{key}"] = metric.get(key, "")

        topk_path = result_dir / "topk_metrics.csv"
        if topk_path.exists():
            topk_rows = _read_table(topk_path)
            for topk in topk_rows:
                k = str(topk.get("top_k", "")).strip()
                if k in {"3", "5", "10"}:
                    summary[f"top{k}_hit_rate"] = topk.get("topk_hit_rate", "")
                    summary[f"top{k}_hotspot_recall"] = topk.get("hotspot_recall", "")

        if cv_path.exists() or metrics_path.exists() or topk_path.exists():
            summary["status"] = "complete" if cv_path.exists() and metrics_path.exists() else "partial"

        rows.append(summary)

    return rows


def compute_topk_metrics(records: Iterable[dict[str, Any]], k_values: Iterable[int]) -> list[dict[str, Any]]:
    """Compute per-protein Top-K hit-rate and hotspot recall from prediction records."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["pdb_id"])].append(record)

    rows: list[dict[str, Any]] = []
    for k in k_values:
        proteins_with_hotspots = 0
        proteins_hit = 0
        total_hotspots = 0
        found_hotspots = 0
        selected_total = 0

        for protein_records in grouped.values():
            positives = sum(int(row["label"]) for row in protein_records)
            if positives <= 0:
                continue

            proteins_with_hotspots += 1
            total_hotspots += positives
            selected = sorted(protein_records, key=lambda row: float(row["prob"]), reverse=True)[:k]
            selected_total += len(selected)
            selected_hits = sum(int(row["label"]) for row in selected)
            found_hotspots += selected_hits
            if selected_hits > 0:
                proteins_hit += 1

        rows.append(
            {
                "top_k": int(k),
                "proteins_with_hotspots": proteins_with_hotspots,
                "proteins_hit": proteins_hit,
                "topk_hit_rate": proteins_hit / proteins_with_hotspots if proteins_with_hotspots else 0.0,
                "total_hotspots": total_hotspots,
                "hotspots_found": found_hotspots,
                "hotspot_recall": found_hotspots / total_hotspots if total_hotspots else 0.0,
                "selected_residues": selected_total,
            }
        )
    return rows


def write_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
