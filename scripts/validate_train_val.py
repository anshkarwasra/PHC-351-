"""Verify train/validation mesh availability without consulting the test split."""
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    from src.utils.run_logging import start_run_log
    start_run_log("validate_train_val")
    library_dir = PROJECT_ROOT / "data" / "synthetic"
    mesh_dir = PROJECT_ROOT / "data" / "meshes"
    splits = json.loads((library_dir / "splits.json").read_text(encoding="utf-8"))
    train_ids, val_ids, test_ids = map(
        set, (splits["train"], splits["val"], splits["test"]),
    )
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("instance IDs overlap across fixed splits")

    metrics_path = PROJECT_ROOT / "data" / "processed" / "classical_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"classical metrics not found: {metrics_path}")
    metrics = pd.read_csv(metrics_path, dtype={"id": str})
    train_val_ids = train_ids | val_ids
    expected = {(mesh_id, tier) for mesh_id in train_val_ids
                for tier in ("clean", "moderate", "poor")}
    available = {
        (row.id, row.tier)
        for row in metrics.itertuples(index=False)
        if row.id in train_val_ids and row.status == "ok"
    }
    missing_rows = expected - available
    missing_meshes = [
        f"{mesh_id}_{tier}.vtu" for mesh_id, tier in sorted(expected)
        if not (mesh_dir / f"{mesh_id}_{tier}.vtu").is_file()
    ]
    if missing_rows or missing_meshes:
        raise ValueError(
            f"train/validation baseline incomplete: {len(missing_rows)} metric rows "
            f"and {len(missing_meshes)} mesh files missing"
        )

    summary = metrics[
        metrics["id"].isin(train_val_ids) & (metrics["status"] == "ok")
    ].groupby("tier").agg(
        meshes=("id", "count"),
        median_poor_fraction=("poor_frac", "median"),
        median_min_sj=("min_sj", "median"),
    )
    print(
        f"train instances={len(train_ids)}, validation instances={len(val_ids)}, "
        f"held-out test instances={len(test_ids)}"
    )
    print("Train/validation baseline coverage: complete.")
    print(summary.to_string())


if __name__ == "__main__":
    main()
