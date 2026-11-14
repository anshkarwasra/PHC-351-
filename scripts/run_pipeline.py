"""Run the classical pipeline for the synthetic library manifest.

Usage:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --ids 0000
"""
import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.meshing.pipeline import process_manifest
from src.utils.inspection import summarise_pipeline

VOXEL_SIZE = 2.0 / 128


def main(argv=None):
    from src.utils.run_logging import start_run_log
    start_run_log("run_pipeline")
    parser = argparse.ArgumentParser(description="Run the classical mesh pipeline")
    parser.add_argument(
        "--manifest",
        default="data/synthetic/manifest.csv",
        help="Synthetic library manifest, relative to the project root by default",
    )
    parser.add_argument(
        "--out-root",
        default="data",
        help="Directory for processed metrics and generated meshes",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=VOXEL_SIZE,
        help="Physical voxel spacing used for surface extraction",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Optional manifest IDs to process instead of the full library",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help="Write stage previews for the first N meshes after processing",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "train_val", "test", "all"),
        default="all",
        help="Limit processing to instance IDs in this fixed split",
    )
    args = parser.parse_args(argv)
    if args.voxel_size <= 0:
        parser.error("--voxel-size must be greater than zero")
    if args.preview < 0:
        parser.error("--preview must be non-negative")

    os.chdir(PROJECT_ROOT)
    if args.split != "all":
        import json
        split_path = os.path.join(os.path.dirname(args.manifest), "splits.json")
        with open(split_path) as split_fh:
            splits = json.load(split_fh)
        selected = splits["train"] + splits["val"] if args.split == "train_val" else splits[args.split]
        args.ids = list(selected) if args.ids is None else list(set(args.ids) & set(selected))
    metrics_csv = process_manifest(
        args.manifest,
        out_root=args.out_root,
        voxel_size=args.voxel_size,
        only_ids=args.ids,
    )
    summarise_pipeline(metrics_csv)
    if args.preview:
        from scripts.preview_steps import preview_one
        import csv
        with open(args.manifest, newline="", encoding="utf-8") as manifest_fh:
            rows = list(csv.DictReader(manifest_fh))
        if args.ids is not None:
            selected_ids = {str(value).zfill(4) for value in args.ids}
            rows = [row for row in rows if str(row["id"]).zfill(4) in selected_ids]
        completed = 0
        for row in rows:
            for tier in ("clean", "moderate", "poor"):
                preview_one(row["id"], tier)
                completed += 1
                if completed >= args.preview:
                    return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
