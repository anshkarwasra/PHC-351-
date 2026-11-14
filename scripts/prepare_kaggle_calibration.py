"""Stage only the checkpoint and validation inputs needed for Kaggle calibration."""
import argparse
import json
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIERS = ("clean", "moderate", "poor")


def _copy_file(source, destination):
    if not source.is_file():
        raise FileNotFoundError(f"required calibration input not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare(output_dir):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose another path or clear it first"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_path = PROJECT_ROOT / "data" / "synthetic" / "splits.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    val_ids = sorted({str(mesh_id).zfill(4) for mesh_id in splits["val"]})
    bundle_splits = {"train": [], "val": val_ids, "test": []}

    shutil.copytree(PROJECT_ROOT / "src", output_dir / "src", dirs_exist_ok=True)
    _copy_file(
        PROJECT_ROOT / "configs" / "synthetic.yaml",
        output_dir / "configs" / "synthetic.yaml",
    )
    _copy_file(
        PROJECT_ROOT / "data" / "gnn_guidance_model.pt",
        output_dir / "data" / "gnn_guidance_model.pt",
    )
    splits_destination = output_dir / "data" / "synthetic" / "splits.json"
    splits_destination.parent.mkdir(parents=True, exist_ok=True)
    splits_destination.write_text(json.dumps(bundle_splits, indent=2), encoding="utf-8")

    mesh_dir = PROJECT_ROOT / "data" / "meshes"
    cache_dir = PROJECT_ROOT / "data" / "graph_cache"
    for mesh_id in val_ids:
        for tier in TIERS:
            mesh_name = f"{mesh_id}_{tier}.vtu"
            _copy_file(mesh_dir / mesh_name, output_dir / "data" / "meshes" / mesh_name)
            cache_matches = sorted(cache_dir.glob(f"{mesh_id}_{tier}.*.pt"))
            for cache_path in cache_matches:
                _copy_file(
                    cache_path,
                    output_dir / "data" / "graph_cache" / cache_path.name,
                )

    print(f"Prepared Kaggle calibration bundle: {output_dir}")
    print(f"Validation meshes: {len(val_ids) * len(TIERS)}")
    print(
        "Graph caches included: "
        f"{len(list((output_dir / 'data' / 'graph_cache').glob('*.pt')))}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "kaggle_calibration" / "dataset"),
        help="destination for the private Kaggle input dataset",
    )
    args = parser.parse_args()
    prepare(args.output)
