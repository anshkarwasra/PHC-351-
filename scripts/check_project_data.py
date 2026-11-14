"""Verify the existing synthetic library against Project 75 without regenerating it."""
import csv
import inspect
import json
from pathlib import Path

import numpy as np

from src.meshing.surface_extract import repair_surface, segment
from src.synthetic.csg_phantom import csg_density
from src.synthetic.phantom_library import PhantomLibrary

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TIERS = ("clean", "moderate", "poor")


def _check_data():
    manifest_path = PROJECT_ROOT / "data" / "synthetic" / "manifest.csv"
    split_path = PROJECT_ROOT / "data" / "synthetic" / "splits.json"
    with manifest_path.open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    instances = sorted({row["id"] for row in rows})
    tiers_by_id = {
        mesh_id: {row["tier"] for row in rows if row["id"] == mesh_id}
        for mesh_id in instances
    }

    checks = {
        "instance_count_100_to_150": 100 <= len(instances) <= 150,
        "three_required_tiers": all(tiers_by_id[mid] == set(EXPECTED_TIERS) for mid in instances),
        "splits_disjoint": (
            set(splits["train"]).isdisjoint(splits["val"])
            and set(splits["train"]).isdisjoint(splits["test"])
            and set(splits["val"]).isdisjoint(splits["test"])
        ),
        "split_ids_match_manifest": set(instances) == (
            set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        ),
        "split_ratios_70_15_15": all(
            abs(actual - expected) <= 0.02
            for actual, expected in zip(
                (
                    len(splits["train"]) / max(len(instances), 1),
                    len(splits["val"]) / max(len(instances), 1),
                    len(splits["test"]) / max(len(instances), 1),
                ),
                (0.70, 0.15, 0.15),
            )
        ),
    }
    volume_shapes = set()
    missing_paths = []
    for row in rows:
        for column in ("volume_path", "reference_path", "density_path", "spec_path"):
            path = Path(row[column])
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if not path.is_file():
                missing_paths.append(str(path))
        volume_path = Path(row["volume_path"])
        if not volume_path.is_absolute():
            volume_path = PROJECT_ROOT / volume_path
        if volume_path.is_file():
            volume_shapes.add(tuple(np.load(volume_path, mmap_mode="r").shape))
    checks["required_files_exist"] = not missing_paths
    checks["input_volume_resolution_128_cubed"] = volume_shapes == {(128, 128, 128)}

    csg_source = inspect.getsource(csg_density)
    segment_source = inspect.getsource(segment)
    repair_source = inspect.getsource(repair_surface)
    degradation_source = inspect.getsource(PhantomLibrary.degrade)
    checks["csg_ribs_fillets_channels"] = all(
        phrase in csg_source for phrase in (
            "rib_thickness", "channel_radius", "_smooth_union", "_rounded_box_sdf",
        )
    )
    checks["gaussian_noise_and_threshold_degradation"] = (
        "gaussian_filter" in degradation_source
        and "rng.normal" in degradation_source
        and "seg_threshold" in inspect.getsource(PhantomLibrary.write_instance_atomic)
    )
    checks["morphological_cleanup_before_surface"] = (
        "binary_fill_holes" in segment_source and "ndimage.label" in segment_source
    )
    checks["trimesh_surface_repair"] = all(
        token in repair_source for token in ("nondegenerate_faces", "fill_holes", "fix_normals")
    )
    return checks, instances, splits, volume_shapes, missing_paths


def main():
    from src.utils.run_logging import start_run_log
    start_run_log("check_project_data")
    checks, instances, splits, shapes, missing = _check_data()
    print("Project 75 synthetic-data verification")
    print(f"instances: {len(instances)}; tiers: {', '.join(EXPECTED_TIERS)}")
    print(
        "split counts: "
        + ", ".join(f"{name}={len(splits[name])}" for name in ("train", "val", "test"))
    )
    print(f"input volume shapes: {sorted(shapes)}")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    if missing:
        print(f"missing files ({len(missing)}): {missing[:5]}")
    if not checks["csg_ribs_fillets_channels"]:
        raise SystemExit("Required CSG ribs, fillets, or channels are missing; stopping for owner review.")
    if not all(checks.values()):
        raise SystemExit("Synthetic-data verification found spec mismatches; no data was changed.")


if __name__ == "__main__":
    main()
