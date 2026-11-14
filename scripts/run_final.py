"""Run the frozen Project 75 method set on validation or held-out test meshes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyvista as pv
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.meshing.correction import correct_mesh, load_guidance_model
from src.meshing.quality import _element_geometry, get_tets, get_correction_config
from src.simulation.fe_solver import run_fe_sanity_check
from src.utils.evaluation import evaluate_simulation_readiness
from src.utils.progress import progress

LIB_DIR = PROJECT_ROOT / "data" / "synthetic"
MESH_DIR = PROJECT_ROOT / "data" / "meshes"
CLASSICAL_CSV = PROJECT_ROOT / "data" / "processed" / "classical_metrics.csv"
CHECKPOINT = PROJECT_ROOT / "data" / "gnn_guidance_model.pt"
METHODS = (
    "classical", "global_taubin", "global_taubin_retet",
    "blind_global_smart", "rule_targeted", "gnn", "retet_global_quality",
)
TIERS = ("clean", "moderate", "poor")
FIELDS = (
    "id", "tier", "method", "config_hash", "n_tets",
    "pre_poor_count", "poor_count", "poor_count_change", "pre_poor_pct",
    "poor_pct", "rel_count_change_pct", "severe_pct", "pre_min_sj",
    "inverted_pct", "min_sj", "mean_sj", "p1_sj", "hausdorff_mean", "hausdorff_p95",
    "hausdorff_max", "vol_err_pct", "watertight", "flagged_elements",
    "modified_elements", "moved_points", "growth", "fe_success", "fe_iters",
    "fe_residual", "t_classical_sec", "t_features_sec", "t_gnn_sec",
    "t_smoothing_sec", "t_reverification_sec", "t_fe_sec", "t_total_sec",
    "fallback", "selection_status", "status", "error",
)


def _normalise_id(value):
    value = str(value)
    return value.zfill(4) if value.isdigit() else value


def _load_manifest():
    manifest_path = LIB_DIR / "manifest.csv"
    with manifest_path.open("r", newline="", encoding="utf-8") as source:
        return {
            (_normalise_id(row["id"]), row["tier"]): row
            for row in csv.DictReader(source)
        }


def _load_classical_times():
    if not CLASSICAL_CSV.is_file():
        raise FileNotFoundError(
            f"classical stage timings not found: {CLASSICAL_CSV}"
        )
    frame = pd.read_csv(CLASSICAL_CSV, dtype={"id": str})
    required = ("id", "tier", "status", "t_segment", "t_extract", "t_repair", "t_tetgen")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"classical timing file is missing: {', '.join(missing)}")
    frame = frame[frame["status"] == "ok"]
    times = {}
    for row in frame.itertuples(index=False):
        values = (row.t_segment, row.t_extract, row.t_repair, row.t_tetgen)
        if any(pd.isna(value) for value in values):
            continue
        times[(_normalise_id(row.id), row.tier)] = float(sum(values))
    return times


def _split_meshes(split):
    split_path = LIB_DIR / "splits.json"
    with split_path.open("r", encoding="utf-8") as source:
        instance_ids = sorted({_normalise_id(value) for value in json.load(source)[split]})
    return [
        (mesh_id, tier, MESH_DIR / f"{mesh_id}_{tier}.vtu")
        for mesh_id in instance_ids for tier in TIERS
        if (MESH_DIR / f"{mesh_id}_{tier}.vtu").is_file()
    ]


def _row_config_hash(method, correction, threshold, label_version):
    payload = {
        "method": method,
        "correction": correction,
        "threshold": threshold,
        "label_version": label_version,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _mesh_quality(grid):
    points = np.asarray(grid.points, dtype=np.float64)
    tets = get_tets(grid)
    volumes, sj, _ = _element_geometry(points, tets)
    return {
        "tets": tets,
        "volumes": volumes,
        "sj": sj,
        "n_tets": len(tets),
        "poor_count": int(np.count_nonzero(sj < 0.25)),
        "poor_pct": float(np.mean(sj < 0.25) * 100.0),
        "severe_pct": float(np.mean(sj < 0.1) * 100.0),
        "inverted_pct": float(np.mean(volumes <= 0.0) * 100.0),
    }


def _volume_error_pct(grid, reference_path):
    reference = trimesh.load(str(reference_path), force="mesh")
    mesh_volume = float(np.sum(np.abs(_mesh_quality(grid)["volumes"])))
    reference_volume = abs(float(reference.volume))
    if reference_volume <= 0 or not np.isfinite(reference_volume):
        raise ValueError(f"reference mesh has invalid volume: {reference_path}")
    return 100.0 * (mesh_volume - reference_volume) / reference_volume


def _read_existing(output_path):
    if not output_path.exists():
        return set()
    with output_path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(
                f"{output_path} has a stale or incompatible schema; archive it before rerunning"
            )
        return {
            (row["id"], row["tier"], row["method"], row["config_hash"])
            for row in reader
            if row.get("config_hash") and row.get("status") == "ok"
        }


def _evaluate_one(
    mesh_id, tier, method, source_grid, reference_path, model,
    classical_sec, correction_config, threshold,
):
    total_start = time.perf_counter()
    before = _mesh_quality(source_grid)
    if method == "classical":
        corrected = source_grid
        stats = {
            "flagged_cells": 0, "modified_elements": 0, "moved_points": 0,
            "fallback": False, "growth": 1.0,
        }
    else:
        corrected, stats = correct_mesh(
            source_grid,
            model=model if method == "gnn" else None,
            method=method,
            prob_threshold=threshold if method == "gnn" else None,
            mesh_id=mesh_id,
            tier=tier,
            correction_config=correction_config,
        )
    verification_start = time.perf_counter()
    after = _mesh_quality(corrected)
    geometry = evaluate_simulation_readiness(
        source_grid, corrected, reference_stl_path=str(reference_path),
    )
    volume_error = _volume_error_pct(corrected, reference_path)
    surface = corrected.extract_surface().triangulate()
    watertight = bool(trimesh.Trimesh(
        np.asarray(surface.points),
        np.asarray(surface.faces).reshape(-1, 4)[:, 1:],
        process=False,
    ).is_watertight)
    verification_sec = time.perf_counter() - verification_start
    fe_start = time.perf_counter()
    fe = run_fe_sanity_check(corrected)
    fe_sec = time.perf_counter() - fe_start

    poor_change = after["poor_count"] - before["poor_count"]
    pre_count = before["poor_count"]
    return {
        "id": mesh_id,
        "tier": tier,
        "method": method,
        "n_tets": after["n_tets"],
        "pre_poor_count": pre_count,
        "poor_count": after["poor_count"],
        "poor_count_change": poor_change,
        "pre_poor_pct": before["poor_pct"],
        "poor_pct": after["poor_pct"],
        "rel_count_change_pct": 100.0 * poor_change / max(pre_count, 1),
        "severe_pct": after["severe_pct"],
        "inverted_pct": after["inverted_pct"],
        "pre_min_sj": float(before["sj"].min()),
        "min_sj": float(after["sj"].min()),
        "mean_sj": float(after["sj"].mean()),
        "p1_sj": float(np.percentile(after["sj"], 1)),
        "hausdorff_mean": geometry["post_chamfer_mean"],
        "hausdorff_p95": geometry["post_hausdorff_p95"],
        "hausdorff_max": geometry["post_hausdorff"],
        "vol_err_pct": volume_error,
        "watertight": watertight,
        "flagged_elements": int(stats.get("flagged_cells", 0)),
        "modified_elements": int(stats.get("modified_elements", 0)),
        "moved_points": int(stats.get("moved_points", 0)),
        "growth": float(after["n_tets"] / max(before["n_tets"], 1)),
        "fe_success": bool(fe["assembly_success"] and fe["solver_converged"]),
        "fe_iters": int(fe["iterations"]),
        "fe_residual": fe["residual"],
        "t_classical_sec": classical_sec,
        "t_features_sec": float(stats.get("t_feature_sec", 0.0)),
        "t_gnn_sec": float(stats.get("t_inference_sec", 0.0)),
        "t_smoothing_sec": float(stats.get("t_smoothing_sec", 0.0)),
        "t_reverification_sec": verification_sec,
        "t_fe_sec": fe_sec,
        "t_total_sec": time.perf_counter() - total_start + classical_sec,
        "fallback": bool(stats.get("fallback", False)),
        "status": "ok",
        "error": "",
    }


def main(argv=None):
    from src.utils.run_logging import start_run_log
    start_run_log("run_final")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--confirm-test", action="store_true")
    parser.add_argument(
        "--allow-exploratory", action="store_true",
        help="allow an explicitly marked, non-qualified exploratory validation configuration",
    )
    parser.add_argument(
        "--max-phantoms", type=int, default=None,
        help="evaluate an evenly spaced subset of phantom IDs (exploratory validation only)",
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "synthetic.yaml"))
    parser.add_argument("--preview", type=int, default=0, metavar="N")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.confirm_test:
        parser.error("--split test requires --confirm-test")
    if args.allow_exploratory and args.split != "val":
        parser.error("--allow-exploratory is restricted to --split val")
    if args.max_phantoms is not None:
        if args.max_phantoms <= 0:
            parser.error("--max-phantoms must be greater than zero")
        if args.split != "val" or not args.allow_exploratory:
            parser.error("--max-phantoms requires --split val --allow-exploratory")
    if args.preview < 0:
        parser.error("--preview must be non-negative")

    os.chdir(PROJECT_ROOT)
    with Path(args.config).open("r", encoding="utf-8") as source:
        import yaml
        project_config = yaml.safe_load(source) or {}
    correction_config = get_correction_config(project_config.get("correction", {}))
    from src.meshing.quality import LABEL_VERSION

    manifest = _load_manifest()
    classical_times = _load_classical_times()
    meshes = _split_meshes(args.split)
    if not meshes:
        raise FileNotFoundError(f"no classical meshes found for split {args.split!r}")
    selected_ids = sorted({mesh_id for mesh_id, _, _ in meshes})
    if args.max_phantoms is not None and args.max_phantoms < len(selected_ids):
        indices = np.linspace(
            0, len(selected_ids) - 1, args.max_phantoms,
        ).round().astype(int)
        selected_ids = [selected_ids[index] for index in indices]
        selected_id_set = set(selected_ids)
        meshes = [mesh for mesh in meshes if mesh[0] in selected_id_set]

    model = None
    threshold = correction_config.get("gnn_prob_threshold")
    selection_status = "not_applicable"
    if "gnn" in METHODS:
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f"trained GNN checkpoint not found: {CHECKPOINT}")
        model = load_guidance_model(str(CHECKPOINT))
        if model.label_version != LABEL_VERSION:
            raise ValueError(
                "GNN checkpoint labels do not match the final operator; retrain before evaluation"
            )
        selected_path = PROJECT_ROOT / "data" / "evaluation_reports" / "selected_config.json"
        if not selected_path.is_file():
            raise FileNotFoundError(
                f"validation-selected GNN configuration not found: {selected_path}"
            )
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        selection_status = selected.get(
            "selection_status",
            selected.get("validation_selection", {}).get("selection_status", "eligible"),
        )
        if selected.get("label_version") != LABEL_VERSION:
            raise ValueError("selected GNN threshold was calibrated for different labels")
        if selected.get("correction") != correction_config:
            raise ValueError(
                "correction config differs from the validation-selected configuration"
            )
        if selection_status == "exploratory_ineligible":
            if args.split != "val":
                raise ValueError("exploratory calibration configurations cannot be used on test")
            if not args.allow_exploratory:
                raise ValueError(
                    "selected GNN configuration is exploratory and failed validation gates; "
                    "rerun with --split val --allow-exploratory to produce explicitly "
                    "exploratory results"
                )
        elif selection_status != "eligible":
            raise ValueError(
                f"unsupported validation selection status: {selection_status!r}"
            )
        threshold = float(selected["gnn_prob_threshold"])
        if not 0.0 <= threshold <= 1.0 or not np.isfinite(threshold):
            raise ValueError(f"invalid validation-selected GNN threshold: {threshold}")
        if not np.isclose(threshold, model.prob_threshold):
            raise ValueError(
                "selected GNN threshold differs from the checkpoint calibration"
            )

    output_name = f"final_{args.split}"
    if selection_status == "exploratory_ineligible":
        output_name += "_exploratory"
    if args.max_phantoms is not None:
        output_name += f"_{len(selected_ids)}phantoms"
    output_path = (
        PROJECT_ROOT / "data" / "evaluation_reports" / f"{output_name}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if selection_status == "exploratory_ineligible":
        sample_path = output_path.with_name(f"{output_name}_sample.json")
        sample_path.write_text(
            json.dumps({
                "status": selection_status,
                "split": args.split,
                "phantom_ids": selected_ids,
                "tiers": list(TIERS),
                "methods": list(METHODS),
                "row_count_expected": len(meshes) * len(METHODS),
                "note": (
                    "Exploratory validation subset only; not validation-qualified "
                    "and not a held-out test result."
                ),
            }, indent=2),
            encoding="utf-8",
        )
    seen = _read_existing(output_path)

    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
            destination.flush()
        with progress(meshes, desc=f"{args.split} final study", unit="mesh") as bar:
            for index, (mesh_id, tier, mesh_path) in enumerate(bar, start=1):
                classical_key = (mesh_id, tier)
                if classical_key not in classical_times:
                    raise KeyError(f"classical stage timings missing for {mesh_id}_{tier}")
                if classical_key not in manifest:
                    raise KeyError(f"manifest entry missing for {mesh_id}_{tier}")
                reference_path = Path(manifest[classical_key]["reference_path"])
                if not reference_path.is_absolute():
                    reference_path = PROJECT_ROOT / reference_path
                source_grid = pv.read(str(mesh_path))
                for method in METHODS:
                    config_hash = _row_config_hash(
                        method, correction_config,
                        threshold if method == "gnn" else None, LABEL_VERSION,
                    )
                    key = (mesh_id, tier, method, config_hash)
                    if key in seen:
                        continue
                    try:
                        row = _evaluate_one(
                            mesh_id, tier, method, source_grid, reference_path,
                            model, classical_times[classical_key], correction_config,
                            threshold,
                        )
                    except Exception as exc:
                        row = {field: "" for field in FIELDS}
                        row.update(
                            id=mesh_id, tier=tier, method=method,
                            t_classical_sec=classical_times[classical_key],
                            status="error",
                            error=f"{type(exc).__name__}: {exc}"[:300],
                        )
                    row["config_hash"] = config_hash
                    row["selection_status"] = selection_status
                    writer.writerow(row)
                    destination.flush()
                    seen.add(key)
                if args.preview and index <= args.preview:
                    from scripts.preview_steps import preview_one
                    preview_one(
                        mesh_id=mesh_id, tier=tier,
                        correction_config=correction_config, model=model,
                        reference_path=reference_path,
                    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
