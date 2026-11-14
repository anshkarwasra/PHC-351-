"""Create deterministic visual and numeric previews of the Project 75 pipeline."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
import pyvista as pv
import trimesh
import yaml
from scipy.ndimage import zoom
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.meshing.correction import correct_mesh, load_guidance_model, predict_gnn_elements
from src.meshing.pipeline import run_classical
from src.meshing.quality import _element_geometry, get_tets, get_correction_config
from src.meshing.surface_extract import extract_surface, segment
from src.simulation.fe_solver import run_fe_sanity_check
from src.utils.evaluation import _boundary_trimesh, evaluate_simulation_readiness
from src.utils.preview import (
    render_volume, render_segmentation, render_surface, render_repair,
    render_tetrahedral_quality, render_gnn_flags, render_correction,
    render_geometry, render_fe,
)

TIERS = ("clean", "moderate", "poor")
VOXEL_SIZE = 2.0 / 128


def _read_manifest():
    with (PROJECT_ROOT / "data" / "synthetic" / "manifest.csv").open(
        "r", newline="", encoding="utf-8",
    ) as source:
        return list(csv.DictReader(source))


def _get_row(mesh_id, tier):
    for row in _read_manifest():
        if row["id"].zfill(4) == str(mesh_id).zfill(4) and row["tier"] == tier:
            return row
    raise KeyError(f"manifest entry not found for {mesh_id}_{tier}")


def _load_classical(row, cache_dir):
    cache_mesh = cache_dir / f"{row['id']}_{row['tier']}_classical.vtu"
    cache_surface = cache_dir / f"{row['id']}_{row['tier']}_repaired.stl"
    cache_repair = cache_dir / f"{row['id']}_{row['tier']}_repair.json"
    cache_times = cache_dir / f"{row['id']}_{row['tier']}_timings.json"
    if (
        cache_mesh.is_file() and cache_surface.is_file()
        and cache_repair.is_file() and cache_times.is_file()
    ):
        return (
            trimesh.load(str(cache_surface), force="mesh"),
            pv.read(str(cache_mesh)),
            json.loads(cache_repair.read_text(encoding="utf-8")),
            json.loads(cache_times.read_text(encoding="utf-8")),
        )
    volume = np.load(row["volume_path"]).astype(np.float32)
    surface, grid, repair, timings = run_classical(
        volume, float(row["seg_threshold"]), VOXEL_SIZE,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    surface.export(str(cache_surface))
    grid.save(str(cache_mesh))
    cache_repair.write_text(json.dumps(repair, indent=2), encoding="utf-8")
    cache_times.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    return surface, grid, repair, timings


def _quality_values(grid):
    volumes, sj, _ = _element_geometry(np.asarray(grid.points), get_tets(grid))
    return {
        "n_tets": int(len(sj)),
        "poor_count": int(np.count_nonzero(sj < 0.25)),
        "poor_pct": float(np.mean(sj < 0.25) * 100.0),
        "severe_pct": float(np.mean(sj < 0.1) * 100.0),
        "min_sj": float(sj.min()),
        "mean_sj": float(sj.mean()),
        "p1_sj": float(np.percentile(sj, 1)),
        "volume": float(np.abs(volumes).sum()),
    }


def _write_report(cases):
    report_path = PROJECT_ROOT / "data" / "previews" / "preview_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sections = ["# Project 75 Preview Report", ""]
    for case in cases:
        sections.extend([
            f"## {case['id']} / {case['tier']}", "",
            "| Stage | Measurements |", "|---|---|",
        ])
        for stage, values in case["metrics"].items():
            sections.append(
                f"| {stage} | " + "; ".join(f"{key}: {value}" for key, value in values.items()) + " |"
            )
        sections.append("")
        for image in case["images"]:
            relative = image.relative_to(report_path.parent).as_posix()
            sections.extend([f"### {image.stem}", "", f"![{image.stem}]({relative})", ""])
    report_path.write_text("\n".join(sections), encoding="utf-8")
    return report_path


def preview_one(
    mesh_id, tier, correction_config=None, model=None,
    reference_path=None, include_rule=False,
):
    row = _get_row(mesh_id, tier)
    if reference_path is None:
        reference_path = Path(row["reference_path"])
        if not reference_path.is_absolute():
            reference_path = PROJECT_ROOT / reference_path
    volume_path = Path(row["volume_path"])
    if not volume_path.is_absolute():
        volume_path = PROJECT_ROOT / volume_path
    density_path = Path(row["density_path"])
    if not density_path.is_absolute():
        density_path = PROJECT_ROOT / density_path

    cache_dir = PROJECT_ROOT / "data" / "previews" / "cache"
    output_dir = PROJECT_ROOT / "data" / "previews" / f"{row['id']}_{tier}"
    output_dir.mkdir(parents=True, exist_ok=True)
    volume = np.load(volume_path).astype(np.float32)
    density = np.load(density_path)["density"].astype(np.float32)
    reference_mask = density >= 0.5
    if reference_mask.shape != volume.shape:
        if all(clean_size % degraded_size == 0
               for clean_size, degraded_size in zip(reference_mask.shape, volume.shape)):
            factors = tuple(
                clean_size // degraded_size
                for clean_size, degraded_size in zip(reference_mask.shape, volume.shape)
            )
            shaped = reference_mask.reshape(
                volume.shape[0], factors[0],
                volume.shape[1], factors[1],
                volume.shape[2], factors[2],
            )
            reference_mask = shaped.mean(axis=(1, 3, 5)) >= 0.5
        else:
            reference_mask = zoom(
                reference_mask.astype(np.uint8),
                tuple(target / source for target, source in zip(volume.shape, reference_mask.shape)),
                order=0,
            ).astype(bool)
    threshold = float(row["seg_threshold"])
    settings_path = PROJECT_ROOT / "configs" / "synthetic.yaml"
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
    tier_settings = settings.get("tiers", {}).get(tier, {})
    correction_config = get_correction_config(correction_config)
    mask = segment(volume, threshold)
    raw_surface = extract_surface(mask, VOXEL_SIZE)
    repaired_surface, classical_grid, repair, timings = _load_classical(row, cache_dir)
    reference_surface = trimesh.load(str(reference_path), force="mesh")

    images = []
    metrics = {}
    emit = lambda name, fn, *args: _record_image(
        images, output_dir / f"{name}.png", fn, *args,
    )

    emit("stage1_phantom", render_volume, density, volume, tier, tier_settings, threshold)
    metrics["1. Phantom/degraded volume"] = {
        "tier": tier, "blur": tier_settings.get("blur"),
        "noise": tier_settings.get("noise"), "threshold": threshold,
    }
    emit("stage2_segmentation", render_segmentation, volume, mask, reference_mask, tier)
    metrics["2. Segmentation"] = {
        "voxel_count": int(mask.sum()),
        "components_removed": max(
            0, int(ndimage.label(volume > threshold)[1] - ndimage.label(mask)[1]),
        ),
        "mask_difference_voxels": int(np.count_nonzero(mask != reference_mask)),
        "mask_volume_pct_of_reference": round(
            100.0 * float(mask.sum()) / max(int(reference_mask.sum()), 1), 3,
        ),
    }
    emit("stage3_marching_cubes", render_surface, raw_surface)
    metrics["3. Marching-cubes surface"] = {
        "vertices": len(raw_surface.vertices),
        "faces": len(raw_surface.faces),
        "watertight_before_repair": bool(raw_surface.is_watertight),
        "edge_length_min_median_max": _edge_length_summary(raw_surface),
    }
    emit("stage4_surface_repair", render_repair, raw_surface, repaired_surface, repair)
    metrics["4. Surface repair"] = {
        "faces_filled": repair.get("n_faces_filled", repair.get("n_loop_faces", 0)),
        "degenerate_faces_removed": repair.get("n_degenerate_removed", 0),
        "nonmanifold_faces_removed": repair.get("n_broken", 0),
        "watertight_after": bool(repaired_surface.is_watertight),
        "volume_change_pct": repair.get("volume_change_pct", ""),
    }

    classical_quality = _quality_values(classical_grid)
    emit("stage5_classical_quality", render_tetrahedral_quality, classical_grid)
    emit("stage5_gnn_quality", render_tetrahedral_quality, classical_grid)
    metrics["5. Classical tetrahedralisation"] = {
        key: classical_quality[key]
        for key in ("n_tets", "poor_pct", "severe_pct", "min_sj", "mean_sj", "p1_sj")
    }
    metrics["5. Classical tetrahedralisation"]["tetgen_time_sec"] = timings.get("tetgen", "")
    methods = ["gnn"] + (["rule_targeted"] if include_rule else [])
    _, classical_sj, _ = _element_geometry(
        np.asarray(classical_grid.points), get_tets(classical_grid),
    )
    emit(
        "stage6_classical_quality_flags", render_gnn_flags,
        classical_grid, (classical_sj < 0.25).astype(np.float32), 0.5,
    )
    classical_identity, classical_stats = correct_mesh(
        classical_grid, method="classical", correction_config=correction_config,
    )
    emit(
        "stage7_classical_correction", render_correction,
        classical_grid, classical_identity, classical_stats,
    )
    classical_geometry = evaluate_simulation_readiness(
        classical_grid, classical_identity, reference_stl_path=str(reference_path),
    )
    classical_fe = run_fe_sanity_check(classical_grid)
    emit(
        "stage8_classical_geometry", render_geometry,
        classical_grid, reference_surface,
    )
    emit(
        "stage9_classical_fe", render_fe,
        classical_grid, classical_fe["solution"],
        classical_fe["residual_history"], classical_fe,
    )
    metrics["7. Correction (classical)"] = {
        "poor_count_before_after": (
            classical_quality["poor_count"], classical_quality["poor_count"],
        ),
        "poor_pct_before_after": (
            round(classical_quality["poor_pct"], 3),
            round(classical_quality["poor_pct"], 3),
        ),
        "moved_points": 0, "modified_elements": 0,
    }
    metrics["8-9. Geometry/FE (classical)"] = {
        "hausdorff_mean": classical_geometry["post_chamfer_mean"],
        "hausdorff_p95": classical_geometry["post_hausdorff_p95"],
        "hausdorff_max": classical_geometry["post_hausdorff"],
        "volume_error_pct": round(
            100.0 * (classical_quality["volume"] - abs(reference_surface.volume))
            / max(abs(reference_surface.volume), 1e-12), 4,
        ),
        "watertight": bool(_boundary_trimesh(classical_grid).is_watertight),
        "assembly_ok": classical_fe["assembly_success"],
        "converged": classical_fe["solver_converged"],
        "iterations": classical_fe["iterations"],
        "final_residual": classical_fe["residual"],
    }
    for method in methods:
        if method == "gnn" and model is None:
            print(f"{row['id']}_{tier}: GNN stages skipped (checkpoint not supplied).")
            metrics["6. GNN flags"] = {"status": "skipped: checkpoint unavailable"}
            continue
        method_suffix = method
        before = classical_grid
        method_model = model if method == "gnn" else None
        threshold_value = None
        if method == "gnn":
            threshold_value = correction_config.get("gnn_prob_threshold")
            prediction = predict_gnn_elements(
                before, method_model, prob_threshold=threshold_value,
            )
            emit(
                f"stage6_{method}_flags", render_gnn_flags,
                before, prediction["probabilities"], method_model.prob_threshold,
            )
            labels = None
            if model is not None:
                from src.meshing.quality import compute_tetrahedral_metrics
                labels = compute_tetrahedral_metrics(
                    before, correction_config=correction_config,
                )["needs_correction"]
            flagged = prediction["flagged"]
            true_labels = (
                np.asarray(labels, dtype=bool) if labels is not None
                else prediction["metrics"]["scaled_jacobian"] < 0.25
            )
            true_positive = int(np.count_nonzero(flagged & true_labels))
            precision = true_positive / max(int(flagged.sum()), 1)
            recall = true_positive / max(int(true_labels.sum()), 1)
            metrics["6. GNN flags"] = {
                "flagged_count": int(flagged.sum()),
                "poor_tets_flagged_share_pct": round(
                    100.0 * true_positive / max(int(true_labels.sum()), 1), 3,
                ),
                "precision": round(precision, 4), "recall": round(recall, 4),
                "threshold": method_model.prob_threshold if threshold_value is None else threshold_value,
            }
        else:
            metrics["6. Rule-targeted flags"] = {
                "poor_element_count": int(np.count_nonzero(classical_sj < 0.25)),
                "rule": "elements with SJ < 0.25",
            }
        correction_start = time.perf_counter()
        corrected, correction_stats = correct_mesh(
            before, method_model, method=method, correction_config=correction_config,
            prob_threshold=threshold_value,
        )
        correction_stats["t_total_sec"] = time.perf_counter() - correction_start
        after_quality = _quality_values(corrected)
        emit(
            f"stage7_{method_suffix}_correction", render_correction,
            before, corrected, correction_stats,
        )
        metrics[f"7. Correction ({method})"] = {
            "poor_count_before_after": (
                classical_quality["poor_count"], after_quality["poor_count"],
            ),
            "poor_pct_before_after": (
                round(classical_quality["poor_pct"], 3), round(after_quality["poor_pct"], 3),
            ),
            "min_sj": round(after_quality["min_sj"], 5),
            "moved_points": correction_stats.get("moved_points", 0),
            "modified_elements": correction_stats.get("modified_elements", 0),
            "rejected_moves": correction_stats.get("rejected_moves", 0),
            "smoothing_time_sec": correction_stats.get("t_smoothing_sec", 0.0),
            "time_sec": correction_stats["t_total_sec"],
        }
        emit(
            f"stage8_{method_suffix}_geometry", render_geometry,
            corrected, reference_surface,
        )
        geometry = evaluate_simulation_readiness(
            before, corrected, reference_stl_path=str(reference_path),
        )
        fe = run_fe_sanity_check(corrected)
        emit(
            f"stage9_{method_suffix}_fe", render_fe,
            corrected, fe["solution"], fe["residual_history"], fe,
        )
        metrics[f"8-9. Geometry/FE ({method})"] = {
            "hausdorff_mean": geometry["post_chamfer_mean"],
            "hausdorff_p95": geometry["post_hausdorff_p95"],
            "hausdorff_max": geometry["post_hausdorff"],
            "volume_error_pct": round(
                100.0 * (after_quality["volume"] - abs(reference_surface.volume))
                / max(abs(reference_surface.volume), 1e-12), 4,
            ),
            "watertight": bool(_boundary_trimesh(corrected).is_watertight),
            "assembly_ok": fe["assembly_success"],
            "converged": fe["solver_converged"],
            "iterations": fe["iterations"],
            "final_residual": fe["residual"],
        }

    case = {"id": row["id"], "tier": tier, "images": images, "metrics": metrics}
    print(f"{row['id']}_{tier}: " + json.dumps(metrics, default=str))
    return case


def _record_image(images, path, render, *args):
    if render(path, *args):
        images.append(path)
    else:
        print(f"preview stage skipped; no image produced: {path}")


def _edge_length_summary(mesh):
    edges = np.asarray(mesh.edges_unique)
    lengths = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
    return tuple(round(float(value), 6) for value in (
        lengths.min(), np.median(lengths), lengths.max(),
    ))


def _sample_cases(split, n, instance_id=None):
    split_path = PROJECT_ROOT / "data" / "synthetic" / "splits.json"
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    ids = sorted(str(value).zfill(4) for value in splits[split])
    if instance_id is not None:
        selected_id = str(instance_id).zfill(4)
        if selected_id not in ids:
            raise ValueError(f"instance {instance_id!r} is not in the {split} split")
        ids = [selected_id]
    rows = []
    for mesh_id in ids:
        for tier in TIERS:
            rows.append((mesh_id, tier))
            if len(rows) == n:
                return rows
    return rows


def main(argv=None):
    from src.utils.run_logging import start_run_log
    start_run_log("preview_steps")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--instance-id")
    parser.add_argument("--with-rule", action="store_true")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "synthetic.yaml"))
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be at least one")

    os.chdir(PROJECT_ROOT)
    project_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    correction_config = get_correction_config(project_config.get("correction", {}))
    checkpoint = PROJECT_ROOT / "data" / "gnn_guidance_model.pt"
    model = load_guidance_model(str(checkpoint)) if checkpoint.is_file() else None
    cases = [
        preview_one(
            mesh_id, tier, correction_config=correction_config,
            model=model, include_rule=args.with_rule,
        )
        for mesh_id, tier in _sample_cases(args.split, args.n, args.instance_id)
    ]
    report = _write_report(cases)
    print(f"preview report: {report}")


if __name__ == "__main__":
    main()
