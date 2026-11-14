"""Classical baseline: threshold -> marching cubes -> trimesh repair -> TetGen, per manifest row."""
import csv
import os
import time

import numpy as np
import trimesh

from src.meshing.surface_extract import (
    segment, extract_surface, repair_surface,
)
from src.meshing.tetgen_wrapper import tetrahedralise
from src.meshing.quality import _element_geometry, get_tets, SJ_BAD
from src.utils.evaluation import _sampled_points, _hausdorff
from src.utils.progress import progress


def run_classical(volume, seg_threshold, voxel_size):
    timings = {}

    def timed(name, fn, *args):
        t0 = time.perf_counter()
        out = fn(*args)
        timings[name] = time.perf_counter() - t0
        return out

    mask = timed("segment", segment, volume, seg_threshold)
    surface = timed("extract", extract_surface, mask, voxel_size)
    surface, repair = timed("repair", repair_surface, surface)
    grid = timed("tetgen", tetrahedralise, surface)
    return surface, grid, repair, timings


def process_manifest(manifest_path, out_root="data", voxel_size=2.0 / 128,
                     only_ids=None):
    with open(manifest_path) as fh:
        rows = list(csv.DictReader(fh))
    if only_ids is not None:
        rows = [r for r in rows if r["id"] in only_ids]

    proc_dir = os.path.join(out_root, "processed")
    mesh_dir = os.path.join(out_root, "meshes")
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)

    results = []
    out_csv = os.path.join(proc_dir, "classical_metrics.csv")
    if os.path.isfile(out_csv):
        with open(out_csv, newline="") as fh:
            results.extend(csv.DictReader(fh))
    result_by_key = {(row["id"], row["tier"]): row for row in results}
    with progress(rows, desc="classical pipeline", unit="mesh") as bar:
        for r in bar:
            key = f"{r['id']}_{r['tier']}"
            res = {"id": r["id"], "tier": r["tier"], "status": "ok", "error": ""}
            try:
                volume = np.load(r["volume_path"]).astype(np.float32)
                surface, grid, repair, timings = run_classical(
                    volume, float(r["seg_threshold"]), voxel_size)
                surface.export(os.path.join(proc_dir, f"{key}.stl"))
                grid.save(os.path.join(mesh_dir, f"{key}.vtu"))
                pts = np.asarray(grid.points, dtype=np.float64)
                tets = get_tets(grid)
                _, sj, _ = _element_geometry(pts, tets)
                res.update(
                    surface_faces=repair["faces_out"],
                    n_broken=repair["n_broken"],
                    watertight=repair["watertight"],
                    n_tets=grid.n_cells,
                    n_points=grid.n_points,
                    min_sj=round(float(sj.min()), 4),
                    mean_sj=round(float(sj.mean()), 4),
                    poor_frac=round(float(np.mean(sj < SJ_BAD)), 5),
                    **{f"t_{k}": round(v, 4) for k, v in timings.items()},
                )
                reference_path = r.get("reference_path")
                if reference_path and os.path.isfile(reference_path):
                    reference = trimesh.load(reference_path, force="mesh")
                    res["surface_hausdorff"] = round(
                        _hausdorff(_sampled_points(surface), _sampled_points(reference)), 6
                    )
            except Exception as exc:   # record failures as data, keep going
                res.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:200])
            result_by_key[(res["id"], res["tier"])] = res
            bar.write(f"{key}: {res['status']}")

    results = list(result_by_key.values())
    fields = sorted({k for r in results for k in r})
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    return out_csv
