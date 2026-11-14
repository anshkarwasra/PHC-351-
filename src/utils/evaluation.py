"""Simulation-readiness metrics, computed identically for pre and post meshes.

Poor element = scaled Jacobian < POOR_SJ, applied to both meshes.
Keys match what scripts/evaluate.py reads:
  pre_/post_ min_scaled_jacobian, poor_fraction, inverted_fraction
  volume_change_pct, n_elements
"""
import os
import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import cKDTree

from src.meshing.quality import _element_geometry, _tet_dihedral_angles, get_tets

POOR_SJ = 0.25
N_SURFACE_SAMPLES = 20000
SAMPLE_SEED = 0


def _sj_and_vol(grid):
    pts = np.asarray(grid.points, dtype=np.float64)
    tets = get_tets(grid)
    vol, sj, _ = _element_geometry(pts, tets)
    return vol, sj


def _boundary_trimesh(grid):
    """Boundary surface of a tet mesh, taken from element faces owned by one tet (not extract_surface)."""
    from src.meshing.quality import _face_table
    pts = np.asarray(grid.points, dtype=np.float64)
    tets = get_tets(grid)
    faces, _, _, row_count = _face_table(tets)
    bf = faces[row_count == 1]
    return trimesh.Trimesh(pts, bf, process=False)


def _sampled_points(tm, seed=SAMPLE_SEED):
    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(tm, N_SURFACE_SAMPLES, seed=rng)
    return pts


def _hausdorff(a, b):
    return float(max(cKDTree(b).query(a)[0].max(), cKDTree(a).query(b)[0].max()))


def _surface_metrics(a, b):
    a_to_b = cKDTree(b).query(a)[0]
    b_to_a = cKDTree(a).query(b)[0]
    return {
        "hausdorff": float(max(a_to_b.max(), b_to_a.max())),
        "hausdorff_p95": float(np.percentile(np.concatenate([a_to_b, b_to_a]), 95)),
        "chamfer_mean": float((a_to_b.mean() + b_to_a.mean()) * 0.5),
    }


def evaluate_simulation_readiness(mesh_before, mesh_after, reference_stl_path=None):
    out = {}
    v0, sj0 = _sj_and_vol(mesh_before)
    v1, sj1 = _sj_and_vol(mesh_after)
    out["n_elements"] = int(len(sj0))
    out["n_elements_after"] = int(len(sj1))

    for tag, sj, v in (("pre", sj0, v0), ("post", sj1, v1)):
        out[f"{tag}_min_scaled_jacobian"] = float(sj.min())
        out[f"{tag}_mean_scaled_jacobian"] = float(sj.mean())
        out[f"{tag}_poor_fraction"] = float(np.mean(sj < POOR_SJ))
        out[f"{tag}_poor_count"] = int(np.sum(sj < POOR_SJ))
        out[f"{tag}_inverted_fraction"] = float(np.mean(v <= 0))
        out[f"{tag}_sj_p1"] = float(np.percentile(sj, 1))
        out[f"{tag}_sj_p5"] = float(np.percentile(sj, 5))
        out[f"{tag}_sj_median"] = float(np.median(sj))
        prefix = "pre" if tag == "pre" else "post"
        pts = np.asarray((mesh_before if tag == "pre" else mesh_after).points, dtype=np.float64)
        tets = get_tets(mesh_before if tag == "pre" else mesh_after)
        angles = _tet_dihedral_angles(pts, tets)
        _, _, lengths = _element_geometry(pts, tets)
        edge_ratio = lengths.max(axis=1) / np.maximum(lengths.min(axis=1), 1e-12)
        out[f"{prefix}_min_dihedral"] = float(angles.min())
        out[f"{prefix}_max_dihedral"] = float(angles.max())
        out[f"{prefix}_mean_aspect_ratio"] = float(edge_ratio.mean())
        out[f"{prefix}_max_aspect_ratio"] = float(edge_ratio.max())

    # signed and absolute totals: an inverted tet cancels volume in the signed sum
    tot0 = float(np.sum(np.abs(v0)))
    tot1 = float(np.sum(np.abs(v1)))
    out["volume_change_pct"] = float(100.0 * abs(tot1 - tot0) / max(tot0, 1e-12))
    out["post_signed_volume_change_pct"] = float(
        100.0 * abs(np.sum(v1) - np.sum(v0)) / max(abs(np.sum(v0)), 1e-12))

    if reference_stl_path and os.path.exists(reference_stl_path):
        ref = trimesh.load(reference_stl_path, force="mesh")
        ref_pts = _sampled_points(ref)
        for tag, grid in (("pre", mesh_before), ("post", mesh_after)):
            surf = _boundary_trimesh(grid)
            out.update({
                f"{tag}_{key}": value
                for key, value in _surface_metrics(_sampled_points(surf), ref_pts).items()
            })
    return out