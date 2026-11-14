"""Shared correction operators and GNN guidance for the frozen Project 75 study."""
import time
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import torch
import trimesh

from src.models.gnn_model import MeshQualityGNN
from src.meshing.quality import (
    compute_tetrahedral_metrics, build_element_dual_graph, _element_geometry,
    _boundary_vertices_and_elements, _face_table, _EDGE_PAIRS, TAUBIN_LAMBDA,
    TAUBIN_MU, N_SMOOTH, SJ_BAD, VOL_EPS, smart_smooth,
    get_correction_config,
)
from src.meshing.tetgen_wrapper import tetrahedralise

GNN_PROB_THRESHOLD = 0.5


def load_guidance_model(ckpt_path, device="cpu"):
    """Loads the checkpoint dict written by gnn_train.train()."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ck, dict) or "state_dict" not in ck:
        raise ValueError(f"{ckpt_path} is not a training checkpoint dict; retrain")
    cfg = ck["config"]
    model = MeshQualityGNN(ck["in_channels"], cfg["hidden_dim"], cfg["n_blocks"], cfg["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    model.feature_mean = ck["feature_mean"].to(device)
    model.feature_std = ck["feature_std"].to(device)
    model.label_version = ck.get("label_version")
    calibration = ck.get("calibration", {})
    model.prob_threshold = float(calibration.get("prob_threshold", GNN_PROB_THRESHOLD))
    model.smoothing_scale = float(calibration.get("smoothing_scale", 1.0))
    return model


def _build_adjacency(tets, n):
    e = np.concatenate([tets[:, [a, b]] for a, b in _EDGE_PAIRS])
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    return A, deg


def _apply_with_rejection(points, tets, vertex_mask, vertex_weights,
                          A, inv_deg, lam, mu, n_iter):
    """Taubin iterations where a vertex's move is kept only if all incident tets stay positive.

    Rejection is per vertex per iteration. Rejected vertices keep their previous position.
    Returns new points and the number of rejected vertex moves.
    """
    rejected = 0
    x = points.copy()
    for _ in range(n_iter):
        for w in (lam, mu):
            nbr = (A @ x) * inv_deg[:, None]
            strength = vertex_mask * vertex_weights
            proposal = x + w * strength[:, None] * (nbr - x)
            moved = np.where(vertex_mask & (np.abs(proposal - x).sum(axis=1) > 0))[0]
            if len(moved) == 0:
                continue
            # test all moved vertices at once; fall back to per-vertex check only where needed
            trial = x.copy()
            trial[moved] = proposal[moved]
            vol_ok = _tets_positive(trial, tets)
            bad_tets = np.where(~vol_ok)[0]
            if len(bad_tets) == 0:
                x = trial
                continue
            # Roll back until no combined moves leave an incident tet inverted.
            while len(bad_tets):
                bad_vertices = np.unique(tets[bad_tets].ravel())
                revert = np.intersect1d(bad_vertices, moved)
                if len(revert) == 0:
                    break
                trial[revert] = x[revert]
                rejected += len(revert)
                bad_tets = np.where(~_tets_positive(trial, tets))[0]
            x = trial
    return x, rejected


def _tets_positive(points, tets):
    p = points[tets]
    det = np.einsum("ij,ij->i",
                    np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                    p[:, 3] - p[:, 0])
    return det > VOL_EPS


def predict_gnn_elements(grid, model, face_table=None, prob_threshold=None):
    """Return per-element GNN probabilities, weights, and inference timings."""
    t0 = time.perf_counter()
    metrics = compute_tetrahedral_metrics(grid, face_table, include_labels=False)
    feature_time = time.perf_counter() - t0
    tets = metrics["tets"]
    mean = model.feature_mean.cpu().numpy()
    std = model.feature_std.cpu().numpy()
    x = (metrics["features"] - mean) / std
    edge_index = build_element_dual_graph(tets, face_table)
    seeds = metrics["scaled_jacobian"] < SJ_BAD
    region = seeds.copy()
    frontier = np.flatnonzero(seeds)
    adjacency = sp.csr_matrix(
        (np.ones(edge_index.shape[1]), edge_index),
        shape=(len(tets), len(tets)),
    )
    for _ in range(2):
        if not len(frontier):
            break
        neighbors = adjacency[frontier].indices
        frontier = np.setdiff1d(neighbors, np.flatnonzero(region), assume_unique=False)
        region[frontier] = True
    nodes = np.flatnonzero(region)
    inference_start = time.perf_counter()
    probabilities = np.zeros(len(tets), dtype=np.float32)
    predicted_weights = np.zeros(len(tets), dtype=np.float32)
    if len(nodes):
        node_map = np.full(len(tets), -1, dtype=np.int64)
        node_map[nodes] = np.arange(len(nodes))
        keep_edges = region[edge_index[0]] & region[edge_index[1]]
        local_edges = node_map[edge_index[:, keep_edges]]
        x_tensor = torch.from_numpy(x[nodes]).float().to(next(model.parameters()).device)
        edge_tensor = torch.from_numpy(local_edges).long().to(x_tensor.device)
        with torch.no_grad():
            logit, weight = model(x_tensor, edge_tensor)
            threshold = prob_threshold if prob_threshold is not None else getattr(
                model, "prob_threshold", GNN_PROB_THRESHOLD,
            )
            local_probabilities = torch.sigmoid(logit).cpu().numpy()
            local_weights = weight.cpu().numpy()
        probabilities[nodes] = local_probabilities
        predicted_weights[nodes] = local_weights
    inference_time = time.perf_counter() - inference_start
    threshold = prob_threshold if prob_threshold is not None else getattr(
        model, "prob_threshold", GNN_PROB_THRESHOLD,
    )
    return {
        "probabilities": probabilities,
        "flagged": probabilities > float(threshold),
        "weights": np.clip(
            predicted_weights * float(getattr(model, "smoothing_scale", 1.0)),
            0.0, 1.0,
        ),
        "feature_time": feature_time,
        "inference_time": inference_time,
        "metrics": metrics,
    }


def _gnn_vertex_mask(grid, model, tets, points, face_table,
                     allow_boundary=False, weight_mode="predicted",
                     prob_threshold=None):
    """Return flagged vertices and their strength weights from the trained GNN."""
    prediction = predict_gnn_elements(
        grid, model, face_table=face_table, prob_threshold=prob_threshold,
    )
    flagged = prediction["flagged"]
    predicted_weights = prediction["weights"]
    metrics = prediction["metrics"]
    bv = metrics["boundary_vertex"]
    vmask = np.zeros(len(points), dtype=bool)
    vmask[np.unique(tets[flagged].ravel())] = True
    if not allow_boundary:
        vmask &= ~bv
    flagged_tets = tets[flagged]
    vertex_weights = np.zeros(len(points), dtype=np.float64)
    if len(flagged_tets):
        flagged_vertex_ids = flagged_tets.ravel()
        if weight_mode == "unit":
            vertex_weights[flagged_vertex_ids] = 1.0
        elif weight_mode == "predicted":
            vertex_weights += np.bincount(
                flagged_vertex_ids,
                weights=np.repeat(predicted_weights[flagged], 4),
                minlength=len(points),
            )
            counts = np.bincount(flagged_vertex_ids, minlength=len(points))
            vertex_weights /= np.maximum(counts, 1)
        else:
            raise ValueError("weight_mode must be 'predicted' or 'unit'")
    return (
        vmask, int(flagged.sum()), vertex_weights,
        prediction["feature_time"], prediction["inference_time"],
    )


def retet_from_surface(surface_trimesh, params):
    """TetGen re-tetrahedralization from a surface, with a guarded fallback."""
    params = dict(params or {})
    timeout = float(params.pop("timeout", 120.0))
    max_growth = float(params.pop("max_growth", 1.6))
    n_before = int(params.pop("n_before", params.pop("classical_count", 0)))
    source = surface_trimesh
    if not source.is_watertight:
        raise ValueError("re-tetrahedralization requires a watertight surface")
    start = time.perf_counter()
    if n_before <= 0 and hasattr(source, "vertices"):
        n_before = 0
    grid = None
    failure_reason = None
    try:
        if timeout > 0:
            project_root = Path(__file__).resolve().parents[2]
            with tempfile.TemporaryDirectory(prefix="mesh-retet-") as temp_dir:
                temp_path = Path(temp_dir)
                input_path = temp_path / "surface.stl"
                output_path = temp_path / "mesh.vtu"
                params_path = temp_path / "params.json"
                source.export(input_path)
                params_path.write_text(json.dumps(params), encoding="utf-8")
                try:
                    result = subprocess.run(
                        [
                            sys.executable, "-m", "src.meshing.retet_worker",
                            str(input_path), str(output_path), str(params_path),
                        ],
                        cwd=project_root,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise TimeoutError(
                        f"re-tetrahedralization exceeded {timeout:g} seconds"
                    ) from exc
                if result.returncode != 0:
                    detail = result.stderr.strip() or result.stdout.strip()
                    raise RuntimeError(
                        f"re-tetrahedralization worker exited with "
                        f"{result.returncode}: {detail[-1000:]}"
                    )
                if not output_path.is_file():
                    raise RuntimeError(
                        "re-tetrahedralization worker did not produce an output mesh"
                    )
                grid = pv.read(str(output_path))
        else:
            grid = tetrahedralise(source, **params)
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        grid = None
    t_retet = time.perf_counter() - start
    info = {"n_before": n_before, "n_after": 0 if grid is None else int(grid.n_cells),
            "t_retet": float(t_retet), "fallback": False}
    if grid is None:
        info["fallback"] = True
        info["fallback_reason"] = failure_reason
        return None, info
    if n_before > 0 and grid.n_cells > max_growth * n_before:
        info["fallback"] = True
        return None, info
    return grid, info


REMOVED_METHODS = {
    "classical_cond", "global_smart", "global_smart_boundary",
    "global_smart_boundary_penalty", "rule_targeted_poor_count",
    "rule_targeted_penalty", "rule_targeted_boundary",
    "rule_targeted_boundary_penalty", "gnn_poor_count", "gnn_penalty",
    "gnn_taubin_legacy", "gnn_boundary", "gnn_boundary_penalty",
    "gnn_unit", "gnn_unit_penalty", "retet_q_a", "retet_q_b",
    "retet_q_c", "retet_q_b_nobisect",
}


def correct_mesh(
    grid, model=None, method="gnn", prob_threshold=None,
    weight_mode="predicted", mesh_id=None, tier=None, correction_config=None,
):
    """Returns a corrected grid and stats for the selected correction method."""
    if method in REMOVED_METHODS:
        raise ValueError(f"method {method!r} was removed in V3; use the consolidated method set")
    methods = {
        "classical", "global_taubin", "global_taubin_retet",
        "blind_global_smart", "rule_targeted", "gnn", "retet_global_quality",
    }
    if method not in methods:
        raise ValueError(f"unknown method {method!r}")
    correction = get_correction_config(correction_config)
    points = np.asarray(grid.points, dtype=np.float64)
    tets = np.asarray(grid.cells_dict[pv.CellType.TETRA], dtype=np.int64)
    n = len(points)
    face_table = _face_table(tets)
    A, deg = _build_adjacency(tets, n)
    inv_deg = 1.0 / np.maximum(deg, 1.0)
    bv, _ = _boundary_vertices_and_elements(tets, n, face_table)
    interior = (~bv) & (deg > 0)
    feature_time = inference_time = 0.0
    revalidation_time = 0.0
    smoothing_time = 0.0
    moved_points = 0
    rejected = 0
    retet_grid = None
    retet_info = {}
    vmask = np.zeros(n, dtype=bool)
    flagged = 0
    vertex_weights = np.ones(n, dtype=np.float64)

    if method == "classical":
        new_pts = points.copy()
    elif method == "global_taubin":
        vmask, flagged = interior, int(np.sum(_element_geometry(points, tets)[1] < SJ_BAD))
        vertex_weights = np.ones(n, dtype=np.float64)
        smoothing_start = time.perf_counter()
        new_pts, rejected = _apply_with_rejection(
            points, tets, vmask, vertex_weights, A, inv_deg,
            TAUBIN_LAMBDA, TAUBIN_MU, N_SMOOTH,
        )
        smoothing_time = time.perf_counter() - smoothing_start
    elif method in ("blind_global_smart", "rule_targeted"):
        sj = _element_geometry(points, tets)[1]
        targeted = sj < SJ_BAD
        allow_boundary = bool(correction["allow_boundary"])
        if method == "blind_global_smart":
            flagged = int(targeted.sum())
            vmask = deg > 0
        else:
            flagged = int(targeted.sum())
            vmask[:] = False
            vmask[np.unique(tets[targeted].ravel())] = True
        vmask &= (deg > 0)
        vertex_weights = np.ones(n, dtype=np.float64)
        smoothing_start = time.perf_counter()
        new_pts, smooth_stats = smart_smooth(
            points, tets, vmask, int(correction["n_iter"]),
            allow_boundary=allow_boundary,
            face_table=face_table,
            acceptance=str(correction["acceptance"]),
            floor_ratio=float(correction["floor_ratio"]),
            sj_target=float(correction["sj_target"]),
        )
        smoothing_time = time.perf_counter() - smoothing_start
        rejected = sum(smooth_stats["rejected_per_iteration"])
    elif method == "gnn":
        if model is None:
            raise ValueError(f"method={method!r} requires a model")
        allow_boundary = bool(correction["allow_boundary"])
        configured_threshold = correction.get("gnn_prob_threshold")
        selected_threshold = (
            prob_threshold if prob_threshold is not None else configured_threshold
        )
        vmask, flagged, vertex_weights, feature_time, inference_time = _gnn_vertex_mask(
            grid, model, tets, points, face_table,
            allow_boundary=allow_boundary,
            weight_mode=weight_mode,
            prob_threshold=selected_threshold,
        )
        smoothing_start = time.perf_counter()
        new_pts, smooth_stats = smart_smooth(
            points, tets, vmask, int(correction["n_iter"]),
            vertex_weights=vertex_weights, face_table=face_table,
            allow_boundary=allow_boundary,
            acceptance=str(correction["acceptance"]),
            floor_ratio=float(correction["floor_ratio"]),
            sj_target=float(correction["sj_target"]),
        )
        smoothing_time = time.perf_counter() - smoothing_start
        rejected = sum(smooth_stats["rejected_per_iteration"])
    elif method == "retet_global_quality":
        params = {
            "mindihedral": 10,
            "minratio": 1.5,
            "max_growth": 1.6,
            "n_before": len(tets),
        }
        surface = None
        repaired_surface = (
            Path("data/processed") / f"{mesh_id}_{tier}.stl"
            if mesh_id is not None and tier is not None else None
        )
        if repaired_surface is not None and repaired_surface.exists():
            surface = trimesh.load_mesh(str(repaired_surface), process=True)
        if surface is None:
            surface = grid.extract_surface().triangulate()
        if isinstance(surface, trimesh.Trimesh):
            tm = surface
        else:
            faces = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
            tm = trimesh.Trimesh(np.asarray(surface.points), faces, process=True)
        retet_grid, info = retet_from_surface(tm, params)
        retet_info = info
        if retet_grid is None:
            new_pts = points.copy()
            out = grid
            flagged = int(np.sum(_element_geometry(points, tets)[1] < SJ_BAD))
            rejected = 0
        else:
            new_pts = points.copy()
            out = retet_grid
            flagged = int(np.sum(_element_geometry(points, tets)[1] < SJ_BAD))
        moved_points = int(np.sum(np.any(new_pts != points, axis=1)))
    elif method == "global_taubin_retet":
        vmask = interior
        flagged = int(np.sum(_element_geometry(points, tets)[1] < SJ_BAD))
        vertex_weights = np.ones(n, dtype=np.float64)
        smoothing_start = time.perf_counter()
        smoothed, rejected = _apply_with_rejection(
            points, tets, vmask, vertex_weights, A, inv_deg,
            TAUBIN_LAMBDA, TAUBIN_MU, N_SMOOTH,
        )
        smoothing_time = time.perf_counter() - smoothing_start
        moved_points = int(np.sum(np.any(smoothed != points, axis=1)))
        surface = grid.extract_surface().triangulate()
        faces = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
        tm = trimesh.Trimesh(np.asarray(surface.points), faces, process=True)
        if not tm.is_watertight:
            raise ValueError("retetrahedralisation requires a watertight boundary")
        mean_volume = float(np.mean(np.abs(_element_geometry(smoothed, tets)[0])))
        retet_start = time.perf_counter()
        retet = tetrahedralise(tm, maxvolume=mean_volume * 0.5)
        revalidation_time = time.perf_counter() - retet_start
        retet_grid = retet
        new_pts = smoothed
    else:
        raise ValueError(f"unknown method {method}")

    if method == "global_taubin_retet":
        if retet_grid is None:
            raise RuntimeError("retetrahedralisation did not produce a mesh")
        out = retet_grid
    elif method != "retet_global_quality":
        out = pv.UnstructuredGrid(grid.cells, grid.celltypes, new_pts)
        moved_points = int(np.sum(np.any(new_pts != points, axis=1)))
    if method not in ("global_taubin_retet", "retet_global_quality"):
        validation_start = time.perf_counter()
        _element_geometry(new_pts, tets)
        revalidation_time = time.perf_counter() - validation_start
    modified_elements = (
        len(tets) if method in ("global_taubin_retet", "retet_global_quality")
        and not retet_info.get("fallback", False)
        else int(np.sum(np.any(np.isin(tets, np.flatnonzero(np.any(new_pts != points, axis=1))), axis=1)))
    )
    boundary_idxs = np.flatnonzero(bv)
    stats = {
        "flagged_cells": flagged,
        "rejected_moves": rejected,
        "moved_points": moved_points,
        "modified_elements": modified_elements,
        "mean_smoothing_weight": (
            float(vertex_weights[vmask].mean()) if np.any(vmask) else 0.0
        ),
        "boundary_moved_points": 0,
        "max_boundary_displacement": 0.0,
        "t_feature_sec": feature_time,
        "t_inference_sec": inference_time,
        "t_smoothing_sec": smoothing_time,
        "t_revalidation_sec": revalidation_time,
        "fallback": bool(retet_info.get("fallback", False)),
        "growth": (
            float(retet_info.get("n_after", 0)) / max(retet_info.get("n_before", len(tets)), 1)
            if retet_info else float(out.n_cells) / max(len(tets), 1)
        ),
        "t_retet_sec": float(retet_info.get("t_retet", 0.0)),
    }
    if len(boundary_idxs):
        displacement = np.linalg.norm(new_pts[boundary_idxs] - points[boundary_idxs], axis=1)
        stats["boundary_moved_points"] = int(np.count_nonzero(displacement > 0.0))
        stats["max_boundary_displacement"] = float(np.max(displacement)) if len(displacement) else 0.0
    return out, stats