"""Geometric features, quality metrics and training labels for tetrahedral meshes."""
from pathlib import Path

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import yaml

# Thresholds. Changing any of these changes the labels, so record them with each run.
SJ_BAD = 0.25          # scaled Jacobian below this is a poor element
SJ_FIXED = 0.25        # after smoothing, at or above this counts as repaired
SJ_TARGET = 0.30        # penalty target for quality-aware acceptance
ASPECT_BAD = 4.5       # edge-ratio flag, used only as a baseline (not a label)
VOL_EPS = 1e-12
TAUBIN_LAMBDA = 0.5
TAUBIN_MU = -0.53
N_SMOOTH = 8           # Quality-aware Taubin cycles used for labels and correction
LABEL_VERSION = "csg-smart-smooth-boundary-penalty-n8-sj0.30-v4"

DEFAULT_CORRECTION_CONFIG = {
    "operator": "smart_smooth",
    "allow_boundary": True,
    "acceptance": "penalty",
    "sj_target": 0.30,
    "floor_ratio": 0.5,
    "n_iter": N_SMOOTH,
    "gnn_prob_threshold": None,
}


def get_correction_config(config=None):
    """Resolve the shared operator settings from the project config or explicit values."""
    settings = dict(DEFAULT_CORRECTION_CONFIG)
    if config is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "synthetic.yaml"
        with config_path.open("r", encoding="utf-8") as source:
            project_config = yaml.safe_load(source) or {}
        config = project_config.get("correction", {})
    settings.update(config or {})
    if settings["operator"] != "smart_smooth":
        raise ValueError(f"unsupported correction operator: {settings['operator']!r}")
    if settings["acceptance"] not in ("min_sj", "poor_count", "penalty"):
        raise ValueError(f"unsupported correction acceptance: {settings['acceptance']!r}")
    if int(settings["n_iter"]) < 0:
        raise ValueError("correction n_iter must be non-negative")
    return settings

FEATURE_NAMES = [
    "scaled_jacobian", "edge_ratio_norm", "log10_volume",
    "log10_lmax", "log10_lmin", "log10_lrms",
    "min_dihedral", "max_dihedral", "neighbor_mean_sj", "neighbor_min_sj",
    "neighbor_max_sj", "boundary_hops", "volume_neighbor_ratio",
    "edge_neighbor_ratio", "touches_boundary",
]

_EDGE_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
_FACE_IDX = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])


def get_tets(grid):
    """Return (n, 4) int64 connectivity. Raises if the grid holds anything but linear tets."""
    if grid.n_cells == 0:
        raise ValueError("grid has no cells")
    if not np.all(np.asarray(grid.celltypes) == pv.CellType.TETRA):
        raise ValueError("grid must contain only linear tetrahedra")
    return np.asarray(grid.cells_dict[pv.CellType.TETRA], dtype=np.int64)


def _face_table(tets):
    """All 4n faces (node-sorted), the owning element of each, and how many tets share each."""
    n = len(tets)
    faces = np.sort(tets[:, _FACE_IDX].reshape(-1, 3), axis=1)
    owner = np.repeat(np.arange(n), 4)
    _, inv, counts = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    return faces, owner, inv, counts[inv]


def _boundary_vertices_and_elements(tets, n_points, face_table=None):
    """Boundary = faces owned by exactly one tet. Returns vertex mask and element mask."""
    faces, owner, _, row_count = _face_table(tets) if face_table is None else face_table
    bf = row_count == 1
    bv = np.zeros(n_points, dtype=bool)
    bv[faces[bf].ravel()] = True
    be = np.zeros(len(tets), dtype=bool)
    be[owner[bf]] = True
    return bv, be


def build_element_dual_graph(tets, face_table=None):
    """Dual graph: nodes are tets, edges join tets sharing a face. Returns (2, E), both directions.

    Only faces shared by exactly two tets produce edges. Faces with one owner (boundary)
    or three or more owners (non-manifold) produce none.
    """
    tets = np.asarray(tets, dtype=np.int64)
    _, owner, inv, row_count = _face_table(tets) if face_table is None else face_table
    order = np.argsort(inv, kind="stable")
    s_inv = inv[order]
    s_own = owner[order]
    s_cnt = row_count[order]
    pair = (s_inv[:-1] == s_inv[1:]) & (s_cnt[:-1] == 2)
    u = s_own[:-1][pair]
    v = s_own[1:][pair]
    return np.stack([np.concatenate([u, v]), np.concatenate([v, u])]).astype(np.int64)


def _element_geometry_from_tet_points(p):
    """Signed volume, scaled Jacobian and six edge lengths from (n, 4, 3) coordinates."""
    det0 = np.einsum("ij,ij->i",
                     np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                     p[:, 3] - p[:, 0])
    vol = det0 / 6.0

    # At corner k, the edges to the other three nodes give a Jacobian whose determinant
    # equals det0 (the sign alternation cancels in the ordering). Normalise by the
    # product of those three edge lengths; a regular tet gives exactly 1.
    corner_prod = np.empty((len(p), 4))
    for k in range(4):
        others = [j for j in range(4) if j != k]
        L = np.stack([np.linalg.norm(p[:, j] - p[:, k], axis=1) for j in others], axis=1)
        corner_prod[:, k] = np.prod(L, axis=1)
    sj_corner = np.sqrt(2.0) * det0[:, None] / np.maximum(corner_prod, VOL_EPS)
    sj = np.clip(sj_corner.min(axis=1), -1.0, 1.0)

    lengths = np.stack([np.linalg.norm(p[:, b] - p[:, a], axis=1)
                        for a, b in _EDGE_PAIRS], axis=1)
    return vol, sj, lengths


def _element_geometry(points, tets):
    """Signed volume, scaled Jacobian (min over 4 corners), and six edge lengths."""
    return _element_geometry_from_tet_points(points[tets])


def _greedy_vertex_coloring(adjacency):
    """Greedy coloring of the vertex graph; adjacent vertices get distinct colors."""
    n = adjacency.shape[0]
    colors = np.full(n, -1, dtype=np.int64)
    order = np.argsort(-np.diff(adjacency.indptr), kind="stable")
    for vertex in order:
        neighbors = adjacency.indices[adjacency.indptr[vertex]:adjacency.indptr[vertex + 1]]
        used = set(colors[neighbors][colors[neighbors] >= 0])
        color = 0
        while color in used:
            color += 1
        colors[vertex] = color
    return colors


def smart_smooth(points, tets, vertex_mask, n_iter, allow_boundary=False,
                 boundary_tol=1e-3, vertex_weights=None, face_table=None,
                 acceptance="min_sj", floor_ratio=0.5, sj_target=SJ_TARGET):
    """Greedily improve selected vertices using a local mesh-quality objective."""
    points = np.asarray(points, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    vertex_mask = np.asarray(vertex_mask, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError("tets must have shape (m, 4)")
    if len(vertex_mask) != len(points):
        raise ValueError("vertex_mask length must match points")
    if n_iter < 0 or boundary_tol < 0:
        raise ValueError("n_iter and boundary_tol must be non-negative")
    if acceptance not in ("min_sj", "poor_count", "penalty"):
        raise ValueError("acceptance must be 'min_sj', 'poor_count', or 'penalty'")
    if floor_ratio < 0.0:
        raise ValueError("floor_ratio must be non-negative")
    if vertex_weights is None:
        vertex_weights = np.ones(len(points), dtype=np.float64)
    else:
        vertex_weights = np.clip(np.asarray(vertex_weights, dtype=np.float64), 0.0, 1.0)
        if vertex_weights.shape != (len(points),):
            raise ValueError("vertex_weights length must match points")

    n = len(points)
    edges = np.concatenate([tets[:, [a, b]] for a, b in _EDGE_PAIRS])
    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    weights = 1.0 / np.maximum(lengths, 1e-12)
    adjacency = sp.csr_matrix(
        (weights, (edges[:, 0], edges[:, 1])), shape=(n, n)
    )
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    colors = _greedy_vertex_coloring(adjacency)

    incident = sp.csr_matrix(
        (np.ones(tets.size), (tets.ravel(), np.repeat(np.arange(len(tets)), 4))),
        shape=(n, len(tets)),
    )
    incident_counts = np.diff(incident.indptr)
    max_incident = int(incident_counts.max(initial=0))
    padded_incident = np.full((n, max_incident), -1, dtype=np.int64)
    for vertex in np.flatnonzero(incident_counts):
        start, stop = incident.indptr[vertex:vertex + 2]
        padded_incident[vertex, :stop - start] = incident.indices[start:stop]
    boundary, _ = _boundary_vertices_and_elements(tets, n, face_table)
    movable = vertex_mask & (degree > 0)
    if not allow_boundary:
        movable &= ~boundary

    boundary_normals = np.zeros((n, 3), dtype=np.float64)
    if allow_boundary:
        _, owner, _, counts = _face_table(tets) if face_table is None else face_table
        oriented_faces = tets[:, _FACE_IDX].reshape(-1, 3)
        boundary_faces = oriented_faces[counts == 1]
        boundary_owners = owner[counts == 1]
        tri = points[boundary_faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        face_centres = tri.mean(axis=1)
        tet_centres = points[tets[boundary_owners]].mean(axis=1)
        flip = np.einsum("ij,ij->i", normals, face_centres - tet_centres) < 0
        normals[flip] *= -1
        for corner in range(3):
            np.add.at(boundary_normals, boundary_faces[:, corner], normals)
        norms = np.linalg.norm(boundary_normals, axis=1)
        valid = boundary & (norms > 0)
        boundary_normals[valid] /= norms[valid, None]
        movable &= ~boundary | valid

    result = points.copy()
    accepted_per_iteration = []
    rejected_per_iteration = []
    accepted_scale = np.zeros(n, dtype=np.float64)
    accepted_count = np.zeros(n, dtype=np.int64)
    max_color = int(colors.max()) + 1 if n else 0
    for _ in range(n_iter):
        accepted = rejected = 0
        for color in range(max_color):
            vertices = np.flatnonzero(movable & (colors == color))
            if not len(vertices):
                continue
            target = (adjacency[vertices] @ result) / degree[vertices, None]
            delta = (target - result[vertices]) * vertex_weights[vertices, None]
            boundary_vertices = boundary[vertices]
            if np.any(boundary_vertices):
                normals = boundary_normals[vertices[boundary_vertices]]
                tangent = delta[boundary_vertices]
                tangent -= np.einsum("ij,ij->i", tangent, normals)[:, None] * normals
                delta[boundary_vertices] = tangent
            active = np.any(delta != 0, axis=1)
            if not np.any(active):
                continue

            cell_ids = padded_incident[vertices]
            valid_cells = cell_ids >= 0
            cell_vertices = tets[np.maximum(cell_ids, 0)]
            local_points = result[cell_vertices]
            _, old_sj, _ = _element_geometry_from_tet_points(
                local_points.reshape(-1, 4, 3)
            )
            old_sj = old_sj.reshape(cell_ids.shape)
            old_min_sj = np.where(valid_cells, old_sj, np.inf).min(axis=1)
            old_poor_count = np.sum(
                valid_cells & (old_sj < SJ_BAD), axis=1,
            )
            old_penalty = np.sum(
                np.maximum(sj_target - np.where(valid_cells, old_sj, 0.0), 0.0) ** 2,
                axis=1,
            )

            for scale in (1.0, 0.5, 0.25):
                indices = np.flatnonzero(active)
                if not len(indices):
                    break
                allowed = np.ones(len(indices), dtype=bool)
                is_boundary = boundary_vertices[indices]
                displacement = delta[indices] * scale
                if np.any(is_boundary):
                    allowed[is_boundary] = (
                        np.linalg.norm(displacement[is_boundary], axis=1) < boundary_tol
                    )
                indices = indices[allowed]
                if not len(indices):
                    continue

                trial = local_points[indices].copy()
                moved_coordinates = (
                    cell_vertices[indices] == vertices[indices, None, None]
                )
                trial += moved_coordinates[..., None] * (delta[indices] * scale)[:, None, None, :]
                volumes, sj, _ = _element_geometry_from_tet_points(trial.reshape(-1, 4, 3))
                volumes = volumes.reshape(len(indices), -1)
                sj = sj.reshape(len(indices), -1)
                local_valid = valid_cells[indices]
                all_positive = np.all((volumes > VOL_EPS) | ~local_valid, axis=1)
                min_sj = np.where(local_valid, sj, np.inf).min(axis=1)
                poor_count = np.sum(local_valid & (sj < SJ_BAD), axis=1)
                penalty = np.sum(
                    np.maximum(sj_target - np.where(local_valid, sj, 0.0), 0.0) ** 2,
                    axis=1,
                )
                if acceptance == "poor_count":
                    objective_improved = poor_count < old_poor_count[indices]
                elif acceptance == "penalty":
                    objective_improved = (
                        poor_count <= old_poor_count[indices]
                        ) & (penalty < old_penalty[indices]) & (
                            min_sj >= floor_ratio * old_min_sj[indices]
                        )
                else:
                    objective_improved = min_sj > old_min_sj[indices]
                newly_accepted = all_positive & objective_improved
                if np.any(newly_accepted):
                    selected = indices[newly_accepted]
                    result[vertices[selected]] += delta[selected] * scale
                    accepted_scale[vertices[selected]] += scale * vertex_weights[vertices[selected]]
                    accepted_count[vertices[selected]] += 1
                    active[selected] = False
                    accepted += int(newly_accepted.sum())
            rejected += int(active.sum())
        accepted_per_iteration.append(accepted)
        rejected_per_iteration.append(rejected)

    return result, {
        "accepted_per_iteration": accepted_per_iteration,
        "rejected_per_iteration": rejected_per_iteration,
        "vertex_colors": colors,
        "accepted_scale": accepted_scale / np.maximum(accepted_count, 1),
        "accepted_count": accepted_count,
    }


def _tet_dihedral_angles(points, tets):
    tet_points = points[tets]
    angles = []
    for a, b, c, d in (
        (0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2),
        (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1),
    ):
        edge = tet_points[:, b] - tet_points[:, a]
        edge /= np.maximum(np.linalg.norm(edge, axis=1), 1e-12)[:, None]
        u = tet_points[:, c] - tet_points[:, a]
        v = tet_points[:, d] - tet_points[:, a]
        u -= np.einsum("ij,ij->i", u, edge)[:, None] * edge
        v -= np.einsum("ij,ij->i", v, edge)[:, None] * edge
        cosine = np.einsum("ij,ij->i", u, v)
        cosine /= np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-12)
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return np.column_stack(angles)


def _neighbor_statistics(values, edge_index):
    n = len(values)
    source, target = edge_index
    counts = np.bincount(source, minlength=n)
    sums = np.bincount(source, weights=values[target], minlength=n)
    means = sums / np.maximum(counts, 1)
    mins = np.full(n, np.inf)
    maxs = np.full(n, -np.inf)
    np.minimum.at(mins, source, values[target])
    np.maximum.at(maxs, source, values[target])
    isolated = counts == 0
    mins[isolated] = values[isolated]
    maxs[isolated] = values[isolated]
    means[isolated] = values[isolated]
    return means, mins, maxs


def compute_tetrahedral_metrics(
    grid, face_table=None, include_labels=True, correction_config=None,
):
    """Per-element features, geometric baseline flag, and training labels.

    Returns a dict. Keys used by training:
      features (n, 10) float32, tets (n, 4), needs_correction (n,) int64,
      smoothing_target (n,) float32.
    Labels are generated by the same locally selected, quality-guarded Taubin
    correction used at inference. The classifier identifies poor elements that
    improve under that action; the regressor learns accepted smoothing strength.
    """
    points = np.asarray(grid.points, dtype=np.float64)
    tets = get_tets(grid)
    if face_table is None:
        face_table = _face_table(tets)

    vol, sj, lengths = _element_geometry(points, tets)
    lmax = lengths.max(axis=1)
    lmin = np.maximum(lengths.min(axis=1), 1e-8)
    lrms = np.sqrt((lengths ** 2).mean(axis=1))
    aspect = lmax / lmin
    bv, touches_boundary = _boundary_vertices_and_elements(tets, len(points), face_table)
    edge_index = build_element_dual_graph(tets, face_table)
    neighbor_mean_sj, neighbor_min_sj, neighbor_max_sj = _neighbor_statistics(sj, edge_index)
    volume_mean, _, _ = _neighbor_statistics(np.abs(vol), edge_index)
    edge_mean, _, _ = _neighbor_statistics(lrms, edge_index)
    dihedral = _tet_dihedral_angles(points, tets)
    dual = sp.csr_matrix(
        (np.ones(edge_index.shape[1]), edge_index),
        shape=(len(tets), len(tets)),
    )
    boundary_hops = np.zeros(len(tets), dtype=np.float64)
    boundary_ids = np.flatnonzero(touches_boundary)
    if len(boundary_ids):
        boundary_hops = sp.csgraph.dijkstra(
            dual, directed=False, indices=boundary_ids, min_only=True, unweighted=True,
        )

    features = np.column_stack([
        sj,
        np.clip(aspect, 1.0, 20.0) / 20.0,
        np.log10(np.maximum(np.abs(vol), VOL_EPS)),
        np.log10(np.maximum(lmax, 1e-8)),
        np.log10(lmin),
        np.log10(np.maximum(lrms, 1e-8)),
        dihedral.min(axis=1),
        dihedral.max(axis=1),
        neighbor_mean_sj,
        neighbor_min_sj,
        neighbor_max_sj,
        boundary_hops,
        np.abs(vol) / np.maximum(volume_mean, VOL_EPS),
        lrms / np.maximum(edge_mean, 1e-8),
        touches_boundary.astype(np.float64),
    ]).astype(np.float32)

    bad_before = (sj < SJ_BAD) | (vol <= VOL_EPS)
    geometric_flag = bad_before | (aspect > ASPECT_BAD)

    metrics = {
        "features": features,
        "feature_names": FEATURE_NAMES,
        "tets": tets,
        "scaled_jacobian": sj,
        "aspect_ratio": aspect,
        "volumes": vol,
        "boundary_element": touches_boundary,
        "boundary_vertex": bv,
        "geometric_flag": geometric_flag.astype(np.int64),
        "label_version": LABEL_VERSION,
    }
    if not include_labels:
        return metrics

    correction = get_correction_config(correction_config)
    candidate_vertices = np.zeros(len(points), dtype=bool)
    candidate_vertices[np.unique(tets[bad_before].ravel())] = True
    smoothed, smooth_stats = smart_smooth(
        points, tets, candidate_vertices, int(correction["n_iter"]),
        allow_boundary=bool(correction["allow_boundary"]),
        face_table=face_table,
        acceptance=str(correction["acceptance"]),
        floor_ratio=float(correction["floor_ratio"]),
        sj_target=float(correction["sj_target"]),
    )
    vol_a, sj_a, _ = _element_geometry(smoothed, tets)
    needs_correction = bad_before & (sj_a > sj + 1e-8) & (vol_a > VOL_EPS)
    smoothing_target = np.zeros(len(tets), dtype=np.float32)
    accepted_strength = smooth_stats["accepted_scale"]
    smoothing_target[needs_correction] = accepted_strength[tets[needs_correction]].mean(axis=1)
    metrics.update(
        smoothed_scaled_jacobian=sj_a,
        needs_correction=needs_correction.astype(np.int64),
        smoothing_target=smoothing_target,
    )
    return metrics