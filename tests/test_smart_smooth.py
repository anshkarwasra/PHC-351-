from itertools import combinations

import numpy as np

from src.meshing.quality import (
    FEATURE_NAMES,
    _element_geometry,
    _greedy_vertex_coloring,
    _boundary_vertices_and_elements,
    compute_tetrahedral_metrics,
    smart_smooth,
)
import scipy.sparse as sp
import pyvista as pv
import torch

from src.training.train_gnn import _rescale_geometric_features


def _two_tets():
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ])
    tets = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
    return points, tets


def test_smart_smooth_keeps_boundary_fixed_and_tets_valid():
    points, tets = _two_tets()
    boundary, _ = _boundary_vertices_and_elements(tets, len(points))

    result, stats = smart_smooth(points, tets, np.ones(len(points), bool), n_iter=2)

    assert np.array_equal(result[boundary], points[boundary])
    assert np.all(_element_geometry(result, tets)[0] > 0)
    assert len(stats["accepted_per_iteration"]) == 2
    assert len(stats["rejected_per_iteration"]) == 2


def test_poor_count_acceptance_never_increases_local_poor_element_count():
    points, tets = _two_tets()
    _, initial_sj, _ = _element_geometry(points, tets)
    initial_poor = np.count_nonzero(initial_sj < 0.25)

    result, stats = smart_smooth(
        points, tets, np.ones(len(points), bool), n_iter=2,
        acceptance="poor_count",
    )

    _, result_sj, _ = _element_geometry(result, tets)
    assert np.all(_element_geometry(result, tets)[0] > 0)
    assert np.count_nonzero(result_sj < 0.25) <= initial_poor
    assert len(stats["accepted_per_iteration"]) == 2
    assert len(stats["rejected_per_iteration"]) == 2


def test_poor_count_acceptance_accepts_a_move_that_reduces_poor_tets():
    outer = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3) / 2, 0.0],
        [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)],
    ])
    points = np.vstack((outer, np.array([0.4, 0.1, 0.1, 0.4]) @ outer))
    tets = []
    for face in combinations(range(4), 3):
        tet = np.array([*face, 4])
        if _element_geometry(points, tet[None, :])[0][0] < 0:
            tet[[0, 1]] = tet[[1, 0]]
        tets.append(tet)
    tets = np.asarray(tets)
    movable = np.array([False, False, False, False, True])
    initial_poor = np.count_nonzero(_element_geometry(points, tets)[1] < 0.25)

    result, stats = smart_smooth(
        points, tets, movable, n_iter=2, acceptance="poor_count",
    )
    final_volumes, final_sj, _ = _element_geometry(result, tets)

    assert initial_poor == 2
    assert np.count_nonzero(final_sj < 0.25) < initial_poor
    assert np.all(final_volumes > 0)
    assert stats["accepted_per_iteration"][0] == 1


def test_smart_smooth_rejects_unknown_acceptance_objective():
    points, tets = _two_tets()

    with np.testing.assert_raises_regex(ValueError, "acceptance must be"):
        smart_smooth(
            points, tets, np.ones(len(points), bool), n_iter=1,
            acceptance="unknown",
        )


def test_greedy_coloring_separates_adjacent_vertices():
    points, tets = _two_tets()
    edges = np.concatenate([tets[:, [a, b]] for a, b in
                            ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))])
    adjacency = sp.csr_matrix(
        (np.ones(len(edges) * 2),
         (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(points), len(points)),
    )
    colors = _greedy_vertex_coloring(adjacency)

    assert all(colors[a] != colors[b] for a, b in edges)


def test_regular_tetrahedron_has_scaled_jacobian_one():
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3) / 2, 0.0],
        [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)],
    ])

    assert np.isclose(_element_geometry(points, np.array([[0, 1, 2, 3]]))[1][0], 1.0)


def test_features_have_expected_shape_and_finite_values():
    points, tets = _two_tets()
    grid = pv.UnstructuredGrid(
        np.array([4, *tets[0]]), np.array([pv.CellType.TETRA]), points,
    )

    metrics = compute_tetrahedral_metrics(grid, include_labels=False)

    assert metrics["features"].shape == (1, len(FEATURE_NAMES))
    assert np.isfinite(metrics["features"]).all()
    assert not any(name.startswith("centre_") for name in FEATURE_NAMES)


def test_uniform_rescale_augmentation_matches_direct_mesh_features():
    points, tets = _two_tets()
    cells = np.column_stack((np.full(len(tets), 4), tets)).ravel()
    cell_types = np.full(len(tets), pv.CellType.TETRA)
    original = pv.UnstructuredGrid(cells, cell_types, points)
    scale = 2.0
    transformed = pv.UnstructuredGrid(cells, cell_types, points * scale)

    original_metrics = compute_tetrahedral_metrics(original)
    transformed_metrics = compute_tetrahedral_metrics(transformed)
    augmented = _rescale_geometric_features(
        torch.as_tensor(original_metrics["features"], dtype=torch.float32), scale
    ).numpy()

    assert np.allclose(augmented, transformed_metrics["features"], atol=1e-6)
    assert np.array_equal(
        original_metrics["needs_correction"], transformed_metrics["needs_correction"]
    )
    assert np.allclose(
        original_metrics["smoothing_target"], transformed_metrics["smoothing_target"]
    )
