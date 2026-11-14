import numpy as np
import pyvista as pv

import src.meshing.correction as correction
from src.meshing.correction import correct_mesh


def _single_tet_grid():
    points = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    ])
    cells = np.array([4, 0, 1, 2, 3])
    return pv.UnstructuredGrid(cells, np.array([pv.CellType.TETRA]), points)


def test_targeted_methods_are_model_free():
    grid = _single_tet_grid()

    for method in ("rule_targeted", "blind_global_smart"):
        corrected, stats = correct_mesh(grid, method=method)
        assert corrected.n_cells == 1
        assert stats["flagged_cells"] == 0
        assert stats["t_smoothing_sec"] >= 0


def test_removed_method_names_raise_v3_error():
    try:
        correct_mesh(_single_tet_grid(), method="gnn_penalty")
    except ValueError as exc:
        assert "removed in V3" in str(exc)
    else:
        raise AssertionError("removed method name must be rejected")


def test_quality_retet_uses_explicit_mesh_identity_without_grid_name(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    repaired_dir = tmp_path / "data" / "processed"
    repaired_dir.mkdir(parents=True)
    tetra_surface = correction.trimesh.Trimesh(
        vertices=np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ]),
        faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
        process=True,
    )
    tetra_surface.export(repaired_dir / "0000_clean.stl")
    captured = {}

    def fake_retet(surface, params):
        assert isinstance(surface, correction.trimesh.Trimesh)
        assert surface.is_watertight
        captured["params"] = params
        return None, {"n_before": 1, "n_after": 0, "t_retet": 0.0, "fallback": True}

    monkeypatch.setattr(correction, "retet_from_surface", fake_retet)
    grid = _single_tet_grid()
    result, stats = correct_mesh(
        grid, method="retet_global_quality", mesh_id="0000", tier="clean",
    )
    assert result is grid
    assert captured["params"]["n_before"] == grid.n_cells
    assert captured["params"]["mindihedral"] == 10
    assert captured["params"]["minratio"] == 1.5
    assert stats["fallback"]


def test_retet_timeout_worker_runs_in_subprocess_without_multiprocessing_spawn():
    surface = correction.trimesh.Trimesh(
        vertices=np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ]),
        faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
        process=True,
    )
    retet, info = correction.retet_from_surface(
        surface, {"timeout": 30, "max_growth": 10, "n_before": 10},
    )
    assert retet is not None, info
    assert retet.n_cells > 0
    assert info["fallback"] is False


def test_gnn_boundary_setting_changes_boundary_moves(monkeypatch):
    grid = _single_tet_grid()
    observed = []

    def fake_mask(grid, model, tets, points, face_table, **kwargs):
        return np.ones(len(points), dtype=bool), 1, np.ones(len(points)), 0.0, 0.0

    def fake_smooth(points, tets, vertex_mask, n_iter, **kwargs):
        result = points.copy()
        if kwargs["allow_boundary"]:
            result[0, 0] += 0.1
        observed.append(kwargs["allow_boundary"])
        return result, {"rejected_per_iteration": [0] * n_iter}

    monkeypatch.setattr(correction, "_gnn_vertex_mask", fake_mask)
    monkeypatch.setattr(correction, "smart_smooth", fake_smooth)
    interior_only, _ = correct_mesh(
        grid, model=object(), method="gnn",
        correction_config={"allow_boundary": False},
    )
    boundary_tangential, _ = correct_mesh(
        grid, model=object(), method="gnn",
        correction_config={"allow_boundary": True},
    )
    assert observed == [False, True]
    assert not np.array_equal(interior_only.points, boundary_tangential.points)
