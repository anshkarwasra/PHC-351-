"""Step 3: correction moves never create inverted tets.

Works on a grid and a proposed displacement. Counts rejected moves and verifies
no tet is inverted after the accepted moves.
"""
import numpy as np
import pyvista as pv

from src.validation.report import CheckResult


def signed_volumes(points, cells):
    p = points[cells]
    return np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                     p[:, 3] - p[:, 0])


def apply_with_rejection(points, cells, vertex_ids, displacement, max_halvings=4):
    """Move vertex_ids by displacement; halve the step until no tet inverts.

    Returns new points, number of accepted moves, number of rejected moves.
    """
    new = points.copy()
    accepted = rejected = 0
    adj = {v: np.where((cells == v).any(axis=1))[0] for v in vertex_ids}
    for v in vertex_ids:
        step = displacement[v]
        ok = False
        for _ in range(max_halvings + 1):
            trial = new.copy()
            trial[v] = new[v] + step
            if (signed_volumes(trial, cells[adj[v]]) > 0).all():
                new = trial
                ok = True
                break
            step = step * 0.5
        accepted += ok
        rejected += (not ok)
    return new, accepted, rejected


def check_rejection_works(report, step=3):
    """Synthetic test: a deliberately inverting move must be rejected, and the
    mesh must stay valid."""
    grid = pv.ImageData(dimensions=(3, 3, 3), spacing=(1.0, 1.0, 1.0)).triangulate()
    pts = np.asarray(grid.points, dtype=float)
    cells = np.asarray(grid.cells_dict[pv.CellType.TETRA])
    interior = np.array([13])   # centre vertex of a 3x3x3 grid
    disp = {13: np.array([10.0, 0.0, 0.0])}   # would pass through neighbours
    new, acc, rej = apply_with_rejection(pts, cells, interior, disp)
    valid = (signed_volumes(new, cells) > 0).all()
    report.add(CheckResult(step, "rejection_keeps_mesh_valid", "pass" if valid else "fail",
                           bool(valid), True, f"accepted={acc}, rejected={rej}"))


def check_correction_output(before_grid, after_grid, report, step=3):
    """For a real correction run: no inverted tets after, and log rejections."""
    cells_after = np.asarray(after_grid.cells_dict[pv.CellType.TETRA])
    n_neg = int((signed_volumes(np.asarray(after_grid.points), cells_after) <= 0).sum())
    report.add(CheckResult(step, "no_inverted_after_correction",
                           "pass" if n_neg == 0 else "fail", n_neg, 0))


def run(report, before_grid=None, after_grid=None, step=3):
    check_rejection_works(report, step)
    if before_grid is not None and after_grid is not None:
        check_correction_output(before_grid, after_grid, report, step)