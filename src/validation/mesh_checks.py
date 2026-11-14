"""Step 1: the classical tet mesh is valid.

Checks signed volumes, scaled Jacobian agreement, and the quality distribution.
"""
import numpy as np
import pyvista as pv

from src.validation.report import CheckResult

MIN_JACOBIAN_TYPICAL = 0.1   # median-ish tets should be well above this
SLIVER_FRACTION_MAX = 0.05   # fraction of tets with scaled Jacobian < 0.05


def scaled_jacobian_manual(points, cell):
    """Scaled Jacobian of a tet, computed from its corners.

    Uses the determinant of the edge matrix at each corner, normalised by the
    edge lengths. Independent of PyVista's implementation, so it can disagree.
    """
    p = points[cell]                       # (4, 3)
    v = np.stack([p[1] - p[0], p[2] - p[0], p[3] - p[0]])
    signed_vol6 = np.linalg.det(v)         # six times the signed volume
    corner_products = []
    for corner in range(4):
        edge_lengths = [
            np.linalg.norm(p[other] - p[corner])
            for other in range(4)
            if other != corner
        ]
        corner_products.append(np.prod(edge_lengths))
    return float(np.min(
        np.sqrt(2) * signed_vol6 / np.maximum(corner_products, 1e-12)
    ))


def check_signed_volumes(grid, report, step=1):
    pts = np.asarray(grid.points)
    cells = np.asarray(grid.cells_dict[pv.CellType.TETRA])
    p = pts[cells]
    vol6 = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                     p[:, 3] - p[:, 0])
    n_neg = int((vol6 <= 0).sum())
    report.add(CheckResult(step, "no_inverted_tets", "pass" if n_neg == 0 else "fail",
                           n_neg, 0, f"{len(cells)} tets"))
    return cells, vol6


def check_jacobian_agreement(grid, cells, report, step=1, n_sample=3, seed=0):
    """Compare manual scaled Jacobian with PyVista's on random tets."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cells), size=min(n_sample, len(cells)), replace=False)
    q_pv = np.asarray(grid.cell_quality(quality_measure="scaled_jacobian")
                      ["scaled_jacobian"])
    worst = 0.0
    for i in pick:
        manual = scaled_jacobian_manual(np.asarray(grid.points), cells[i])
        worst = max(worst, abs(manual - q_pv[i]))
    status = "pass" if worst < 1e-3 else "fail"
    report.add(CheckResult(step, "jacobian_agreement", status, float(worst), 1e-3,
                           f"sampled {len(pick)} tets"))


def check_quality_distribution(grid, report, step=1):
    q = np.asarray(grid.cell_quality(quality_measure="scaled_jacobian")
                   ["scaled_jacobian"])
    median = float(np.median(q))
    sliver_frac = float((q < 0.05).mean())
    report.add(CheckResult(step, "median_scaled_jacobian",
                           "pass" if median >= MIN_JACOBIAN_TYPICAL else "warn",
                           median, MIN_JACOBIAN_TYPICAL))
    report.add(CheckResult(step, "sliver_fraction",
                           "pass" if sliver_frac <= SLIVER_FRACTION_MAX else "warn",
                           sliver_frac, SLIVER_FRACTION_MAX))
    report.add(CheckResult(step, "min_scaled_jacobian",
                           "pass" if q.min() > 0 else "fail",
                           float(q.min()), 0.0,
                           "min must be positive for a valid mesh"))


def run(grid, report, step=1):
    cells, _ = check_signed_volumes(grid, report, step)
    check_jacobian_agreement(grid, cells, report, step)
    check_quality_distribution(grid, report, step)