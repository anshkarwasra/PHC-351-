"""Downstream FE check: vectorised Poisson assembly, reusing the validated solver.

Returns the keys used by scripts/evaluate.py:
  assembly_success, solver_converged, iterations, solver_time, assembly_time, residual, error
Inverted or non-tet cells make assembly_success False. They are counted, not dropped.
"""
import time
import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from src.validation.fe_checks import solve_dirichlet

GRAD_REF = np.array([[-1.0, -1.0, -1.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]])


def assemble_stiffness(points, tets):
    """Vectorised linear-tet Laplacian stiffness. Raises if any tet is inverted or degenerate."""
    p = points[tets]                                   # (n, 4, 3)
    J = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=-1)  # (n,3,3)
    detJ = np.linalg.det(J)
    if np.any(detJ <= 0):
        raise ValueError(f"{int(np.sum(detJ <= 0))} inverted or degenerate tets")
    invJT = np.linalg.inv(J).transpose(0, 2, 1)        # (n,3,3)
    grad = np.einsum("ij,mkj->mik", GRAD_REF, invJT)   # (n,4,3) physical gradients
    Ke = np.einsum("mia,mja->mij", grad, grad) * (detJ / 6.0)[:, None, None]
    I = np.repeat(tets, 4, axis=1).ravel()
    Jc = np.tile(tets, (1, 4)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (I, Jc)), shape=(len(points),) * 2).tocsr()
    return K


def _condition_estimate(K, fixed, max_dofs=2000):
    free = np.setdiff1d(np.arange(K.shape[0]), fixed)
    if not len(free) or len(free) > max_dofs:
        return None
    A = K[free][:, free].tocsr()
    diagonal = A.diagonal()
    if np.any(diagonal <= 0):
        return None
    scale = sp.diags(1.0 / np.sqrt(diagonal))
    scaled = (scale @ A @ scale).tocsr()
    try:
        largest = float(eigsh(scaled, k=1, which="LA", return_eigenvectors=False,
                              tol=1e-2, maxiter=100)[0])
        smallest = float(eigsh(scaled, k=1, which="SA", return_eigenvectors=False,
                               tol=1e-2, maxiter=100)[0])
    except ArpackNoConvergence:
        return None
    if smallest <= 0 or not np.isfinite(largest / smallest):
        return None
    return largest / smallest


def _uniform_source(points, tets):
    p = points[tets]
    det = np.einsum("ij,ij->i",
                    np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                    p[:, 3] - p[:, 0])
    volumes = det / 6.0
    source = np.zeros(len(points), dtype=np.float64)
    np.add.at(source, tets.ravel(), np.repeat(volumes / 4.0, 4))
    return source


def run_fe_sanity_check(grid, tol=1e-8, maxiter=2000):
    res = {
        "assembly_success": False, "solver_converged": False, "iterations": 0,
        "solver_time": 0.0, "assembly_time": 0.0, "residual": None,
        "condition_estimate": None, "fixed_left": 0, "fixed_right": 0,
        "heat_converged": False, "heat_iterations": 0, "heat_residual": None,
        "heat_solver_time": 0.0, "heat_condition_estimate": None, "error": "",
        "solution": None, "residual_history": [],
    }
    try:
        if grid.n_cells == 0 or not np.all(np.asarray(grid.celltypes) == pv.CellType.TETRA):
            raise ValueError("grid must be non-empty and contain only linear tets")
        t0 = time.perf_counter()
        pts = np.asarray(grid.points, dtype=np.float64)
        tets = np.asarray(grid.cells_dict[pv.CellType.TETRA], dtype=np.int64)
        K = assemble_stiffness(pts, tets)
        res["assembly_time"] = round(time.perf_counter() - t0, 4)
        res["assembly_success"] = True

        x = pts[:, 0]
        extent = float(x.max() - x.min())
        slab = 0.02 * extent
        left = x <= x.min() + slab
        right = x >= x.max() - slab
        fixed = np.flatnonzero(left | right)
        vals = np.where(right[fixed], 1.0, 0.0)
        res["fixed_left"] = int(left.sum())
        res["fixed_right"] = int(right.sum())
        res["condition_estimate"] = _condition_estimate(K, fixed)
        t1 = time.perf_counter()
        solution, info, iters, residuals = solve_dirichlet(
            K, pts, fixed, vals, tol=tol, maxiter=maxiter,
        )
        res["solver_time"] = round(time.perf_counter() - t1, 4)
        res["iterations"] = int(iters)
        res["solver_converged"] = bool(info == 0)
        res["residual"] = float(residuals[-1]) if residuals else None
        res["solution"] = solution
        res["residual_history"] = residuals

        source = _uniform_source(pts, tets)
        res["heat_condition_estimate"] = res["condition_estimate"]
        t2 = time.perf_counter()
        _, heat_info, heat_iters, heat_residuals = solve_dirichlet(
            K, pts, fixed, np.zeros(len(fixed)), tol=tol, maxiter=maxiter,
            source=source,
        )
        res["heat_solver_time"] = round(time.perf_counter() - t2, 4)
        res["heat_iterations"] = int(heat_iters)
        res["heat_converged"] = bool(heat_info == 0)
        res["heat_residual"] = float(heat_residuals[-1]) if heat_residuals else None
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return res