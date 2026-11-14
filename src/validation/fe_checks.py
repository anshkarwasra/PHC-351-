"""Step 2: the FE solver solves a known problem, then runs on a mesh.

Problem: Laplace equation on a unit cube. u = 0 on x=0, u = 1 on x=1, natural
(zero-flux) elsewhere. The exact solution is u = x, which is linear.
"""
import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from src.validation.report import CheckResult


def tet_stiffness(points, cells):
    """Assemble the Laplacian stiffness matrix for linear tets."""
    n = len(points)
    rows, cols, vals = [], [], []
    for c in cells:
        p = points[c]
        # gradients of the four linear shape functions
        J = np.column_stack([p[1] - p[0], p[2] - p[0], p[3] - p[0]])
        detJ = np.linalg.det(J)
        if abs(detJ) < 1e-14:
            continue
        invJ = np.linalg.inv(J)
        g = np.vstack([-invJ.sum(axis=0), invJ])   # (4, 3) gradients
        Ke = (g @ g.T) * abs(detJ) / 6.0
        for a in range(4):
            for b in range(4):
                rows.append(c[a]); cols.append(c[b]); vals.append(Ke[a, b])
    K = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return K


def solve_dirichlet(K, points, fixed_idx, fixed_vals, tol=1e-10, maxiter=2000,
                    report_every=20, log=None, source=None):
    """Eliminate Dirichlet nodes, solve the rest with Jacobi-preconditioned CG."""
    n = K.shape[0]
    free = np.setdiff1d(np.arange(n), fixed_idx)
    u = np.zeros(n)
    u[fixed_idx] = fixed_vals
    rhs = -K[free][:, fixed_idx] @ fixed_vals
    if source is not None:
        source = np.asarray(source, dtype=np.float64)
        if source.shape != (n,):
            raise ValueError("source vector must have one entry per node")
        rhs = rhs + source[free]
    A = K[free][:, free]
    M = sp.diags(1.0 / A.diagonal())
    residuals = []
    rhs_norm = max(float(np.linalg.norm(rhs)), 1e-30)

    def cb(xk):
        r = np.linalg.norm(rhs - A @ xk) / rhs_norm
        residuals.append(r)
        if log is not None and len(residuals) % report_every == 0:
            log(f"    iter {len(residuals)}: relative residual {r:.3e}")

    x, info = cg(A, rhs, rtol=tol, maxiter=maxiter, M=M, callback=cb)
    u[free] = x
    return u, info, len(residuals), residuals


def check_known_problem(report, step=2, n=4):
    """Unit cube meshed as a structured tet grid. Expect u(x) = x."""
    grid = pv.ImageData(dimensions=(n + 1,) * 3, spacing=(1.0 / n,) * 3).triangulate()
    pts = np.asarray(grid.points)
    cells = np.asarray(grid.cells_dict[pv.CellType.TETRA])
    K = tet_stiffness(pts, cells)
    left = np.where(np.isclose(pts[:, 0], 0.0))[0]
    right = np.where(np.isclose(pts[:, 0], 1.0))[0]
    fixed = np.concatenate([left, right])
    vals = np.concatenate([np.zeros(len(left)), np.ones(len(right))])
    u, info, iters, _ = solve_dirichlet(K, pts, fixed, vals)
    err = float(np.max(np.abs(u - pts[:, 0])))
    ok = info == 0 and err < 1e-6
    report.add(CheckResult(step, "fe_known_problem_linear", "pass" if ok else "fail",
                           err, 1e-6, f"cg info={info}, iters={iters}"))


def check_mesh_solve(grid, report, step=2, log=None):
    """Run the same solver on a real mesh with the same kind of boundary data."""
    pts = np.asarray(grid.points)
    cells = np.asarray(grid.cells_dict[pv.CellType.TETRA])
    K = tet_stiffness(pts, cells)
    x = pts[:, 0]
    extent = x.max() - x.min()
    slab = 0.02 * extent
    fixed = np.where((x <= x.min() + slab) | (x >= x.max() - slab))[0]
    vals = np.where(x[fixed] >= x.max() - slab, 1.0, 0.0)
    _, info, iters, res = solve_dirichlet(K, pts, fixed, vals, log=log)
    status = "pass" if info == 0 else "fail"
    report.add(CheckResult(step, "fe_mesh_converged", status, iters, 2000,
                           f"cg info={info}, final residual={res[-1]:.2e}" if res else ""))


def run(grid, report, step=2, log=print):
    check_known_problem(report, step)
    check_mesh_solve(grid, report, step, log=log)