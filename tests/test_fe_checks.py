import numpy as np
import pyvista as pv

from src.simulation.fe_solver import run_fe_sanity_check


def test_fe_sanity_check_reports_slab_heat_and_condition_fields():
    grid = pv.ImageData(dimensions=(4, 4, 4), spacing=(1 / 3, 1 / 3, 1 / 3))
    grid = grid.triangulate()

    result = run_fe_sanity_check(grid, maxiter=300)

    assert result["assembly_success"]
    assert result["fixed_left"] > 0
    assert result["fixed_right"] > 0
    assert result["residual"] is None or np.isfinite(result["residual"])
    assert result["heat_converged"]
    assert result["heat_iterations"] > 0
