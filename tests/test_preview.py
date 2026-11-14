import numpy as np
import pyvista as pv
import trimesh

from src.utils.preview import (
    render_volume, render_segmentation, render_surface, render_repair,
    render_tetrahedral_quality, render_gnn_flags, render_correction,
    render_geometry, render_fe,
)


def test_preview_stages_write_images_or_log_a_skip(tmp_path):
    axis = np.linspace(-1.0, 1.0, 32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    volume = ((x * x + y * y + z * z) < 0.6).astype(np.float32)
    mask = volume > 0.5
    surface = trimesh.creation.icosphere(subdivisions=1)
    points = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    ])
    grid = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3]),
        np.array([pv.CellType.TETRA]),
        points,
    )
    fe_result = {"assembly_success": True, "solver_converged": True}
    stages = [
        lambda path, logger: render_volume(
            path, volume, volume, "clean", {"blur": 0.0, "noise": 0.0}, 0.5, logger,
        ),
        lambda path, logger: render_segmentation(
            path, volume, mask, mask, "clean",
        ),
        lambda path, logger: render_surface(path, surface, logger),
        lambda path, logger: render_repair(path, surface, surface, {}, logger),
        lambda path, logger: render_tetrahedral_quality(path, grid, logger),
        lambda path, logger: render_gnn_flags(path, grid, np.array([0.8]), 0.5, logger),
        lambda path, logger: render_correction(
            path, grid, grid, {"moved_points": 0, "modified_elements": 0}, logger,
        ),
        lambda path, logger: render_geometry(path, grid, surface, logger),
        lambda path, logger: render_fe(
            path, grid, np.arange(4), [1.0, 0.1, 0.01], fe_result, logger,
        ),
    ]

    for index, render in enumerate(stages):
        path = tmp_path / f"stage_{index}.png"
        reasons = []
        written = render(path, reasons.append)
        if written:
            assert path.is_file()
            assert path.stat().st_size > 0
        else:
            assert reasons
