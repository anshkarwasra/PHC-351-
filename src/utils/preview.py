"""Rendering helpers for cached, stage-by-stage pipeline previews."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


def _log(logger, message):
    if logger is not None:
        logger(message)


def _surface_data(mesh):
    if isinstance(mesh, pv.DataSet):
        surface = mesh.extract_surface().triangulate()
        return np.asarray(surface.points), np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def _pv_surface(mesh):
    if isinstance(mesh, pv.DataSet):
        return mesh.extract_surface().triangulate()
    points, faces = _surface_data(mesh)
    cells = np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces)).ravel()
    return pv.PolyData(points, cells)


def _fallback_surface(path, mesh, title, logger):
    points, faces = _surface_data(mesh)
    if len(points) > 5000:
        selected = np.linspace(0, len(points) - 1, 5000, dtype=np.int64)
        points = points[selected]
        faces = np.empty((0, 3), dtype=np.int64)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if len(faces):
        ax.plot_trisurf(
            points[:, 0], points[:, 1], points[:, 2],
            triangles=faces, color="steelblue", linewidth=0.05, alpha=0.85,
        )
    else:
        ax.scatter(*points.T, s=1, c="steelblue")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    _log(logger, f"saved Matplotlib fallback: {path}")


def _render_3d(path, title, plot, fallback, logger):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pv.OFF_SCREEN = True
    try:
        with pv.Plotter(off_screen=True, window_size=(1200, 900)) as plotter:
            plot(plotter)
            plotter.add_title(title)
            plotter.screenshot(str(path))
        return path.is_file() and path.stat().st_size > 0
    except Exception as exc:
        _log(logger, f"{title}: PyVista render failed ({type(exc).__name__}: {exc}); using fallback.")
        try:
            fallback(path)
            return path.is_file() and path.stat().st_size > 0
        except Exception as fallback_exc:
            _log(
                logger,
                f"{title}: fallback skipped ({type(fallback_exc).__name__}: {fallback_exc}).",
            )
            return False


def _save_figure(path, fig):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path.is_file() and path.stat().st_size > 0


def render_volume(path, clean, degraded, tier, settings, threshold, logger=print):
    clean = np.asarray(clean)
    degraded = np.asarray(degraded)
    clean_center = tuple(size // 2 for size in clean.shape)
    degraded_center = tuple(size // 2 for size in degraded.shape)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for column, name in enumerate(("axial", "coronal", "sagittal")):
        if column == 0:
            clean_slice = clean[:, :, clean_center[2]]
            degraded_slice = degraded[:, :, degraded_center[2]]
        elif column == 1:
            clean_slice = clean[:, clean_center[1], :]
            degraded_slice = degraded[:, degraded_center[1], :]
        else:
            clean_slice = clean[clean_center[0], :, :]
            degraded_slice = degraded[degraded_center[0], :, :]
        axes[0, column].imshow(clean_slice.T, cmap="gray", origin="lower")
        axes[1, column].imshow(degraded_slice.T, cmap="gray", origin="lower")
        axes[0, column].set_title(f"Clean - {name}")
        axes[1, column].set_title(f"{tier} - {name}")
        for ax in axes[:, column]:
            ax.axis("off")
    fig.suptitle(
        f"tier={tier}; blur={settings.get('blur')}; noise={settings.get('noise')}; "
        f"threshold={threshold}"
    )
    return _save_figure(path, fig)


def render_segmentation(path, volume, mask, reference_mask, tier):
    axis = 2
    index = volume.shape[axis] // 2
    vol_slice = np.take(volume, index, axis=axis)
    mask_slice = np.take(mask, index, axis=axis)
    ref_slice = np.take(reference_mask, index, axis=axis)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(vol_slice.T, cmap="gray", origin="lower")
    axes[0].set_title("Input")
    axes[1].imshow(vol_slice.T, cmap="gray", origin="lower")
    axes[1].imshow(mask_slice.T, cmap="Reds", alpha=0.4, origin="lower")
    axes[1].set_title("Segmentation")
    axes[2].imshow((mask_slice != ref_slice).T, cmap="magma", origin="lower")
    axes[2].set_title("Mask vs reference")
    for ax in axes:
        ax.axis("off")
    components = int(np.count_nonzero(mask))
    fig.suptitle(f"{tier}: mask voxels={components:,}")
    return _save_figure(path, fig)


def render_surface(path, surface, logger=print):
    return _render_3d(
        path, "Marching-cubes surface",
        lambda plotter: plotter.add_mesh(
            _pv_surface(surface), color="steelblue", show_edges=True,
        ),
        lambda output: _fallback_surface(output, surface, "Marching-cubes surface", logger),
        logger,
    )


def render_repair(path, before, after, repair_report, logger=print):
    def plot(plotter):
        plotter.add_mesh(_pv_surface(before), color="tomato", opacity=0.45, show_edges=True)
        plotter.add_mesh(_pv_surface(after), color="seagreen", opacity=0.75)

    return _render_3d(
        path, f"Surface repair: holes filled={repair_report.get('n_loop_faces', 0)}",
        plot,
        lambda output: _fallback_surface(output, after, "Repaired surface", logger),
        logger,
    )


def render_tetrahedral_quality(path, grid, logger=print):
    from src.meshing.quality import _element_geometry, get_tets

    points = np.asarray(grid.points)
    tets = get_tets(grid)
    _, sj, _ = _element_geometry(points, tets)
    p = points[tets]
    view = grid.copy()
    view.cell_data["scaled_jacobian"] = sj
    cut = view.slice(normal="z")

    def plot(plotter):
        plotter.add_mesh(cut, scalars="scaled_jacobian", cmap="coolwarm")

    def fallback(output):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        centers = p.mean(axis=1)
        axes[0].scatter(centers[:, 0], centers[:, 1], c=sj, s=3, cmap="coolwarm")
        axes[0].set_title("Tet centres, colored by SJ")
        axes[1].hist(sj, bins=40)
        axes[1].set_title("Scaled Jacobian")
        _save_figure(output, fig)

    rendered = _render_3d(path, "Tetrahedral quality cut plane", plot, fallback, logger)
    if rendered:
        try:
            image = plt.imread(path)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(image)
            axes[0].axis("off")
            axes[1].hist(sj, bins=40)
            axes[1].set_title("Scaled Jacobian")
            rendered = _save_figure(path, fig)
        except Exception as exc:
            _log(logger, f"Quality histogram composition failed: {type(exc).__name__}: {exc}")
    return rendered


def render_gnn_flags(path, grid, probabilities, threshold, logger=print):
    flags = np.asarray(probabilities) > float(threshold)
    view = grid.copy()
    view.cell_data["gnn_flag"] = flags.astype(np.int8)
    cut = view.slice(normal="z")

    def plot(plotter):
        plotter.add_mesh(cut, scalars="gnn_flag", cmap="coolwarm", clim=(0, 1))

    def fallback(output):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        centers = np.asarray(grid.points)[
            np.asarray(grid.cells_dict[pv.CellType.TETRA])
        ].mean(axis=1)
        axes[0].scatter(centers[:, 0], centers[:, 1], c=flags, s=3, cmap="coolwarm")
        axes[0].set_title("Flagged tet centres")
        axes[1].hist(probabilities, bins=40)
        axes[1].axvline(threshold, color="red")
        axes[1].set_title(f"Probability; flagged={int(flags.sum())}")
        _save_figure(output, fig)

    rendered = _render_3d(path, "GNN correction flags", plot, fallback, logger)
    if rendered:
        try:
            image = plt.imread(path)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(image)
            axes[0].axis("off")
            axes[1].hist(probabilities, bins=40)
            axes[1].axvline(threshold, color="red")
            axes[1].set_title(f"Probability; flagged={int(flags.sum())}")
            rendered = _save_figure(path, fig)
        except Exception as exc:
            _log(logger, f"Probability histogram composition failed: {type(exc).__name__}: {exc}")
    return rendered


def render_correction(path, before, after, stats, logger=print):
    def plot(plotter):
        plotter.add_mesh(_pv_surface(before), color="tomato", opacity=0.4)
        plotter.add_mesh(_pv_surface(after), color="seagreen", opacity=0.8)

    rendered = _render_3d(
        path,
        f"Correction: moved={stats.get('moved_points', 0)}, "
        f"modified elements={stats.get('modified_elements', 0)}",
        plot,
        lambda output: _fallback_surface(output, after, "Corrected mesh", logger),
        logger,
    )
    if not rendered:
        return False
    try:
        from src.meshing.quality import _element_geometry, get_tets

        before_points = np.asarray(before.points)
        after_points = np.asarray(after.points)
        before_sj = _element_geometry(before_points, get_tets(before))[1]
        after_sj = _element_geometry(after_points, get_tets(after))[1]
        image = plt.imread(path)
        fig = plt.figure(figsize=(15, 5))
        axes = [
            fig.add_subplot(131),
            fig.add_subplot(132),
            fig.add_subplot(133, projection="3d"),
        ]
        axes[0].imshow(image)
        axes[0].axis("off")
        axes[1].hist(before_sj, bins=40, alpha=0.5, label="before")
        axes[1].hist(after_sj, bins=40, alpha=0.5, label="after")
        axes[1].legend()
        axes[1].set_title("Scaled Jacobian")
        if before_points.shape == after_points.shape:
            moved = np.any(before_points != after_points, axis=1)
            if np.any(moved):
                axes[2].scatter(*after_points[moved].T, s=5, c="red")
        axes[2].set_title("Moved vertices")
        return _save_figure(path, fig)
    except Exception as exc:
        _log(logger, f"Correction histogram composition failed: {type(exc).__name__}: {exc}")
        return rendered


def render_geometry(path, grid, reference, logger=print):
    points, _ = _surface_data(grid)
    reference_points, _ = _surface_data(reference)
    distances = cKDTree(reference_points).query(points)[0]
    surface = grid.extract_surface().triangulate()
    surface.point_data["reference_distance"] = distances

    return _render_3d(
        path, "Corrected surface distance to reference",
        lambda plotter: plotter.add_mesh(
            surface, scalars="reference_distance", cmap="viridis",
        ),
        lambda output: _fallback_surface(
            output, surface, "Distance to reference", logger,
        ),
        logger,
    )


def render_fe(path, grid, solution, residuals, fe_result, logger=print):
    if solution is None:
        _log(logger, "FE preview skipped: solver produced no solution field.")
        return False
    view = grid.copy()
    view.point_data["solution"] = np.asarray(solution)
    cut = view.slice(normal="z")

    def plot(plotter):
        plotter.add_mesh(cut, scalars="solution", cmap="viridis")

    def fallback(output):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        points = np.asarray(grid.points)
        axes[0].scatter(points[:, 0], points[:, 1], c=solution, s=4, cmap="viridis")
        axes[0].set_title("FE solution")
        axes[1].semilogy(np.maximum(residuals, 1e-16))
        axes[1].set_title("CG residual")
        _save_figure(output, fig)

    title = (
        f"FE solution; assembly={fe_result.get('assembly_success')}, "
        f"converged={fe_result.get('solver_converged')}"
    )
    rendered = _render_3d(path, title, plot, fallback, logger)
    if rendered and residuals:
        try:
            image = plt.imread(path)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(image)
            axes[0].axis("off")
            axes[1].semilogy(np.maximum(residuals, 1e-16))
            axes[1].set_title("CG residual")
            rendered = _save_figure(path, fig)
        except Exception as exc:
            _log(logger, f"Residual-curve composition failed: {type(exc).__name__}: {exc}")
    return rendered
