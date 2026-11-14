"""Helpers for looking at library instances and pipeline outputs. Call them from a notebook."""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv
import trimesh


def to_pyvista(mesh):
    faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel()
    return pv.PolyData(np.asarray(mesh.vertices), faces)


def plot_instance_slices(library_dir, id_, tiers=("clean", "moderate", "poor")):
    """Middle axial slice: clean reference density next to each degraded tier."""
    dens = np.load(os.path.join(library_dir, "density", f"{id_}.npz"))["density"].astype(np.float32)
    panels = [("reference density", dens)]
    for tier in tiers:
        vol = np.load(os.path.join(library_dir, "degraded", f"{id_}_{tier}.npy")).astype(np.float32)
        panels.append((tier, vol))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (title, vol) in zip(axes, panels):
        ax.imshow(vol[:, :, vol.shape[2] // 2].T, origin="lower", cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_surface(mesh, decimate=0.0, show_edges=False, title=""):
    """Hide edges by default: a surface with ~100k+ triangles is solid black with edges drawn."""
    pvm = to_pyvista(mesh)
    if decimate > 0:
        pvm = pvm.decimate_pro(decimate, preserve_topology=True)
    p = pv.Plotter()
    p.add_mesh(pvm, color="ivory", smooth_shading=True, show_edges=show_edges)
    p.add_text(f"{title} ({pvm.n_cells} triangles)", font_size=10)
    p.show()


def plot_comparison(meshes, decimate=0.0):
    """Side-by-side views in one window. meshes: {name: trimesh.Trimesh}."""
    p = pv.Plotter(shape=(1, len(meshes)))
    for i, (name, mesh) in enumerate(meshes.items()):
        p.subplot(0, i)
        pvm = to_pyvista(mesh)
        if decimate > 0:
            pvm = pvm.decimate_pro(decimate, preserve_topology=True)
        p.add_mesh(pvm, color="ivory", smooth_shading=True)
        p.add_text(name, font_size=10)
    p.link_views()
    p.show()


def plot_tet_quality(grid, threshold=0.2, title=""):
    """Shows the tets below threshold in red, over a faint outer surface."""
    grid = grid.cell_quality(quality_measure="scaled_jacobian")
    q = np.asarray(grid["scaled_jacobian"])
    bad = q < threshold
    print(f"{title} tets={grid.n_cells} min={q.min():.3f} mean={q.mean():.3f} "
          f"below {threshold}: {bad.mean():.2%}")
    p = pv.Plotter()
    p.add_mesh(grid.extract_surface(), color="lightgrey", opacity=0.3)
    if bad.any():
        p.add_mesh(grid.extract_cells(np.flatnonzero(bad)), color="red", show_edges=True)
    p.add_text(title, font_size=10)
    p.show()


def summarise_library(library_dir):
    df = pd.read_csv(os.path.join(library_dir, "manifest.csv"))
    print(df.groupby("tier").agg(rows=("id", "count"),
                                 ref_watertight=("ref_watertight", "mean")))
    return df


def summarise_pipeline(metrics_csv):
    df = pd.read_csv(metrics_csv)
    out = df.groupby("tier").agg(
        runs=("id", "count"),
        ok_rate=("status", lambda s: (s == "ok").mean()),
        watertight_rate=("watertight", "mean"),
        median_faces=("surface_faces", "median"),
        median_tets=("n_tets", "median"),
    )
    print(out)
    failures = df.loc[df["status"] != "ok", "error"].value_counts()
    if len(failures):
        print("\nFailures:")
        print(failures)
    return out