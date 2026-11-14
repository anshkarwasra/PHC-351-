"""Pipeline stages 1 to 3: threshold segmentation, marching cubes, surface repair."""
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure


def segment(volume, threshold):
    """Threshold the intensity volume, then keep one solid body."""
    mask = volume > threshold
    mask = ndimage.binary_fill_holes(mask)        # noise-induced cavities become solid
    labels, n = ndimage.label(mask)
    if n > 1:                                      # keep the largest connected component
        sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
        mask = labels == (np.argmax(sizes) + 1)
    return mask


def extract_surface(mask, voxel_size, smooth_sigma=0.0):
    """Marching cubes on the binary mask. Zero padding guarantees a closed surface."""
    padded = np.pad(mask.astype(np.float32), 1)
    if smooth_sigma < 0:
        raise ValueError("smooth_sigma must be non-negative")
    if smooth_sigma > 0:
        padded = ndimage.gaussian_filter(padded, sigma=smooth_sigma)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5,
                                                spacing=(voxel_size,) * 3)
    # undo the one-voxel pad and move to cell-centre coordinates in [-1, 1]
    verts = verts + (-1.0 - 0.5 * voxel_size)
    return trimesh.Trimesh(verts, faces, process=True)


def condition_surface(mesh, n_taubin=10, target_faces=None):
    """Smooth and optionally decimate a surface, then restore and verify topology."""
    if n_taubin < 0:
        raise ValueError("n_taubin must be non-negative")
    if target_faces is not None and target_faces < 4:
        raise ValueError("target_faces must be at least 4")

    mesh = mesh.copy()
    faces_before = len(mesh.faces)
    volume_before = abs(float(mesh.volume))
    if n_taubin:
        trimesh.smoothing.filter_taubin(mesh, iterations=n_taubin)
    if target_faces is not None and target_faces < len(mesh.faces):
        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)

    mesh, repair = repair_surface(mesh)
    if not mesh.is_watertight:
        raise ValueError("conditioned surface is not watertight")
    volume_after = abs(float(mesh.volume))
    report = {
        **repair,
        "faces_before": faces_before,
        "faces_after": len(mesh.faces),
        "volume_change_pct": 100.0 * abs(volume_after - volume_before) /
        max(volume_before, 1e-12),
    }
    return mesh, report


def _fill_boundary_loops(mesh):
    """Fan-fill closed boundary loops larger than trimesh's triangle/quad repair."""
    edges = mesh.edges_sorted
    unique_edges, inverse, counts = np.unique(
        edges, axis=0, return_inverse=True, return_counts=True,
    )
    boundary_edges = unique_edges[counts == 1]
    if not len(boundary_edges):
        return mesh, 0

    adjacency = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        return mesh, 0

    loops = []
    remaining = set(adjacency)
    while remaining:
        start = next(iter(remaining))
        loop = [start]
        previous = None
        current = start
        while True:
            choices = [vertex for vertex in adjacency[current] if vertex != previous]
            following = choices[0]
            if following == start:
                break
            if following in loop:
                return mesh, 0
            loop.append(following)
            previous, current = current, following
        remaining.difference_update(loop)
        if len(loop) >= 5:
            loops.append(loop)

    if not loops:
        return mesh, 0

    vertices = np.asarray(mesh.vertices)
    faces = [np.asarray(mesh.faces)]
    new_vertices = []
    new_faces = []
    for loop in loops:
        centre_id = len(vertices) + len(new_vertices)
        new_vertices.append(vertices[loop].mean(axis=0))
        new_faces.extend(
            (centre_id, loop[index], loop[(index + 1) % len(loop)])
            for index in range(len(loop))
        )
    mesh.vertices = np.vstack((vertices, np.asarray(new_vertices)))
    mesh.faces = np.vstack((*faces, np.asarray(new_faces, dtype=np.int64)))
    return mesh, len(new_faces)


def repair_surface(mesh):
    """Give TetGen a closed, manifold, consistently oriented surface."""
    n_in = len(mesh.faces)
    volume_before = abs(float(mesh.volume))
    mesh.merge_vertices()
    before_degenerate_removal = len(mesh.faces)
    mesh.update_faces(mesh.nondegenerate_faces())
    n_degenerate_removed = before_degenerate_removal - len(mesh.faces)
    before_duplicate_removal = len(mesh.faces)
    mesh.update_faces(mesh.unique_faces())
    n_duplicate_removed = before_duplicate_removal - len(mesh.faces)
    mesh.remove_unreferenced_vertices()

    broken = trimesh.repair.broken_faces(mesh)        # edges shared by more than two faces
    if len(broken):
        keep = np.ones(len(mesh.faces), dtype=bool)
        keep[broken] = False
        mesh.update_faces(keep)

    faces_before_fill = len(mesh.faces)
    trimesh.repair.fill_holes(mesh)
    mesh, n_loop_faces = _fill_boundary_loops(mesh)
    trimesh.repair.fix_normals(mesh)                  # consistent winding, outward normals

    report = dict(
        faces_in=n_in,
        faces_out=len(mesh.faces),
        n_broken=int(len(broken)),
        n_degenerate_removed=int(n_degenerate_removed),
        n_duplicate_removed=int(n_duplicate_removed),
        n_loop_faces=int(n_loop_faces),
        n_faces_filled=int(len(mesh.faces) - faces_before_fill),
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        volume_positive=bool(mesh.volume > 0),
        volume_change_pct=100.0 * abs(abs(float(mesh.volume)) - volume_before) /
        max(volume_before, 1e-12),
    )
    return mesh, report