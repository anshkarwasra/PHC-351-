"""Pipeline stage 4: fill a closed surface with tetrahedra using TetGen's defaults."""
import warnings

import numpy as np
import pyvista as pv
import tetgen
import trimesh


def tetrahedralise(mesh: trimesh.Trimesh, maxvolume=None, mindihedral=None,
                  minratio=None, nobisect=False, opt_iterations=None,
                  opt_max_flip_level=None, steinerleft=None) -> pv.UnstructuredGrid:
    if not mesh.is_watertight:
        raise ValueError("TetGen needs a watertight surface")
    faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel()
    surface = pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64), faces)
    tet = tetgen.TetGen(surface)
    options = {}
    if maxvolume is not None:
        if maxvolume <= 0:
            raise ValueError("maxvolume must be positive")
        options["maxvolume"] = float(maxvolume)
    if mindihedral is not None:
        if not 0 <= mindihedral <= 40:
            raise ValueError("mindihedral must be between 0 and 40 degrees")
        options["mindihedral"] = float(mindihedral)
    if minratio is not None:
        if minratio < 1.0:
            raise ValueError("minratio must be >= 1.0")
        if minratio < 1.2:
            warnings.warn("minratio below 1.2 is aggressive and may produce poor quality output", stacklevel=2)
        options["minratio"] = float(minratio)
    if nobisect:
        options["nobisect"] = True
    if opt_iterations is not None:
        options["optiterations"] = int(opt_iterations)
    if opt_max_flip_level is not None:
        options["optmaxfliplevel"] = int(opt_max_flip_level)
    if steinerleft is not None:
        options["steinerleft"] = int(steinerleft)
    tet.tetrahedralize(**options)
    return tet.grid