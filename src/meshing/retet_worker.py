"""Isolated TetGen worker used to enforce a hard timeout on Windows."""
import json
import sys
from pathlib import Path

import pyvista as pv
import trimesh

from src.meshing.tetgen_wrapper import tetrahedralise


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        raise SystemExit("usage: retet_worker INPUT.stl OUTPUT.vtu PARAMS.json")
    input_path, output_path, params_path = map(Path, argv)
    surface = trimesh.load_mesh(str(input_path), process=True)
    with params_path.open(encoding="utf-8") as param_file:
        params = json.load(param_file)
    grid = tetrahedralise(surface, **params)
    grid.save(str(output_path))


if __name__ == "__main__":
    main()
