"""Entry point. Runs steps 1 to 4 on one mesh and the library metrics.

Usage:
  python -m scripts.validate_steps --mesh data/meshes/0000_clean.vtu \
      --metrics data/processed/classical_metrics.csv
"""
import argparse
import sys

import pyvista as pv

from src.validation import mesh_checks, fe_checks, correction_checks, library_checks
from src.validation.report import Report


def main():
    from src.utils.run_logging import start_run_log
    start_run_log("validate_steps")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--out", default="validation/steps_1_4.json")
    args = ap.parse_args()

    report = Report()
    grid = pv.read(args.mesh)

    mesh_checks.run(grid, report, step=1)
    fe_checks.run(grid, report, step=2, log=print)
    correction_checks.run(report, step=3)
    if args.metrics:
        library_checks.run(args.metrics, report, step=4)

    for r in report.results:
        print(f"[{r.status.upper():4}] step {r.step} {r.name}: "
              f"measured={r.measured} threshold={r.threshold} {r.detail}")
    print("summary:", report.summary())

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    report.write(args.out)
    sys.exit(report.exit_code())


if __name__ == "__main__":
    main()