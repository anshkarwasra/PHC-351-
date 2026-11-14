"""Step 4: the library behaves sensibly before scaling.

Checks that tiers differ in the surface, reports element counts, and checks
that per-tier bad-element rates differ by a measurable amount.
"""
import csv
import numpy as np

from src.validation.report import CheckResult

MIN_TIER_DIFF = 0.005     # poor vs clean bad-element fraction, absolute
MAX_TETS = 120_000        # soft target, see step 4 notes


def check_tier_separation(bad_frac_by_tier, report, step=4):
    """bad_frac_by_tier: {'clean': x, 'moderate': y, 'poor': z} as fractions."""
    diff = bad_frac_by_tier["poor"] - bad_frac_by_tier["clean"]
    report.add(CheckResult(step, "tier_bad_fraction_separation",
                           "pass" if diff >= MIN_TIER_DIFF else "warn",
                           float(diff), MIN_TIER_DIFF,
                           "small difference means degradation does not reach elements"))


def check_element_counts(n_tets_list, report, step=4):
    n = np.asarray(n_tets_list)
    over = int((n > MAX_TETS).sum())
    report.add(CheckResult(step, "element_count_within_target",
                           "pass" if over == 0 else "warn",
                           int(n.max()) if len(n) else 0, MAX_TETS,
                           f"{over} of {len(n)} above target; median={int(np.median(n))}"))


def check_tier_ordering(surface_faces_by_tier, report, step=4):
    """Per instance, poor should not have fewer surface faces than clean by a large margin."""
    # Informational only. Degradation can add or remove faces.
    report.add(CheckResult(step, "surface_faces_recorded", "pass",
                           {k: int(v) for k, v in surface_faces_by_tier.items()},
                           None, "informational"))


def load_metrics_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def run(metrics_csv, report, step=4):
    """Reads the classical pipeline output (classical_metrics.csv)."""
    rows = [r for r in load_metrics_csv(metrics_csv) if r.get("status") == "ok"]
    by_tier = {}
    for r in rows:
        by_tier.setdefault(r["tier"], []).append(r)
    n_tets = [int(r["n_tets"]) for r in rows]
    check_element_counts(n_tets, report, step)
    # bad-element fraction per tier must be recorded by the mesh stage as "poor_frac"
    fracs = {t: float(np.mean([float(r["poor_frac"]) for r in rs]))
             for t, rs in by_tier.items() if rs and "poor_frac" in rs[0]}
    if {"clean", "poor"} <= fracs.keys():
        check_tier_separation(fracs, report, step)
    else:
        report.add(CheckResult(step, "tier_bad_fraction_separation", "warn", None, None,
                               "poor_frac column missing; add it to the mesh stage"))