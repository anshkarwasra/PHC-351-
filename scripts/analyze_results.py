"""Analyze only the final held-out test results using the Project 75 experiments."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "data" / "evaluation_reports"
TEST_CSV = REPORT_DIR / "final_test.csv"
REPORT = REPORT_DIR / "approach_and_evaluation_article.md"
BOOTSTRAP_SAMPLES = 5000
SEED = 75

METHODS = (
    "classical", "global_taubin", "global_taubin_retet",
    "blind_global_smart", "rule_targeted", "gnn", "retet_global_quality",
)
METRICS = {
    "poor_count": ("poor_count", False),
    "poor_pct": ("poor_pct", False),
    "inverted_pct": ("inverted_pct", False),
    "min_sj": ("min_sj", True),
    "mean_sj": ("mean_sj", True),
    "p1_sj": ("p1_sj", True),
    "hausdorff_mean": ("hausdorff_mean", False),
    "hausdorff_p95": ("hausdorff_p95", False),
    "hausdorff_max": ("hausdorff_max", False),
    "volume_error_pct": ("abs_vol_err_pct", False),
    "watertight": ("watertight", True),
    "fe_success": ("fe_success", True),
    "fe_iters": ("fe_iters", False),
    "runtime_sec": ("t_total_sec", False),
}


def _read_test_results():
    if not TEST_CSV.is_file():
        raise FileNotFoundError(f"final test results not found: {TEST_CSV}")
    frame = pd.read_csv(TEST_CSV, dtype={"id": str})
    required = {
        "id", "tier", "method", "status",
        *[field for field, _ in METRICS.values() if field != "abs_vol_err_pct"],
        "vol_err_pct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"final test CSV is missing columns: {', '.join(missing)}")
    frame = frame[frame["status"] == "ok"].copy()
    if "vol_err_pct" in frame:
        frame["abs_vol_err_pct"] = frame["vol_err_pct"].abs()
    for column in ("watertight", "fe_success"):
        if column in frame:
            frame[column] = frame[column].map(
                lambda value: str(value).strip().lower() in {"true", "1"}
                if not pd.isna(value) else np.nan
            )
            frame[column] = frame[column].astype("Int64")
    return frame


def paired_deltas(results):
    rows = []
    metrics = {name: fields for name, fields in METRICS.items()
               if fields[0] in results.columns}
    for metric, (field, higher_is_better) in metrics.items():
        values = results.pivot(index=["id", "tier"], columns="method", values=field)
        if "classical" not in values:
            continue
        comparisons = ["gnn"] if "gnn" in values else []
        if metric in {
            "poor_count", "poor_pct", "inverted_pct", "min_sj", "mean_sj",
            "p1_sj", "runtime_sec",
        }:
            comparisons = [method for method in METHODS if method in values and method != "classical"]
        for comparison in comparisons:
            for (mesh_id, tier), pair in values[["classical", comparison]].dropna().iterrows():
                baseline_value = pair["classical"]
                comparison_value = pair[comparison]
                rows.append({
                    "id": mesh_id,
                    "tier": tier,
                    "metric": metric,
                    "baseline": "classical",
                    "comparison": comparison,
                    "baseline_value": baseline_value,
                    "comparison_value": comparison_value,
                    "delta": comparison_value - baseline_value,
                    "higher_is_better": higher_is_better,
                })
    return pd.DataFrame(rows)


def summarize_deltas(deltas, stratify_tier=True):
    if deltas.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(SEED)
    group_fields = ["metric", "baseline", "comparison"]
    if stratify_tier:
        group_fields.insert(0, "tier")
    rows = []
    for key, group in deltas.groupby(group_fields, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        fields = dict(zip(group_fields, key))
        values = group["delta"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        p_value = (
            1.0 if np.allclose(values, 0)
            else float(wilcoxon(values, zero_method="wilcox").pvalue)
        )
        samples = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True)
        ci_low, ci_high = np.percentile(np.median(samples, axis=1), [2.5, 97.5])
        higher_is_better = bool(group["higher_is_better"].iloc[0])
        improved = values > 0 if higher_is_better else values < 0
        unchanged = np.isclose(values, 0)
        rows.append({
            **fields,
            "n": len(values),
            "median_delta": float(np.median(values)),
            "median_ci95_low": float(ci_low),
            "median_ci95_high": float(ci_high),
            "wilcoxon_p": p_value,
            "wins": int(np.count_nonzero(improved)),
            "ties": int(np.count_nonzero(unchanged)),
            "losses": int(np.count_nonzero(~improved & ~unchanged)),
        })
    return pd.DataFrame(rows)


def _e3_comparisons(results):
    reference_methods = (
        "global_taubin", "global_taubin_retet", "blind_global_smart",
        "rule_targeted",
    )
    rows = []
    for method in reference_methods:
        if method not in set(results["method"]) or "gnn" not in set(results["method"]):
            continue
        pair = results[results["method"].isin(("gnn", method))].pivot(
            index=["id", "tier"], columns="method",
            values=["poor_count", "poor_pct", "min_sj", "modified_elements", "growth", "t_total_sec"],
        )
        for index, values in pair.dropna().iterrows():
            mesh_id, tier = index
            row = {"id": mesh_id, "tier": tier, "comparison": method}
            for metric in ("poor_count", "poor_pct", "min_sj"):
                row[f"gnn_{metric}"] = values[(metric, "gnn")]
                row[f"{method}_{metric}"] = values[(metric, method)]
                row[f"delta_{metric}"] = values[(metric, "gnn")] - values[(metric, method)]
            for metric in ("modified_elements", "growth", "t_total_sec"):
                row[f"gnn_{metric}"] = values[(metric, "gnn")]
                row[f"{method}_{metric}"] = values[(metric, method)]
            rows.append(row)
    return pd.DataFrame(rows)


def _e4_fe(results):
    fe_summary = results.groupby(["method", "tier"], as_index=False).agg(
        n=("id", "count"),
        fe_success_rate=("fe_success", "mean"),
        median_fe_iters=("fe_iters", "median"),
        median_poor_count=("poor_count", "median"),
        median_min_sj=("min_sj", "median"),
    )
    paired = results.pivot(
        index=["id", "tier"], columns="method",
        values=["fe_success", "fe_iters", "poor_count"],
    )
    rows = []
    if (
        ("fe_iters", "classical") in paired.columns
        and ("fe_iters", "gnn") in paired.columns
    ):
        for (mesh_id, tier), values in paired.dropna(
            subset=[("fe_iters", "classical"), ("fe_iters", "gnn"),
                    ("poor_count", "classical"), ("poor_count", "gnn")],
        ).iterrows():
            rows.append({
                "id": mesh_id,
                "tier": tier,
                "poor_count_change": values[("poor_count", "gnn")] - values[("poor_count", "classical")],
                "fe_iters_change": values[("fe_iters", "gnn")] - values[("fe_iters", "classical")],
                "fe_success_change": int(values[("fe_success", "gnn")]) - int(values[("fe_success", "classical")]),
            })
    relation = pd.DataFrame(rows)
    correlations = []
    if not relation.empty:
        for tier, group in relation.groupby("tier"):
            if len(group) < 2:
                correlation, p_value = float("nan"), float("nan")
            else:
                result = spearmanr(group["poor_count_change"], group["fe_iters_change"])
                correlation, p_value = float(result.statistic), float(result.pvalue)
            correlations.append({
                "tier": tier, "n": len(group),
                "spearman_poor_count_vs_fe_iters": correlation,
                "p_value": p_value,
            })
    return fe_summary, relation, pd.DataFrame(correlations)


def _e5_runtime(results):
    frame = results.groupby("method", as_index=False).agg(
        meshes=("id", "count"),
        classical_sec=("t_classical_sec", "median"),
        features_sec=("t_features_sec", "median"),
        gnn_inference_sec=("t_gnn_sec", "median"),
        smoothing_sec=("t_smoothing_sec", "median"),
        reverification_sec=("t_reverification_sec", "median"),
        total_sec=("t_total_sec", "median"),
    )
    classical_total = float(
        frame.loc[frame["method"] == "classical", "total_sec"].median()
    )
    frame["overhead_pct_vs_classical"] = (
        100.0 * (frame["total_sec"] - classical_total) / max(classical_total, 1e-12)
    )
    return frame


def _expected_vs_observed(results):
    rows = []
    for tier in ("clean", "moderate", "poor"):
        pair = results[results["tier"] == tier].pivot(
            index="id", columns="method",
            values=["pre_poor_count", "poor_count", "pre_poor_pct", "poor_pct"],
        )
        if ("poor_count", "gnn") not in pair.columns:
            continue
        valid = pair.dropna(subset=[
            ("pre_poor_count", "gnn"), ("poor_count", "gnn"),
            ("pre_poor_pct", "gnn"), ("poor_pct", "gnn"),
        ])
        if valid.empty:
            continue
        pre_counts = valid[("pre_poor_count", "gnn")]
        post_counts = valid[("poor_count", "gnn")]
        count_changes = post_counts - pre_counts
        pre_pct = valid[("pre_poor_pct", "gnn")]
        post_pct = valid[("poor_pct", "gnn")]
        rows.append({
            "tier": tier,
            "n": len(valid),
            "median_poor_count_before": float(pre_counts.median()),
            "median_poor_count_after": float(post_counts.median()),
            "median_absolute_count_change": float(count_changes.median()),
            "median_relative_poor_count_reduction_pct": float(
                np.median(100.0 * (pre_counts - post_counts) / np.maximum(pre_counts, 1))
            ),
            "median_poor_pct_before": float(pre_pct.median()),
            "median_poor_pct_after": float(post_pct.median()),
            "median_poor_pct_reduction_pp": float((pre_pct - post_pct).median()),
        })
    return pd.DataFrame(rows)


def _markdown_table(frame):
    if frame.empty:
        return "_No paired rows available._"
    values = frame.fillna("").astype(object)
    columns = list(values.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in values.itertuples(index=False, name=None):
        items = [
            f"{value:.4g}" if isinstance(value, (float, np.floating))
            else str(value).replace("|", "\\|")
            for value in row
        ]
        lines.append("| " + " | ".join(items) + " |")
    return "\n".join(lines)


def _write_report(results, primary, tiered, e3, fe_summary, fe_correlation, runtime, expected):
    success = results.groupby("method")["fe_success"].mean()
    ceiling = [method for method, rate in success.items() if rate >= 1.0]
    ceiling_text = (
        f"FE success is at ceiling for: {', '.join(ceiling)}."
        if ceiling else "FE success is not at ceiling for any method."
    )
    text = f"""# Project 75: Final Test Analysis

## Methods

The frozen method set is:

1. `classical`: threshold segmentation, marching-cubes surface extraction, `trimesh` repair, and default TetGen tetrahedralisation.
2. `global_taubin`: uniform interior Taubin smoothing.
3. `global_taubin_retet`: uniform Taubin smoothing followed by uniform finer-global-size TetGen re-tetrahedralisation.
4. `blind_global_smart`: the final `smart_smooth` operator applied globally without targeting.
5. `rule_targeted`: the final operator applied to vertices incident to elements with scaled Jacobian below 0.25.
6. `gnn`: the final operator applied to GNN-flagged elements, with the predicted smoothing weight.
7. `retet_global_quality` (optional reference): global TetGen quality settings `mindihedral=10`, `minratio=1.5`.

All targeted arms use the same configured boundary-tangential `smart_smooth` operator and penalty acceptance. No removed variants are included.

## E1: Classical vs GNN, paired test results

{_markdown_table(primary)}

Every poor-element percentage result is accompanied by absolute poor-element counts in the per-mesh final CSV and the expected-vs-observed table.

## E2: Tier-stratified results

{_markdown_table(tiered)}

## E3: Targeted vs global baselines

`modified_elements`, element growth, and total runtime are reported beside paired quality comparisons.

{_markdown_table(e3)}

## E4: Downstream FE impact

{ceiling_text}

{_markdown_table(fe_summary)}

Geometric quality changes vs FE iteration changes:

{_markdown_table(fe_correlation)}

## E5: Runtime overhead

{_markdown_table(runtime)}

## Expected (spec section 20) vs observed

The project expectation is a 25-45% reduction in poor/inverted-element fraction, concentrated in the poor tier. Absolute element-count changes are shown alongside both relative count reduction and poor-percentage reduction.

{_markdown_table(expected)}

Test-split results are read only from `final_test.csv`; negative, null, and positive findings are reported without manual edits.
"""
    REPORT.write_text(text, encoding="utf-8")


def main():
    from src.utils.run_logging import start_run_log
    start_run_log("analyze_results")
    results = _read_test_results()
    deltas = paired_deltas(results)
    primary = summarize_deltas(
        deltas[deltas["comparison"] == "gnn"], stratify_tier=False,
    )
    tiered = summarize_deltas(
        deltas[deltas["comparison"] == "gnn"], stratify_tier=True,
    )
    e3 = _e3_comparisons(results)
    fe_summary, fe_relation, fe_correlation = _e4_fe(results)
    runtime = _e5_runtime(results)
    expected = _expected_vs_observed(results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    deltas.to_csv(REPORT_DIR / "paired_deltas.csv", index=False)
    primary.to_csv(REPORT_DIR / "e1_paired_statistics.csv", index=False)
    tiered.to_csv(REPORT_DIR / "e2_tier_statistics.csv", index=False)
    e3.to_csv(REPORT_DIR / "e3_targeted_vs_global.csv", index=False)
    fe_summary.to_csv(REPORT_DIR / "e4_fe_summary.csv", index=False)
    fe_relation.to_csv(REPORT_DIR / "e4_fe_mesh_deltas.csv", index=False)
    fe_correlation.to_csv(REPORT_DIR / "e4_fe_correlation.csv", index=False)
    runtime.to_csv(REPORT_DIR / "e5_runtime.csv", index=False)
    expected.to_csv(REPORT_DIR / "expected_vs_observed.csv", index=False)
    _write_report(results, primary, tiered, e3, fe_summary, fe_correlation, runtime, expected)
    print(f"wrote spec-aligned analysis to {REPORT_DIR}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
