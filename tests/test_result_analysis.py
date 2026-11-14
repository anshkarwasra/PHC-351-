import pandas as pd

from scripts.analyze_results import paired_deltas, summarize_deltas


def test_paired_statistics_include_directional_change_counts():
    rows = []
    for mesh_id, classical, targeted, gnn in (
        ("a", 20.0, 18.0, 16.0),
        ("b", 15.0, 15.0, 16.0),
        ("c", 10.0, 12.0, 11.0),
    ):
        for method, poor in (
            ("classical", classical), ("rule_targeted", targeted), ("gnn", gnn),
        ):
            rows.append({
                "id": mesh_id, "tier": "clean", "method": method,
                "status": "ok", "post_poor_pct": poor,
            })
    deltas = paired_deltas(pd.DataFrame(rows))
    stats = summarize_deltas(deltas)
    gnn_vs_rule = stats[
        (stats["metric"] == "poor_pct") & (stats["baseline"] == "rule_targeted")
    ].iloc[0]

    assert gnn_vs_rule["n"] == 3
    assert gnn_vs_rule["n_improved"] == 2
    assert gnn_vs_rule["n_unchanged"] == 0
    assert gnn_vs_rule["n_worse"] == 1
