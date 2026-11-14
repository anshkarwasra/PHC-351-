"""Compatibility entry point for the consolidated Project 75 final study."""

METHODS = (
    "classical", "global_taubin", "global_taubin_retet",
    "blind_global_smart", "rule_targeted", "gnn", "retet_global_quality",
)


def main():
    from scripts.run_final import main as run_final
    run_final()


if __name__ == "__main__":
    main()
