"""Attempt loop, logging, manifest, and splits for synthetic library generation."""
import argparse
import csv
import datetime
import json
import os
import shutil
import subprocess
import time

import numpy as np

from src.synthetic.phantom_library import PhantomLibrary, load_config
from src.training.dataset import make_splits
from src.utils.progress import progress


def get_git_version():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def main():
    from src.utils.run_logging import start_run_log
    start_run_log("generate_library", extra={"config": "configs/synthetic.yaml"})
    parser = argparse.ArgumentParser(description="Generate synthetic phantom library")
    parser.add_argument("--config", default="configs/synthetic.yaml", help="Path to config file")
    parser.add_argument("--n", type=int, default=None, help="Target instance count")
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate the library from scratch, removing only its generated subfolders",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_target = args.n if args.n is not None else cfg.get("n_target", 100)
    base_seed = cfg.get("base_seed", 42)
    max_attempts = cfg.get("max_attempts", 150)
    out_dir = cfg.get("out_dir", "data/synthetic")

    if args.force:
        for sub in ("specs", "reference", "density", "degraded", "logs", ".tmp"):
            target = os.path.join(out_dir, sub)
            if os.path.isdir(target):
                shutil.rmtree(target)
        for filename in ("manifest.csv", "splits.json", "library_meta.json"):
            target = os.path.join(out_dir, filename)
            if os.path.isfile(target):
                os.remove(target)

    # Clean up stale .tmp directory if present
    tmp_root = os.path.join(out_dir, ".tmp")
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root, ignore_errors=True)

    lib = PhantomLibrary(config_or_path=cfg, out_dir=out_dir)

    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    attempts_log_path = os.path.join(log_dir, "attempts.jsonl")

    manifest_path = os.path.join(out_dir, "manifest.csv")
    accepted_rows = []
    accepted_ids = []
    if os.path.exists(manifest_path):
        rows_by_id = {}
        with open(manifest_path, "r", newline="") as manifest_fh:
            for row in csv.DictReader(manifest_fh):
                rows_by_id.setdefault(row["id"], []).append(row)

        expected_tiers = set(lib.tiers)
        for tag, rows in rows_by_id.items():
            if {row["tier"] for row in rows} != expected_tiers:
                continue
            required_paths = (
                "volume_path",
                "reference_path",
                "density_path",
                "spec_path",
            )
            if all(os.path.isfile(row[path]) for row in rows for path in required_paths):
                accepted_ids.append(tag)
                accepted_rows.extend(rows)

    k = 0
    n_accepted = len(accepted_ids)
    n_rejected = 0
    n_errors = 0
    attempt_offset = 0
    logged_attempts = 0
    next_seed = base_seed
    if os.path.exists(attempts_log_path):
        with open(attempts_log_path, "r") as log_fh:
            for line in log_fh:
                if not line.strip():
                    continue
                previous_attempt = json.loads(line)
                logged_attempts += 1
                attempt_offset = max(attempt_offset, previous_attempt["attempt"] + 1)
                next_seed = max(next_seed, previous_attempt["seed"] + 1)

    bar = progress(total=max_attempts, desc="library generation", unit="attempt")
    with open(attempts_log_path, "a") as log_fh:
        while n_accepted < n_target and k < max_attempts:
            t0 = time.perf_counter()
            attempt_seed = next_seed + k
            ss = np.random.SeedSequence(attempt_seed)
            geom_ss, noise_ss = ss.spawn(2)
            geom_rng = np.random.default_rng(geom_ss)
            noise_rng = np.random.default_rng(noise_ss)

            params = lib.sample_params(geom_rng)
            dens = lib.density(params)
            ok, reason, metrics = lib.check_geometry(dens)

            status = "rejected"
            if not ok:
                n_rejected += 1
            else:
                tag = f"{n_accepted:04d}"
                try:
                    rows = lib.write_instance_atomic(tag, attempt_seed, params, dens, noise_rng)
                    accepted_rows.extend(rows)
                    accepted_ids.append(tag)
                    status = "accepted"
                    n_accepted += 1
                except Exception as exc:
                    status = "error"
                    reason = f"{type(exc).__name__}: {exc}"
                    n_errors += 1

            dur = time.perf_counter() - t0
            log_entry = {
                "attempt": attempt_offset + k,
                "seed": attempt_seed,
                "status": status,
                "reason": reason,
                "metrics": metrics,
                "seconds": round(dur, 4),
            }
            log_fh.write(json.dumps(log_entry) + "\n")
            log_fh.flush()
            k += 1
            bar.update(1)
            bar.set_postfix(accepted=n_accepted, rejected=n_rejected, errors=n_errors)
    bar.close()

    shortfall = max(0, n_target - n_accepted)

    # Atomic write of manifest.csv
    manifest_tmp = os.path.join(out_dir, "manifest.csv.tmp")
    manifest_final = manifest_path
    if accepted_rows:
        with open(manifest_tmp, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(accepted_rows[0].keys()))
            writer.writeheader()
            writer.writerows(accepted_rows)
        if os.path.exists(manifest_final):
            os.remove(manifest_final)
        os.rename(manifest_tmp, manifest_final)

    # Atomic write of splits.json
    splits_final = os.path.join(out_dir, "splits.json")
    splits_tmp = os.path.join(out_dir, "splits.json.tmp")
    make_splits(accepted_ids, splits_tmp, seed=base_seed, split_fractions=cfg.get("splits", None))
    if os.path.exists(splits_final):
        os.remove(splits_final)
    os.rename(splits_tmp, splits_final)

    # Atomic write of library_meta.json
    meta = {
        "config": cfg,
        "base_seed": base_seed,
        "n_accepted": n_accepted,
        "n_attempts": logged_attempts + k,
        "attempts_this_run": k,
        "shortfall": shortfall,
        "code_version": get_git_version(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    meta_final = os.path.join(out_dir, "library_meta.json")
    meta_tmp = os.path.join(out_dir, "library_meta.json.tmp")
    with open(meta_tmp, "w") as fh:
        json.dump(meta, fh, indent=2)
    if os.path.exists(meta_final):
        os.remove(meta_final)
    os.rename(meta_tmp, meta_final)

    print(
        f"Summary: accepted={n_accepted}, rejected={n_rejected}, "
        f"errors={n_errors}, attempts={k}"
    )

    if shortfall > 0:
        print(
            f"Shortfall: target {n_target} not reached (accepted {n_accepted}) "
            f"within {max_attempts} attempts."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
