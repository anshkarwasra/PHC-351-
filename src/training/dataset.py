"""Loader for the synthetic library. Plain Python for now. Wrap it in torch.utils.data.Dataset for the GNN."""
import csv
import json
import os

import numpy as np


def make_splits(ids, path, seed=0, split_fractions=None):
    """Deterministic split by instance, so all tiers of one phantom stay in the same split."""
    ids = sorted(set(ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    if split_fractions is None:
        split_fractions = {"train": 0.70, "val": 0.15, "test": 0.15}
    f_tr = split_fractions.get("train", 0.70)
    f_va = split_fractions.get("val", 0.15)
    n_tr = round(f_tr * len(ids))
    n_va = round(f_va * len(ids))
    splits = {"train": ids[:n_tr], "val": ids[n_tr:n_tr + n_va], "test": ids[n_tr + n_va:]}
    with open(path, "w") as fh:
        json.dump(splits, fh, indent=1)
    return splits


class PhantomDataset:
    """One item is one (instance, tier) pair."""

    def __init__(self, library_dir="data/synthetic", split=None,
                 tiers=("clean", "moderate", "poor")):
        self.library_dir = library_dir
        with open(os.path.join(library_dir, "manifest.csv")) as fh:
            rows = [r for r in csv.DictReader(fh) if r["tier"] in tiers]
        if split is not None:
            with open(os.path.join(library_dir, "splits.json")) as fh:
                keep = set(json.load(fh)[split])
            rows = [r for r in rows if r["id"] in keep]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "id": r["id"],
            "tier": r["tier"],
            "volume": np.load(r["volume_path"]).astype(np.float32),
            "seg_threshold": float(r["seg_threshold"]),
            "reference_path": r["reference_path"],
        }

    def reference_density(self, id_):
        path = os.path.join(self.library_dir, "density", f"{id_}.npz")
        return np.load(path)["density"].astype(np.float32)
