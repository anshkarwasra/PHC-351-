"""Trains MeshQualityGNN on classical tet meshes, using the library's fixed splits.

Assumes src/meshing/quality.py provides:
  compute_tetrahedral_metrics(grid) -> dict with keys
      'features' (N, F), 'tets' (N, 4), 'needs_correction' (N,), 'smoothing_target' (N,)
  build_element_dual_graph(tets) -> (2, E) int array, both directions included
"""
import json
import hashlib
import argparse
import os
import random
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.models.gnn_model import MeshQualityGNN
from src.meshing.quality import (
    LABEL_VERSION, compute_tetrahedral_metrics, build_element_dual_graph,
    get_correction_config,
)
from src.utils.progress import progress

LIB_DIR = "data/synthetic"
MESH_DIR = "data/meshes"
CACHE_DIR = "data/graph_cache"
CKPT_PATH = "data/gnn_guidance_model.pt"

CFG = dict(
    hidden_dim=64, n_blocks=3, dropout=0.1,
    epochs=150, lr=1e-3, weight_decay=1e-4, grad_clip=1.0,
    batch_size=2, focal_alpha=0.75, focal_gamma=2.0, lambda_reg=2.0,
    patience=20, seed=0, tiers=("clean", "moderate", "poor"),
    rotation_augmentation=True, rescale_augmentation=True,
)


def _project_correction_config():
    config_path = Path(__file__).resolve().parents[2] / "configs" / "synthetic.yaml"
    with config_path.open("r", encoding="utf-8") as source:
        project_config = yaml.safe_load(source) or {}
    return get_correction_config(project_config.get("correction", {}))


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_raw_graph(mesh_path, correction_config=None):
    """Build and cache one mesh's graph. Raw features, normalisation happens later."""
    stem = os.path.basename(mesh_path)[:-4]
    correction = get_correction_config(correction_config)
    config_hash = hashlib.sha1(
        json.dumps(correction, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    cache = os.path.join(CACHE_DIR, f"{stem}.{LABEL_VERSION}.{config_hash}.pt")
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    m = compute_tetrahedral_metrics(
        pv.read(mesh_path), correction_config=correction,
    )
    data = Data(
        x=torch.as_tensor(m["features"], dtype=torch.float32),
        edge_index=torch.as_tensor(build_element_dual_graph(m["tets"]), dtype=torch.long),
        y_cls=torch.as_tensor(m["needs_correction"], dtype=torch.float32),
        y_reg=torch.as_tensor(m["smoothing_target"], dtype=torch.float32),
    )
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(data, cache)
    return data


def _rescale_geometric_features(features, scale):
    """Exact feature transform equivalent to a uniform point-cloud rescaling."""
    features = features.clone()
    features[:, 2] += 3.0 * np.log10(scale)
    features[:, 3:6] += np.log10(scale)
    return features


def _augmented_graph(mesh_path, base_graph, cfg):
    """Augment invariant features without repeatedly loading large point clouds.

    All geometric features are rotation invariant. Uniform scaling only changes
    log-volume and log-edge-length columns, so this is exactly the feature
    result of transforming points before computing those features.
    """
    del mesh_path
    scale = np.random.uniform(0.9, 1.1) if cfg["rescale_augmentation"] else 1.0
    return Data(
        x=_rescale_geometric_features(base_graph.x, scale),
        edge_index=base_graph.edge_index,
        y_cls=base_graph.y_cls,
        y_reg=base_graph.y_reg,
    )


def mesh_paths_for(split, tiers):
    """Paths for one split, from splits.json. Never glob the mesh folder."""
    with open(os.path.join(LIB_DIR, "splits.json")) as fh:
        ids = sorted(json.load(fh)[split])
    paths = [os.path.join(MESH_DIR, f"{i}_{t}.vtu") for i in ids for t in tiers]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} meshes missing for split '{split}', e.g. {missing[0]}")
    return paths


def feature_stats(paths, correction_config=None):
    """Mean, std and positive rate from the training split only. Streams one graph at a time."""
    n = 0
    s = s2 = None
    pos = 0.0
    for p in paths:
        d = load_raw_graph(p, correction_config)
        x = d.x.double()
        s = x.sum(0) if s is None else s + x.sum(0)
        s2 = (x * x).sum(0) if s2 is None else s2 + (x * x).sum(0)
        n += x.shape[0]
        pos += d.y_cls.double().sum().item()
    mean = s / n
    std = (s2 / n - mean * mean).clamp_min(1e-12).sqrt()
    return mean.float(), std.float(), pos / n


class GraphDataset(Dataset):
    def __init__(self, paths, mean, std, augment=False, cfg=None, correction_config=None):
        self.paths, self.mean, self.std, self.augment = paths, mean, std, augment
        self.cfg = cfg or CFG
        self.correction_config = correction_config

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        d = load_raw_graph(self.paths[i], self.correction_config)
        if self.augment:
            d = _augmented_graph(self.paths[i], d, self.cfg)
        d.x = (d.x - self.mean) / self.std
        return d


def focal_bce(logits, targets, alpha, gamma):
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * ce).mean()


def reg_loss(reg, y_reg, y_cls):
    """Smoothing weight is only supervised on flagged elements."""
    pos = y_cls > 0.5
    if not pos.any():
        return reg.sum() * 0.0
    return F.mse_loss(reg[pos], y_reg[pos])


def run_epoch(model, loader, device, cfg, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, n_graphs = 0.0, 0
    tp = fp = fn = 0.0
    abs_err = []
    all_probabilities = []
    all_targets = []
    for batch in loader:
        batch = batch.to(device)
        with torch.set_grad_enabled(train):
            logit, reg = model(batch.x, batch.edge_index)
            l_cls = focal_bce(logit, batch.y_cls, cfg["focal_alpha"], cfg["focal_gamma"])
            l_reg = reg_loss(reg, batch.y_reg, batch.y_cls)
            loss = l_cls + cfg["lambda_reg"] * l_reg
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

        total += loss.item() * batch.num_graphs
        n_graphs += batch.num_graphs
        probabilities = torch.sigmoid(logit)
        pred = (probabilities > float(cfg.get("threshold", 0.5))).float()
        tp += (pred * batch.y_cls).sum().item()
        fp += (pred * (1 - batch.y_cls)).sum().item()
        fn += ((1 - pred) * batch.y_cls).sum().item()
        all_probabilities.append(probabilities.detach().cpu().numpy())
        all_targets.append(batch.y_cls.detach().cpu().numpy())
        pos = batch.y_cls > 0.5
        if pos.any():
            abs_err.append((reg[pos] - batch.y_reg[pos]).abs().cpu())

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    mae = torch.cat(abs_err).mean().item() if abs_err else float("nan")
    probabilities = np.concatenate(all_probabilities)
    targets = np.concatenate(all_targets) > 0.5
    order = np.argsort(-probabilities, kind="stable")
    sorted_targets = targets[order]
    positives = int(sorted_targets.sum())
    if positives:
        precision_at_rank = np.cumsum(sorted_targets) / np.arange(1, len(sorted_targets) + 1)
        average_precision = float(np.sum(precision_at_rank * sorted_targets) / positives)
    else:
        average_precision = 0.0
    return dict(loss=total / max(n_graphs, 1), precision=precision,
                recall=recall, f1=f1, average_precision=average_precision,
                reg_mae_pos=mae)


def train(cfg=CFG, ckpt_path=CKPT_PATH):
    from src.utils.run_logging import start_run_log
    cfg = {**CFG, **cfg}
    cfg["correction"] = get_correction_config(
        cfg.get("correction", _project_correction_config())
    )
    start_run_log("train_gnn", extra={"checkpoint": ckpt_path})
    print("training MeshQualityGNN with config:", cfg)
    set_seeds(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_paths = mesh_paths_for("train", cfg["tiers"])
    val_paths = mesh_paths_for("val", cfg["tiers"])
    mean, std, pos_rate = feature_stats(train_paths, cfg["correction"])
    print(f"train graphs={len(train_paths)} val graphs={len(val_paths)} "
          f"positive rate={pos_rate:.4f} device={device}")

    g = torch.Generator().manual_seed(cfg["seed"])
    loader_tr = DataLoader(GraphDataset(
        train_paths, mean, std, augment=True, cfg=cfg,
        correction_config=cfg["correction"],
    ),
                           batch_size=cfg["batch_size"], shuffle=True, generator=g)
    loader_va = DataLoader(GraphDataset(
        val_paths, mean, std, cfg=cfg, correction_config=cfg["correction"],
    ),
                           batch_size=cfg["batch_size"], shuffle=False)

    in_channels = load_raw_graph(train_paths[0], cfg["correction"]).x.shape[1]
    model = MeshQualityGNN(in_channels, cfg["hidden_dim"], cfg["n_blocks"], cfg["dropout"]).to(device)
    model.set_cls_prior(pos_rate)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    best, best_epoch, bad = float("-inf"), 0, 0
    best_val_f1 = float("-inf")
    with progress(
        total=cfg["epochs"], desc="training", unit="epoch",
    ) as bar:
        for epoch in range(1, cfg["epochs"] + 1):
            tr = run_epoch(model, loader_tr, device, cfg, optimizer=opt)
            va = run_epoch(model, loader_va, device, cfg, optimizer=None)
            sched.step()
            best_val_f1 = max(best_val_f1, va["f1"])
            bar.set_postfix(
                train_loss=f"{tr['loss']:.4f}",
                val_loss=f"{va['loss']:.4f}",
                best_val_f1=f"{best_val_f1:.3f}",
            )
            bar.update(1)
            bar.write(f"epoch {epoch:03d} | train loss {tr['loss']:.4f} | "
                      f"val AP {va['average_precision']:.4f} f1 {va['f1']:.3f} "
                      f"P {va['precision']:.3f} R {va['recall']:.3f} reg_mae {va['reg_mae_pos']:.3f}")

            if va["average_precision"] > best:
                best, best_epoch, bad = va["average_precision"], epoch, 0
                os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": cfg,
                    "in_channels": in_channels,
                    "feature_mean": mean,
                    "feature_std": std,
                    "label_version": LABEL_VERSION,
                    "best_epoch": best_epoch,
                    "val_metrics": va,
                }, ckpt_path)
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    bar.write(f"early stop at epoch {epoch}; best epoch {best_epoch}")
                    break

    print(f"saved best checkpoint (epoch {best_epoch}) to {ckpt_path}")
    calibration = calibrate_checkpoint(ckpt_path, val_paths)
    print("validation calibration:", calibration)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    report = {
        "label_version": LABEL_VERSION,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "augmentation": {
            "rotation": bool(cfg["rotation_augmentation"]),
            "rescale": bool(cfg["rescale_augmentation"]),
            "rotation_note": "Geometric input features are rotation invariant.",
        },
        "best_epoch": best_epoch,
        "validation": {
            **checkpoint["val_metrics"],
            "precision": calibration["validation_precision"],
            "recall": calibration["validation_recall"],
            "f1": calibration["validation_f1"],
            "pr_auc": calibration["validation_pr_auc"],
            "regression_mae": calibration["validation_reg_mae"],
        },
        "calibration": calibration,
    }
    report_path = os.path.join(os.path.dirname(ckpt_path), "training_report.json")
    with open(report_path, "w") as report_fh:
        json.dump(report, report_fh, indent=2)
    print(f"saved training report to {report_path}")
    return ckpt_path


def calibrate_checkpoint(
    ckpt_path, val_paths=None, max_correction_evaluations=100,
    allow_ineligible=False, reuse_threshold_report=False,
):
    """Select an operational threshold and regression scale using validation data only."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ck.get("label_version") != LABEL_VERSION:
        raise ValueError("cannot calibrate a checkpoint with mismatched training labels")
    cfg = ck["config"]
    if val_paths is None:
        val_paths = mesh_paths_for("val", cfg["tiers"])
    correction_config = get_correction_config(cfg.get("correction"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeshQualityGNN(
        ck["in_channels"], cfg["hidden_dim"], cfg["n_blocks"], cfg["dropout"],
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    dataset = GraphDataset(
        val_paths, ck["feature_mean"], ck["feature_std"],
        cfg=cfg, correction_config=correction_config,
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False)
    probabilities, targets, weights, regression_targets = [], [], [], []
    with progress(
        total=len(loader), desc="calibration validation inference", unit="batch",
    ) as bar:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits, reg = model(batch.x, batch.edge_index)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
                targets.append(batch.y_cls.cpu().numpy())
                weights.append(reg.cpu().numpy())
                regression_targets.append(batch.y_reg.cpu().numpy())
                bar.update(1)

    probs = np.concatenate(probabilities)
    y_cls = np.concatenate(targets) > 0.5
    predicted_weights = np.concatenate(weights)
    target_weights = np.concatenate(regression_targets)

    positive = y_cls
    if np.any(positive):
        prediction = predicted_weights[positive]
        target = target_weights[positive]
        scale = float(np.dot(prediction, target) / max(np.dot(prediction, prediction), 1e-12))
        scale = float(np.clip(scale, 0.0, 2.0))
        reg_mae = float(np.mean(np.abs(scale * prediction - target)))
    else:
        scale, reg_mae = 1.0, float("nan")
    selected_threshold, threshold_report = _select_validation_threshold(
        ckpt_path, val_paths, correction_config, scale,
        max_correction_evaluations=max_correction_evaluations,
        allow_ineligible=allow_ineligible,
        reuse_report=reuse_threshold_report,
    )
    calibrated_prediction = probs > selected_threshold
    calibrated_tp = int(np.sum(calibrated_prediction & y_cls))
    calibrated_fp = int(np.sum(calibrated_prediction & ~y_cls))
    calibrated_fn = int(np.sum(~calibrated_prediction & y_cls))
    calibrated_precision = calibrated_tp / max(calibrated_tp + calibrated_fp, 1)
    calibrated_recall = calibrated_tp / max(calibrated_tp + calibrated_fn, 1)
    calibrated_f1 = (
        2 * calibrated_precision * calibrated_recall /
        max(calibrated_precision + calibrated_recall, 1e-12)
    )
    order = np.argsort(-probs, kind="stable")
    sorted_targets = y_cls[order]
    ap = (
        float(np.sum((np.cumsum(sorted_targets) / np.arange(1, len(sorted_targets) + 1))
                     * sorted_targets) / max(int(y_cls.sum()), 1))
        if np.any(y_cls) else 0.0
    )

    calibration = {
        "prob_threshold": selected_threshold,
        "smoothing_scale": scale,
        "validation_f1": calibrated_f1,
        "validation_precision": calibrated_precision,
        "validation_recall": calibrated_recall,
        "validation_pr_auc": ap,
        "validation_reg_mae": reg_mae,
        "validation_positive_rate": float(np.mean(y_cls)),
        "threshold_selection": threshold_report,
    }
    ck["calibration"] = calibration
    torch.save(ck, ckpt_path)
    selected_config_path = (
        Path(__file__).resolve().parents[2]
        / "data" / "evaluation_reports" / "selected_config.json"
    )
    selected_config_path.parent.mkdir(parents=True, exist_ok=True)
    selected_config_path.write_text(
        json.dumps({
            "method": "gnn",
            "gnn_prob_threshold": selected_threshold,
            "correction": correction_config,
            "label_version": LABEL_VERSION,
            "selection_status": threshold_report["selection_status"],
            "validation_selection": threshold_report,
        }, indent=2),
        encoding="utf-8",
    )
    return calibration


def _stratified_calibration_paths(val_paths, max_correction_evaluations):
    """Choose a deterministic, tier-balanced subset for the expensive correction search."""
    operations_per_mesh = 5
    if max_correction_evaluations < operations_per_mesh * 3:
        raise ValueError(
            "max_correction_evaluations must allow at least one mesh per quality tier "
            f"({operations_per_mesh * 3} evaluations)"
        )
    if max_correction_evaluations % operations_per_mesh:
        raise ValueError(
            f"max_correction_evaluations must be a multiple of {operations_per_mesh}"
        )

    groups = {}
    for path in val_paths:
        tier = Path(path).stem.rsplit("_", 1)[-1]
        groups.setdefault(tier, []).append(path)
    tier_order = [tier for tier in ("clean", "moderate", "poor") if tier in groups]
    tier_order.extend(sorted(set(groups) - set(tier_order)))
    if len(tier_order) < 3:
        raise ValueError("calibration validation paths must cover all three quality tiers")

    max_meshes = max_correction_evaluations // operations_per_mesh
    selected_count = min(max_meshes, len(val_paths))
    if selected_count < len(tier_order):
        raise ValueError("calibration subset is too small to represent every quality tier")

    quotas = {tier: selected_count // len(tier_order) for tier in tier_order}
    for tier in tier_order[:selected_count % len(tier_order)]:
        quotas[tier] += 1
    for tier in tier_order:
        quotas[tier] = min(quotas[tier], len(groups[tier]))
    while sum(quotas.values()) < selected_count:
        available = [
            tier for tier in tier_order if quotas[tier] < len(groups[tier])
        ]
        if not available:
            break
        tier = min(available, key=lambda item: (quotas[item], tier_order.index(item)))
        quotas[tier] += 1

    selected = []
    for tier in tier_order:
        paths = groups[tier]
        count = quotas[tier]
        indices = np.linspace(0, len(paths) - 1, count).round().astype(int)
        selected.extend(paths[index] for index in indices)
    return selected


def _select_validation_threshold(
    ckpt_path, val_paths, correction_config, smoothing_scale,
    max_correction_evaluations=100, allow_ineligible=False, reuse_report=False,
):
    """Choose the lowest allowed threshold satisfying the validation constraints."""
    import pyvista as pv

    from src.meshing.correction import correct_mesh
    from src.meshing.quality import _element_geometry, get_tets

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MeshQualityGNN(
        checkpoint["in_channels"], model_config["hidden_dim"],
        model_config["n_blocks"], model_config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.feature_mean = checkpoint["feature_mean"].to(device)
    model.feature_std = checkpoint["feature_std"].to(device)
    model.smoothing_scale = smoothing_scale

    candidate_thresholds = (0.2, 0.3, 0.4, 0.5)
    report_path = (
        Path(__file__).resolve().parents[2]
        / "data" / "evaluation_reports" / "threshold_selection_report.json"
    )
    if reuse_report:
        if not report_path.is_file():
            raise FileNotFoundError(f"existing calibration report not found: {report_path}")
        threshold_report = json.loads(report_path.read_text(encoding="utf-8"))
        if threshold_report.get("candidate_thresholds") != list(candidate_thresholds):
            raise ValueError("existing calibration report has incompatible thresholds")
        sampled_paths = threshold_report.get("sampled_validation_meshes", [])
        validation_names = {Path(path).name for path in val_paths}
        if (
            threshold_report.get("validation_meshes_available") != len(val_paths)
            or not sampled_paths
            or not set(sampled_paths).issubset(validation_names)
            or threshold_report.get("correction_evaluations", 0) > max_correction_evaluations
        ):
            raise ValueError("existing calibration report does not match this validation split/run limit")
        if threshold_report.get("selection_status") not in (
            "no_eligible_threshold", "exploratory_ineligible",
        ):
            raise ValueError("existing calibration report has an unsupported selection status")
        rule_median = float(threshold_report["rule_targeted_median_modified_fraction"])
        classical_median_min_sj = float(threshold_report["classical_median_min_sj"])
        candidate_results = threshold_report.get("candidates", {})
        for threshold in candidate_thresholds:
            result = candidate_results.get(str(threshold), {})
            if not np.isfinite(result.get("median_modified_fraction", np.nan)):
                raise ValueError(f"existing report lacks metrics for threshold {threshold}")
            if not np.isfinite(result.get("median_min_sj", np.nan)):
                raise ValueError(f"existing report lacks metrics for threshold {threshold}")
    else:
        correction_paths = _stratified_calibration_paths(
            val_paths, max_correction_evaluations,
        )
        actual_evaluations = len(correction_paths) * (len(candidate_thresholds) + 1)
        outcomes = {threshold: {"modified_fraction": [], "min_sj": []}
                    for threshold in candidate_thresholds}
        rule_modified_fraction = []
        classical_min_sj = []

        with progress(
            total=actual_evaluations,
            desc="calibration correction search", unit="mesh-op",
        ) as bar:
            for path in correction_paths:
                grid = pv.read(path)
                tets = get_tets(grid)
                _, pre_sj, _ = _element_geometry(np.asarray(grid.points), tets)
                classical_min_sj.append(float(pre_sj.min()))
                _, rule_stats = correct_mesh(
                    grid, method="rule_targeted", correction_config=correction_config,
                )
                rule_fraction = float(rule_stats["modified_elements"]) / max(len(tets), 1)
                rule_modified_fraction.append(rule_fraction)
                bar.set_postfix(mesh=Path(path).stem, arm="rule", modified=f"{rule_fraction:.3f}")
                bar.update(1)
                for threshold in candidate_thresholds:
                    corrected, stats = correct_mesh(
                        grid, model, method="gnn", prob_threshold=threshold,
                        correction_config=correction_config,
                    )
                    post_tets = get_tets(corrected)
                    _, post_sj, _ = _element_geometry(np.asarray(corrected.points), post_tets)
                    modified_fraction = float(stats["modified_elements"]) / max(len(tets), 1)
                    min_sj = float(post_sj.min())
                    outcomes[threshold]["modified_fraction"].append(modified_fraction)
                    outcomes[threshold]["min_sj"].append(min_sj)
                    bar.set_postfix(
                        mesh=Path(path).stem, arm=f"gnn@{threshold:.1f}",
                        modified=f"{modified_fraction:.3f}", min_sj=f"{min_sj:.3f}",
                    )
                    bar.update(1)

        rule_median = float(np.median(rule_modified_fraction))
        classical_median_min_sj = float(np.median(classical_min_sj))
        threshold_report = {
            "candidate_thresholds": list(candidate_thresholds),
            "rule_targeted_median_modified_fraction": rule_median,
            "classical_median_min_sj": classical_median_min_sj,
            "minimum_allowed_median_min_sj": 0.9 * classical_median_min_sj,
            "validation_meshes_available": len(val_paths),
            "correction_search_meshes": len(correction_paths),
            "correction_evaluations": actual_evaluations,
            "sampled_validation_meshes": [Path(path).name for path in correction_paths],
            "candidates": {},
        }
        for threshold in candidate_thresholds:
            threshold_report["candidates"][str(threshold)] = {
                "median_modified_fraction": float(
                    np.median(outcomes[threshold]["modified_fraction"])
                ),
                "median_min_sj": float(np.median(outcomes[threshold]["min_sj"])),
            }

    eligible = []
    for threshold in candidate_thresholds:
        candidate = threshold_report["candidates"][str(threshold)]
        median_modified = float(candidate["median_modified_fraction"])
        median_min_sj = float(candidate["median_min_sj"])
        valid = (
            median_modified < rule_median
            and median_min_sj >= 0.9 * classical_median_min_sj
        )
        candidate["eligible"] = valid
        if valid:
            eligible.append(threshold)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not eligible:
        threshold_report["selection_status"] = (
            "exploratory_ineligible" if allow_ineligible else "no_eligible_threshold"
        )
        threshold_report["selection_policy"] = (
            "best_observed_median_min_sj_then_lowest_modified_fraction"
            if allow_ineligible else "strict_validation_gates"
        )
        report_path.write_text(json.dumps(threshold_report, indent=2), encoding="utf-8")
        if allow_ineligible:
            selected = max(
                candidate_thresholds,
                key=lambda threshold: (
                    threshold_report["candidates"][str(threshold)]["median_min_sj"],
                    -threshold_report["candidates"][str(threshold)]["median_modified_fraction"],
                    -threshold,
                ),
            )
            threshold_report["selected_threshold"] = selected
            threshold_report["selected_candidate_eligible"] = False
            threshold_report["selection_status"] = "exploratory_ineligible"
            report_path.write_text(json.dumps(threshold_report, indent=2), encoding="utf-8")
            return selected, threshold_report
        raise RuntimeError(
            "No GNN probability threshold satisfies the validation modification "
            f"and minimum-SJ constraints. Candidate metrics saved to {report_path}:\n"
            f"{json.dumps(threshold_report, indent=2)}"
        )
    selected = min(eligible)
    threshold_report["selected_threshold"] = selected
    threshold_report["selected_candidate_eligible"] = True
    threshold_report["selection_status"] = "eligible"
    report_path.write_text(json.dumps(threshold_report, indent=2), encoding="utf-8")
    return selected, threshold_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or calibrate the Project 75 GNN.")
    parser.add_argument(
        "--calibrate-only", action="store_true",
        help="calibrate the existing checkpoint without retraining",
    )
    parser.add_argument(
        "--calibration-evaluations", type=int, default=100,
        help=(
            "maximum validation correction operations; must be a multiple of 5 "
            "and at least 15 (default: 100)"
        ),
    )
    parser.add_argument(
        "--allow-ineligible", action="store_true",
        help=(
            "explicitly select the least-bad candidate if strict validation gates fail; "
            "marks output exploratory and not validation-qualified"
        ),
    )
    parser.add_argument(
        "--reuse-threshold-report", action="store_true",
        help="reuse a completed threshold search report instead of rerunning corrections",
    )
    args = parser.parse_args()
    if args.calibrate_only:
        from src.utils.run_logging import start_run_log
        start_run_log(
            "train_gnn_calibration",
            extra={
                "checkpoint": CKPT_PATH,
                "max_correction_evaluations": args.calibration_evaluations,
                "allow_ineligible": args.allow_ineligible,
                "reuse_threshold_report": args.reuse_threshold_report,
            },
        )
        calibration = calibrate_checkpoint(
            CKPT_PATH,
            max_correction_evaluations=args.calibration_evaluations,
            allow_ineligible=args.allow_ineligible,
            reuse_threshold_report=args.reuse_threshold_report,
        )
        print("validation calibration:", json.dumps(calibration, indent=2))
    else:
        train()