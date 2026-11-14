"""Audit the configured GraphSAGE model and training protocol against Project 75."""
import inspect
import warnings

from src.models.gnn_model import MeshQualityGNN
from src.meshing.quality import FEATURE_NAMES
from src.training import train_gnn


def main():
    cfg = train_gnn.CFG
    model = MeshQualityGNN(
        in_channels=len(FEATURE_NAMES),
        hidden_dim=cfg["hidden_dim"],
        n_blocks=cfg["n_blocks"],
        dropout=cfg["dropout"],
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_source = inspect.getsource(train_gnn.train)
    calibration_source = inspect.getsource(train_gnn.calibrate_checkpoint)
    focal_source = inspect.getsource(train_gnn.focal_bce)
    regression_source = inspect.getsource(train_gnn.reg_loss)
    model_forward = inspect.getsource(MeshQualityGNN.forward)

    facts = {
        "GraphSAGE layers": len(model.blocks),
        "hidden dimension": cfg["hidden_dim"],
        "output heads": sum(hasattr(model, name) for name in ("cls_head", "reg_head")),
        "flag output is logit": "logit = self.cls_head(h).squeeze(-1)" in model_forward,
        "parameter count": parameter_count,
        "losses": (
            "MSE regression + focal classification"
            if "F.mse_loss" in regression_source
            and "binary_cross_entropy_with_logits" in focal_source else "mismatch"
        ),
        "optimizer": (
            f"Adam, lr={cfg['lr']}" if "torch.optim.Adam" in train_source else "mismatch"
        ),
        "epochs": cfg["epochs"],
        "rotation augmentation": cfg["rotation_augmentation"],
        "rescale augmentation": cfg["rescale_augmentation"],
        "absolute-position features": any(
            name.lower() in {"x", "y", "z", "x_position", "y_position", "z_position"}
            for name in FEATURE_NAMES
        ),
        "weight output constrained to [0, 1]": "torch.sigmoid(self.reg_head(h))" in model_forward,
        "training split": '"train"' in train_source,
        "validation early stopping": '"val"' in train_source and "patience" in train_source,
        "validation calibration": '"val"' in calibration_source,
        "test isolated": '"test"' not in train_source and '"test"' not in calibration_source,
    }

    assert facts["GraphSAGE layers"] == 3
    assert facts["hidden dimension"] == 64
    assert facts["output heads"] == 2
    assert facts["flag output is logit"]
    assert facts["losses"] == "MSE regression + focal classification"
    assert facts["optimizer"] == "Adam, lr=0.001"
    assert facts["weight output constrained to [0, 1]"]
    assert cfg["lr"] == 1e-3
    assert 100 <= cfg["epochs"] <= 200
    assert facts["rotation augmentation"] and facts["rescale augmentation"]
    assert not facts["absolute-position features"]
    assert facts["training split"] and facts["validation early stopping"]
    assert facts["validation calibration"] and facts["test isolated"]
    if not 70_000 <= parameter_count <= 90_000:
        warnings.warn(
            f"parameter count {parameter_count:,} is outside the expected 70k-90k range",
            stacklevel=1,
        )

    for name, value in facts.items():
        print(f"{name}: {value}")
    print("Spec audit passed; parameter-count range is advisory.")


if __name__ == "__main__":
    main()
