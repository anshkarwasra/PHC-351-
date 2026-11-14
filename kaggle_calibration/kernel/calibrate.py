"""Headless Kaggle entry point for validation-only checkpoint calibration."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def _extract_dataset_archives():
    input_root = Path("/kaggle/input")
    unpacked = Path("/kaggle/working/project75_calibration_input")
    if unpacked.exists():
        shutil.rmtree(unpacked)
    unpacked.mkdir(parents=True)

    for archive in input_root.rglob("*.zip"):
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                destination = (unpacked / entry.filename).resolve()
                if not destination.is_relative_to(unpacked.resolve()):
                    raise ValueError(f"unsafe path in Kaggle dataset archive: {entry.filename}")
            source.extractall(unpacked)
    return input_root, unpacked


def _find_project_bundle():
    input_root, unpacked = _extract_dataset_archives()
    matches = sorted({
        path.parent.parent
        for root in (input_root, unpacked)
        for path in root.rglob("gnn_guidance_model.pt")
        if (path.parent.parent / "src" / "training" / "train_gnn.py").is_file()
    })
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one attached Project 75 calibration dataset containing "
            "the checkpoint and source code after inspecting files and ZIP archives; "
            f"found {len(matches)} bundle(s). Verify that the kernel metadata attaches "
            "the calibration dataset and that its uploaded contents include data and src."
        )
    return matches[0]


def _install_missing_dependencies():
    packages = {
        "torch_geometric": "torch-geometric",
        "pyvista": "pyvista",
        "tetgen": "tetgen",
        "trimesh": "trimesh",
        "scipy": "scipy",
        "yaml": "PyYAML",
        "tqdm": "tqdm",
    }
    missing = [
        package for module, package in packages.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        print("Installing missing calibration dependencies:", ", ".join(missing), flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *missing],
            check=True,
        )


def main():
    source = _find_project_bundle()
    work = Path("/kaggle/working/project75_calibration_work")
    output = Path("/kaggle/working/calibration_output")
    if work.exists():
        shutil.rmtree(work)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    _install_missing_dependencies()
    shutil.copytree(source, work)
    result = subprocess.run(
        [sys.executable, "-m", "src.training.train_gnn", "--calibrate-only"],
        cwd=work,
        check=False,
    )

    threshold_report = (
        work / "data" / "evaluation_reports" / "threshold_selection_report.json"
    )
    if result.returncode != 0:
        if threshold_report.is_file():
            report = json.loads(threshold_report.read_text(encoding="utf-8"))
            if report.get("selection_status") == "no_eligible_threshold":
                shutil.copy2(threshold_report, output / threshold_report.name)
                (output / "calibration_status.json").write_text(
                    json.dumps({
                        "status": "no_eligible_threshold",
                        "message": (
                            "Strict validation gates failed. No threshold or updated "
                            "checkpoint was selected; inspect threshold_selection_report.json."
                        ),
                    }, indent=2),
                    encoding="utf-8",
                )
                print(
                    "Calibration stopped because no candidate passed the strict "
                    f"validation gates. Diagnostic report exported to {output}.",
                    flush=True,
                )
                shutil.rmtree(work)
                return
        raise subprocess.CalledProcessError(result.returncode, result.args)

    report_path = work / "data" / "evaluation_reports" / "selected_config.json"
    checkpoint_path = work / "data" / "gnn_guidance_model.pt"
    if not report_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Calibration did not produce the selected config and checkpoint.")
    shutil.copy2(report_path, output / "selected_config.json")
    shutil.copy2(checkpoint_path, output / "gnn_guidance_model.pt")
    if threshold_report.is_file():
        shutil.copy2(threshold_report, output / threshold_report.name)
    print(f"Calibration outputs are in {output}", flush=True)
    shutil.rmtree(work)


if __name__ == "__main__":
    os.chdir("/kaggle/working")
    main()
