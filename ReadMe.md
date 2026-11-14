# Project 75: AI-Based CT-to-Mesh Optimization for Simulation

This repository implements a synthetic CT-to-mesh workflow with one learned
component: a GraphSAGE model that flags poor tetrahedra and predicts a local
smoothing weight. The held-out test split is reserved for one final run.

## Pipeline

```text
Synthetic CSG volume -> threshold segmentation -> marching cubes
-> trimesh surface repair -> default TetGen
-> classical / global baseline / rule-targeted / GNN correction
-> geometric and FE sanity checks
```

The study method set is frozen:

1. `classical`: threshold, marching cubes, `trimesh` repair, default TetGen.
2. `global_taubin`: uniform interior Taubin smoothing.
3. `global_taubin_retet`: uniform Taubin smoothing and finer global-size TetGen.
4. `blind_global_smart`: the shared correction operator applied everywhere.
5. `rule_targeted`: the shared operator on vertices incident to elements with SJ < 0.25.
6. `gnn`: the shared operator on GNN-flagged elements, scaled by the predicted weight.
7. `retet_global_quality`: optional global TetGen quality reference.

Targeted arms use the common `correction` block in
[`configs/synthetic.yaml`](./configs/synthetic.yaml). Legacy method variants
and surface conditioning are not part of the study.

## Run order

Run commands from the project root. Dependencies are pinned in
[`requirements.txt`](./requirements.txt).

```bash
python -m scripts.generate_library --config configs/synthetic.yaml
python -m scripts.run_pipeline --split train_val
python -m scripts.check_project_data
python -m scripts.check_gnn_spec
python -m src.training.train_gnn
python -m scripts.preview_steps --split val --n 3
# Review data/previews/preview_report.md and validation configuration:
python -m scripts.run_final --split val
# Inspect data/evaluation_reports/final_val.csv and selected_config.json.
# After fixing the method/configuration, run the held-out test exactly once:
python -m scripts.run_final --split test --confirm-test
python -m scripts.analyze_results
```

Preview output is written to `data/previews/`, including cached intermediate
meshes and the visual/numeric report. Final split results are written to
`data/evaluation_reports/final_<split>.csv`; the test analysis reads only
`final_test.csv`. Logs are written to `data/logs/`.

The project reports test outcomes as observed, including null or negative
results. Poor-element percentages are presented with absolute element counts
and element growth because re-tetrahedralisation can change the denominator.
FE success, convergence iterations, geometry accuracy, and stage runtimes are
reported as downstream evaluation measures, not as evidence of clinical
validation.

## Headless validation calibration on Kaggle

If local calibration is too slow, stage only the trained checkpoint, the 66
validation meshes, and their graph caches:

```powershell
python -m scripts.prepare_kaggle_calibration
```

The dataset metadata is prefilled for the Kaggle account used in this setup.
Create the dataset privately (private is the default) and preserve its nested
folders with `--dir-mode zip`:

```powershell
kaggle datasets create -p kaggle_calibration\dataset --dir-mode zip
```

Then push the kernel:

```powershell
kaggle kernels push -p kaggle_calibration\kernel
```

The kernel calibrates the existing checkpoint only; it does not retrain or
evaluate the held-out test split. Download its `calibration_output` files with
`kaggle kernels output`, then place the returned checkpoint at
`data\gnn_guidance_model.pt` and `selected_config.json` at
`data\evaluation_reports\selected_config.json` before resuming local validation
and later pipeline stages. If no threshold passes the validation gates, the
kernel exports `threshold_selection_report.json` and a status file instead of
selecting a fallback or exporting a calibrated checkpoint. Kaggle CLI
credentials and internet access are required for package installation and
kernel execution.

For local calibration, the correction search defaults to 100 operations
(20 deterministically sampled validation meshes across the three tiers, with
five correction runs per mesh). Validation model inference still uses the full
validation split. This smaller correction sample is faster but less
representative than searching all 66 validation meshes; the strict threshold
eligibility gates remain enabled:

```powershell
python -m src.training.train_gnn --calibrate-only --calibration-evaluations 100
```

If that correction search completed but found no eligible candidate, an
explicitly exploratory threshold can be selected from its saved report without
repeating the expensive correction operations:

```powershell
python -m src.training.train_gnn --calibrate-only --calibration-evaluations 100 --allow-ineligible --reuse-threshold-report
python -m scripts.run_final --split val --allow-exploratory --max-phantoms 3
```

The three-phantom example evaluates all three tiers and all methods for the
deterministically selected IDs. It writes `final_val_exploratory_3phantoms.csv`
and a matching sample JSON, not `final_val.csv`. Omit `--max-phantoms` to
evaluate the full validation split. The configuration is marked
`exploratory_ineligible`, has failed the validation gates, and is blocked from
test-split evaluation. Do not describe the results as validation-qualified or
as evidence that the method passed its gates.
