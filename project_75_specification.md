# Project 75: AI-Based CT-to-Mesh Optimization for Simulation

## 1. Project Title
**AI-Based CT-to-Mesh Optimization for Simulation**

---

## 2. Background and Problem Statement
Preparing a CT-derived mesh for FE/CFD simulation is a multi-stage pipeline — segmentation, surface extraction, tetrahedralisation, and quality repair/smoothing — in which errors compound:
* A noisy segmentation yields a rough surface.
* A rough surface yields poor-quality tetrahedra.
* Poor-quality tetrahedra can crash a solver at assembly or convergence time.

Manually iterating through individual open-source tools (`PyVista`, `trimesh`, `Open3D`, `TetGen`/`Gmsh`) to fix each stage in turn is slow and ad hoc. Furthermore, a student project attempting to invent a wholly new meshing algorithm to fix this is out of scope for a 35-mark module.

This capstone instead builds an integrated, end-to-end pipeline that chains these existing open-source tools together and inserts exactly one learned component — a lightweight mesh-quality-prediction/smoothing-guidance GNN (in the spirit of Project 74). This GNN automatically flags and locally corrects simulation-readiness problems, allowing for a rigorous evaluation of whether that single learned component improves simulation-readiness over the classical pipeline run with no correction.

---

## 3. Main Objective
To implement and evaluate an integrated CT-to-simulation-ready-mesh pipeline — segmentation-quality-aware surface extraction, `TetGen`/`Gmsh` tetrahedralisation, and a single learned mesh-quality-driven local remeshing/smoothing correction stage — that measurably improves simulation-readiness (poor/inverted-element rate, surface accuracy, volume preservation, and downstream FE-assembly success) over the same classical pipeline run without the learned correction stage.

---

## 4. Specific Tasks to Be Completed
1. **Build a Synthetic Library:** Generate CT-like phantom volumes spanning a spread of segmentation-quality conditions (clean, moderately degraded, poorly thresholded) to represent realistic imperfect student-facing CT preprocessing outcomes.
2. **Implement Classical Baseline Pipeline:** Construct the end-to-end baseline:
   $$\text{Threshold Segmentation} \longrightarrow \text{Marching Cubes Surface (\texttt{scikit-image})} \longrightarrow \text{Surface Repair (\texttt{trimesh})} \longrightarrow \text{Tetrahedralisation (\texttt{TetGen}/\texttt{Gmsh})}$$
   *(Surface repair includes filling holes, removing degenerate/non-manifold faces, and fixing normals. Tetrahedralisation uses default/uniform settings).*
3. **Train Learned Guidance Model:** Train a lightweight GNN quality-prediction/smoothing-guidance model (a simplified, single-purpose adaptation of Project 74) that flags per-element "needs local correction" and suggests a smoothing-iteration weight based on local mesh-quality features.
4. **Integrate Correction Stage:** Insert the learned component as a post-tetrahedralisation correction step: identify flagged poor-quality regions, extract local patches (`PyVista`/`trimesh`), and apply targeted local Laplacian/Taubin smoothing or a local `TetGen` re-mesh with tightened sizing restricted to those regions only.
5. **Quantify Simulation-Readiness:** Measure performance before and after correction using metrics such as poor/inverted-element fraction, surface Hausdorff distance to known reference geometry, volume-preservation error, and watertightness.
6. **Downstream Sanity Check:** Assemble a simple FE stiffness/Laplacian system or a basic steady-state heat-conduction solve (via `scikit-fem` or a minimal custom FE assembly script) on the before/after meshes to demonstrate solver assembly success and convergence behavior as a practical proxy (full CFD/FE simulation study is explicitly out of scope).
7. **Runtime Benchmarking:** Benchmark end-to-end pipeline runtime for classical-only vs. classical-plus-learned-correction.
8. **Critical Discussion:** Discuss where the learned correction stage helps most (e.g., thin/sharp-feature regions, poorly segmented inputs) versus where it offers little benefit over a simple non-learned global fix.

---

## 5. Required Dataset/Data
A synthetic library of degraded CT-like phantom volumes (with a known clean reference mesh for each) generated entirely by the student's own procedural phantom pipeline. Every stage of the pipeline is built on open-source tools operating on this synthetic data, with no dependency on a real CT scanner.

---

## 6. Recommended Number of Samples
* **Dataset Size:** 100–150 phantom instances spanning three segmentation-quality tiers (clean / moderate / poor).
* **Data Split:** 70% Train / 15% Validation / 15% Test.
* **Processing:** Each instance is carried through the full pipeline to produce a before/after mesh pair for evaluation.

---

## 7. Data Format Required
* **Input Volumes:** `.raw` / `.npy`
* **Extracted Surfaces:** `.stl`
* **Tetrahedral Meshes:** `.vtu` / `.msh` (before and after correction)
* **GNN Features/Predictions:** `.npz`
* **FE-Assembly Logs:** `.json` / `.csv`

---

## 8. Data Collection and Synthetic Generation
Phantoms are generated with a procedural CSG pipeline (independently instantiated, including thin ribs, fillets, and internal channels to stress simulation-readiness) and voxelised at a fixed resolution (e.g., $128^3$).

The three segmentation-quality tiers are produced by applying controlled degradation — Gaussian volume smoothing plus additive noise followed by re-thresholding at a deliberately imperfect level — to the clean high-resolution mask. This gives a fully controlled, reproducible spread of realistic imperfect-segmentation inputs without needing a real scanner.

---

## 9. Ground-Truth / Label Generation Method
* **Quality Metrics:** All quality metrics used throughout the pipeline (scaled Jacobian, aspect ratio, dihedral angle) are computed by direct deterministic geometric formulas, requiring no manual annotation.
* **Geometric Reference:** The reference geometry against which Hausdorff distance and volume-preservation error are measured is the known clean high-resolution phantom mesh generated before degradation, providing an exact end-to-end accuracy ground truth.

---

## 10. Data Preprocessing
* Resample/normalise volumes to a consistent voxel spacing.
* Apply morphological cleanup (remove small floating connected components) before surface extraction.
* Run `trimesh` mesh-repair routines (`fill_holes`, `remove_degenerate_faces`, `fix_normals`) on extracted surfaces prior to tetrahedralisation to ensure a valid manifold input to `TetGen`/`Gmsh`.

---

## 11. Recommended AI/ML Algorithms
A single lightweight **GraphSAGE-style GNN** providing the pipeline's one learned component: per-element "needs correction" flagging and smoothing-weight suggestion. No other new model architecture is introduced; the project's contribution is pipeline integration and evaluation rather than new model design.

---

## 12. Proposed Model Architecture
* **Type:** 3-layer GraphSAGE operating on the element-adjacency graph.
* **Hidden Dimension:** 64
* **Outputs:**
  1. Per-element binary "flag for local correction" head.
  2. Continuous suggested smoothing-iteration-weight head ($[0, 1]$).
* **Parameters:** $\sim 80\text{k}$ parameters (deliberately kept small since it serves as one component of a larger system).

---

## 13. Complete Project Workflow
```
 1. Generate phantom volumes across 3 quality tiers (retain clean reference meshes)
    │
 2. Threshold-segment degraded volume & extract surface via Marching Cubes
    │
 3. Repair surface mesh with trimesh
    │
 4. Tetrahedralise with TetGen/Gmsh at default settings ──► [Checkpoint: Classical-Only]
    │
 5. Compute per-element quality via PyVista/VTK cell-quality filters
    │
 6. Run trained GNN (flag poor-quality regions & suggest smoothing weights)
    │
 7. Apply targeted local smoothing/re-meshing to flagged regions only
    │
 8. Re-verify mesh quality on corrected mesh ─────────────► [Checkpoint: Corrected]
    │
 9. Compute geometric accuracy (Hausdorff/volume error) against clean reference
    │
10. Run FE/heat-conduction assembly sanity check on both checkpoints
    │
11. Benchmark runtime & summarise results in comparative report
```

---

## 14. Training Procedure
* **Setup:** Combined MSE (quality-score regression) + Focal Loss (poor/inverted classification).
* **Optimizer:** Adam ($\text{Learning Rate} = 1\times 10^{-3}$).
* **Epochs:** $\sim 150$.
* **Data:** Trained on a subset of tetrahedral meshes across the segmentation-quality spread with rotation/rescaling augmentations.
* **Hardware:** Trainable on a single GPU due to the compact model size.

---

## 15. Testing Procedure
* Split 70/15/15 by phantom instance (each split containing examples from all three quality tiers).
* The full pipeline is evaluated end-to-end **only** on the held-out test split, comparing the classical-only pipeline against the classical-plus-learned-correction pipeline on identical input volumes.

---

## 16. Expected Outputs
* Trained GNN weights.
* Before/after mesh pairs for every test volume.
* Quality-improvement summary tables and plots.
* FE-assembly success/failure logs.
* Runtime-benchmark table.
* Full end-to-end pipeline code repository/notebook.

---

## 17. Evaluation Metrics
* **Poor/inverted-element fraction** (before vs. after correction).
* **Scaled Jacobian** (mean and minimum).
* **Surface Hausdorff distance** relative to reference mesh.
* **Volume-preservation error** ($\%$).
* **Watertightness rate**.
* **FE/heat-conduction assembly success rate** and solver-readiness proxy (condition number or convergence-iteration count).
* **End-to-end wall-clock runtime**.

---

## 18. Minimum Experiments Required
1. **Primary Comparison:** Classical-only pipeline vs. classical-plus-learned-correction pipeline across the full test set.
2. **Quality Tier Stratification:** Performance stratified across input tiers (clean/moderate/poor) to identify where correction matters most.
3. **Targeted vs. Global Correction:** Learned targeted local correction vs. simple non-learned baseline correction (uniform global Taubin smoothing + uniform finer-global-size `TetGen` re-tetrahedralisation) to isolate the value of spatial targeting.
4. **Downstream Impact Ablation:** Verify whether correction actually changes FE/heat-conduction assembly outcomes beyond geometric metric values.
5. **Overhead Analysis:** Runtime overhead analysis of the learned correction stage relative to total execution time.

---

## 19. Suggested Comparison / Baseline Methods
* **Primary Baseline:** Full classical pipeline with no learned correction ($\text{Thresholding} \rightarrow \text{Marching Cubes} \rightarrow \text{\texttt{trimesh} repair} \rightarrow \text{\texttt{TetGen}/\texttt{Gmsh} default}$).
* **Secondary Baseline:** "Blind" uniform global smoothing/re-meshing applied everywhere (no targeting) to isolate the benefit of learned, spatially-targeted correction.

---

## 20. Expected Results
* **Element Quality:** $25\text{--}45\%$ reduction in poor/inverted-element fraction after learned targeted correction, concentrated on the "poor segmentation" tier.
* **Assembly Success:** FE/heat-conduction assembly success rate improving from $\sim 70\text{--}85\%$ (classical-only, poor tier) to $\ge 95\%$ (with correction).
* **Efficiency:** Learned targeted correction achieving quality improvement comparable to or better than blind uniform correction while modifying substantially fewer elements at lower runtime overhead.

---

## 21. Possible Challenges
* **Toolchain Integration:** Interfacing multiple third-party meshing libraries (`TetGen` wrapper, `Gmsh`, `trimesh`, `PyVista`) whose Python APIs can be fragile together.
* **Patch Boundaries:** Defining local re-meshing on flagged regions robustly without introducing new non-manifold boundaries at patch edges.
* **Scope Control:** Keeping the FE sanity-check stage minimal to prevent scope creep into a full CFD/FE simulation project.
* **Fair Benchmarking:** Ensuring fair comparison between classical-only vs. corrected pipelines (e.g., matching element-count budgets or runtime constraints where relevant).
* **Reproducibility:** Managing version dependencies across chained open-source meshing tools.

---

## 22. Suggested Python Libraries / Software
* **Deep Learning & Graphs:** `PyTorch`, `PyTorch Geometric`
* **3D Geometry & Meshing:** `PyVista`, `VTK`, `trimesh`, `Open3D`, `scikit-image`, `tetgen` / `pygmsh` / `gmsh`
* **Finite Element Analysis:** `scikit-fem` (or custom minimal FE assembly scripts)
* **Core Utilities:** `NumPy`, `SciPy`

---

## 23. 35-Mark Evaluation Scheme
| Assessment Criteria | Marks |
| :--- | :---: |
| **Literature Review & Background** | 4 |
| **Data Preparation** | 5 |
| **Methodology & Implementation** | 11 |
| **Results & Evaluation** | 8 |
| **Report & Presentation** | 4 |
| **Critical Discussion** | 3 |
| **Total Marks** | **35** |

---

## 24. Expected Final Deliverables
* Modular end-to-end pipeline code repository ($\text{Segmentation} \rightarrow \text{Surface} \rightarrow \text{Tetrahedralisation} \rightarrow \text{Learned Correction} \rightarrow \text{FE Sanity Check}$).
* Trained GNN weights.
* Before/after mesh dataset.
* FE sanity check scripts and execution logs.
* Comparative project report with figures, plots, and tables.
* Final presentation slides.
* Live demo executing the full end-to-end pipeline on 2–3 sample volumes.

---

## 25. Key Meta-Information
* **Difficulty Level:** Research-oriented
* **Estimated Completion Time:** 14 weeks
* **Data Source:** Synthetic only. A controlled, exact-ground-truth spectrum of segmentation degradation is required to isolate and attribute failures reproducibly. No real CT scanner access is assumed or required.