# CT Spine & Pelvis → 3D Bone Mesh Pipeline

![License: Research & Non-Commercial Use Only](https://img.shields.io/badge/license-research%20%26%20non--commercial-lightgrey.svg)
![Python 3.11–3.14](https://img.shields.io/badge/python-3.11%E2%80%933.14-blue.svg)
![Status: active development](https://img.shields.io/badge/status-active%20development-orange.svg)

> **License, in short (see [`LICENSE`](LICENSE) for the full terms):** free to use for research and
> academic work: please let the author know you're using it (a24mitha@uwaterloo.ca) and cite it
> ([`CITATION.cff`](CITATION.cff)) in anything you publish. **Commercial use is not permitted without
> the author's prior written approval** : contact the same address first.

Turns open, public CT scans of the spine and pelvis into **clean, watertight, finite-element-ready 3D
bone meshes** and lets you compare several automatic segmentation methods against each other, and
against each dataset's own published labels, on the quality of the mesh each one produces.

```
CT scan (NIfTI)  →  segmentation (label map)  →  cleaned watertight STL  →  quality-checked tet mesh
```

> **In development: bone material properties from CT density.** The next milestone is assigning each
> element its own material properties (e.g. elastic modulus) from the CT's Hounsfield-unit density via
> an empirical density–modulus relationship, and exporting ready-to-solve FE material cards (Abaqus /
> FEBio). The groundwork, a two-material (cortical shell / trabecular core) mesh that already samples
> CT intensity per element (`spine_mesh_pipeline/biphasic.py`), exists today; the density → material
> mapping and per-element material-card export do not yet. See [Roadmap](#roadmap--in-development) and
> [`docs/PIPELINE_REFERENCE.md`](docs/PIPELINE_REFERENCE.md).

## Contents

- [What this does](#what-this-does)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Pipeline stages](#pipeline-stages)
- [Choosing a segmentation method](#choosing-a-segmentation-method)
- [Datasets](#datasets)
- [Adding your own segmentation method](#adding-your-own-segmentation-method)
- [Notebooks](#notebooks)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Roadmap / in development](#roadmap--in-development)
- [Before you trust the numbers](#before-you-trust-the-numbers)
- [Citing this work](#citing-this-work)
- [License](#license)

## What this does

1. **Selects** CT cases from one of three public datasets, applying strict "normal spine, no hardware,
   nothing cut off" screening rules, with a full audit trail of every accept/reject decision.
2. **Segments** each CT with one or more automatic methods (five are built in - see
   [Choosing a segmentation method](#choosing-a-segmentation-method)) or lets you bring your own.
3. **Cleans** every segmented bone into a single watertight surface: morphological clean-up, the same
   smoothing 3D Slicer uses, topology repair (closing spurious tunnels, sealing holes), and isotropic
   remeshing to a target element size.
4. **Checks mesh quality** on every triangle and tetrahedron (skewness, aspect ratio, scaled Jacobian,
   size-change ratio, inverted elements) against configurable pass/fail thresholds, and checks
   **segmentation accuracy** against each dataset's published labels (Dice, surface distance).
5. **Reports** everything in one self-contained `report.html`, plus per-structure CSVs for your own
   analysis.
6. **Meshes cortical and trabecular bone separately** (`biphasic.py`) as one conforming two-material
   tetrahedral mesh, the stepping stone to assigning density-based material properties (see
   [Roadmap](#roadmap--in-development)).

Everything is designed to **resume**: re-running any stage skips work that is already on disk unless you
pass `--overwrite`, so you can stop, add a case, or try a new method without redoing finished work.

## Repository layout

```
.
├── run_pipeline.py              CLI entry point: start here (python run_pipeline.py -h)
├── torch_segment.py             standalone runner for the PyTorch/API segmentation backends
├── view_cleaning.py             rotatable 3D viewer: uncleaned vs. cleaned geometry, one script
├── spine_mesh_pipeline/         the library every stage is built from
│   ├── dataset.py, verse.py,    case selection + screening, one module per dataset
│   │   totalseg.py, screening.py
│   ├── sources.py               download helpers (plain files, OSF, ranged remote-zip reads)
│   ├── segmenters.py            label-map bookkeeping: paths, import, zip, PyTorch-env dispatch
│   ├── meshing.py               label map → clean watertight surface
│   ├── quality.py               per-element mesh-quality metrics (matches VTK's vtkMeshQuality)
│   ├── pipeline.py              orchestrates meshing + accuracy, one job per (method, case)
│   ├── biphasic.py              cortical shell / trabecular core two-material tet mesh
│   ├── report.py                HTML report + summary CSVs
│   └── viewer.py                label-agreement and cleaning-diff viewers (used by the notebooks)
├── colab/segment_on_colab.ipynb GPU segmentation on Google Colab (CTSpinoPelvic1K only)
├── label_and_cleaning_viewer.ipynb   inspect labels + cleaning effects interactively
├── learn_biphasic_meshing.ipynb      step-by-step walkthrough of the biphasic mesh
├── docs/PIPELINE_REFERENCE.md   full technical reference: algorithm, metrics, output schema
├── legacy/                      superseded exploratory notebooks/scripts, kept for history
├── requirements.txt             the project environment (selection, meshing, evaluation, report)
└── requirements-torch.txt       the separate PyTorch environment (segmentation models)
```

## Requirements

- **Python 3.11–3.14**, in **two separate environments** (why: below).
- **~10–20 GB free disk** per 10–20 cases you work with (CTs + label maps + meshes); more for larger runs.
- **A GPU is optional but strongly recommended** for segmentation: Apple Silicon (MPS), NVIDIA (CUDA), or
  Google Colab. Everything also runs on CPU, just much slower (see [Troubleshooting](#troubleshooting)).
- `git`, for cloning the NV-Segment model repo on first use.

## Installation

```bash
git clone https://github.com/adityasga/Spine_Model_Segmentation_Pipeline
cd open_spine_data
```

**1. Project environment**: case selection, mesh cleaning, quality metrics, the report. Needs no GPU.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Segmentation environment**: runs the PyTorch models. Separate because PyTorch has no build yet
for the Python version above; this one stays on Python 3.11.

```bash
conda create -y -n seg_torch python=3.11
conda activate seg_torch
pip install -r requirements-torch.txt
```

(No `conda`? Any Python-3.11 virtualenv works, just point `--torch-python` at its interpreter, or set
`SEG_TORCH_PYTHON`, when you run the `segment` command below.)

You do **not** need the second environment if you only plan to import label maps produced elsewhere
(Colab, a colleague, another tool) with `run_pipeline.py import`.

Model weights (NV-Segment, VBSegm2Step, several GB) are cached outside the repo by default, at
`~/.cache/spine_mesh_pipeline/models`, so they are never accidentally committed or synced by cloud-drive
tools watching this folder. Override the location with `SPINE_MODELS_DIR` if you'd rather put them
somewhere else (e.g. a faster or larger disk).

## Quickstart

Everything below runs from the **project environment only** (`source venv/bin/activate`), including
`segment`. It finds the `seg_torch` interpreter itself and runs the model in it as a subprocess; you
never activate `seg_torch` by hand (that's what `default_torch_python()` / `--torch-python` /
`SEG_TORCH_PYTHON` are for).

A first, small run end to end, five CT scans, one segmentation method (TotalSegmentator: no API key,
no non-commercial restriction, moderate hardware needs), compared against the dataset's own labels.

```bash
source venv/bin/activate

# 1. Pick 5 "normal" cases from the default dataset (CTSpinoPelvic1K) and download their labels
python run_pipeline.py select --out my_first_run --n 5

# 2. Segment them (runs totalsegmentator in the seg_torch environment, found automatically).
#    To run several models in one go, list them: --methods nvsegment totalsegmentator vbsegm2step
python run_pipeline.py segment --out my_first_run --methods totalsegmentator

# 3. Clean every bone into a watertight, quality-checked mesh, for both the dataset's labels
#    and TotalSegmentator's
python run_pipeline.py mesh --out my_first_run --methods ground_truth totalsegmentator

# 4. Score accuracy + mesh quality and build the report
python run_pipeline.py evaluate --out my_first_run --methods ground_truth totalsegmentator

# 5. Open it
open my_first_run/evaluation/report.html        # macOS; use `start` on Windows, `xdg-open` on Linux
```

Every command accepts `-h` for its exact options (`python run_pipeline.py segment -h`, etc.), and every
stage **resumes**: re-run the same command and finished cases are skipped. Use `--cases 0264 0535` or
`--limit 2` on any command to work on a smaller subset while you're trying things out.

Segmenting a CT can take anywhere from under a minute (GPU) to tens of minutes (CPU) per case per
method, see [Troubleshooting](#troubleshooting) for what to expect and how to use a GPU, including a
free one on Google Colab.

## Pipeline stages

| Command | What it does |
|---|---|
| `select` | Pick N "normal" cases from a dataset, with full screening audit (`eligibility_audit.csv`) |
| `segment` | Run a built-in PyTorch/API segmentation method locally |
| `import` | Bring in label maps made elsewhere (Colab, another tool); any method name works |
| `mesh` | Clean every bone into a watertight STL + per-element quality metrics |
| `biphasic` | Split each vertebra's mesh into a cortical shell + trabecular core (one conforming tet mesh) |
| `zip` | Package every method's label maps into one `.zip` (e.g. to move between machines) |
| `evaluate` | Score accuracy vs. ground truth, build summary CSVs and `report.html` |

Full detail on each stage: the exact cleaning algorithm, every quality metric's formula and default
threshold, and the complete output file schema is in
[`docs/PIPELINE_REFERENCE.md`](docs/PIPELINE_REFERENCE.md).

## Choosing a segmentation method

| `--method` | Runs on | Notes |
|---|---|---|
| `totalsegmentator` | local GPU/CPU, or Colab | Easiest to start with: fast, no API key, CC-BY-licensed |
| `nvsegment` | local GPU/CPU, or Colab | NVIDIA NV-Segment-CTMR; **non-commercial** weight license |
| `nvsegment_ct` | local GPU/CPU | NV-Segment-CT, the CT-only sibling model |
| `vbsegm2step` | local GPU/CPU | Vertebral bodies only (no posterior arch, no hips); ~5 GB of weights |
| *your own* | anywhere | `run_pipeline.py import --method <name> --source <folder-or-zip>` |

Full licensing notes, expected runtimes, and `vbsegm2step`'s vertebral-body-only scoring caveats are in
[`docs/PIPELINE_REFERENCE.md`](docs/PIPELINE_REFERENCE.md#approaches).

## Datasets

```bash
python run_pipeline.py select --dataset <name> --out <folder> --n <count>
```

| `--dataset` | Source | License |
|---|---|---|
| `ctspinopelvic1k` (default) | [CTSpinoPelvic1K](https://huggingface.co/datasets/OpenSpineConsortium/CTSpinoPelvic1K) | CC BY-NC 4.0 |
| `verse2020` | [VerSe 2020](https://osf.io/t98fz) | CC BY-SA 4.0 |
| `totalseg_v3` | [TotalSegmentator dataset v3](https://zenodo.org/records/22688904) | CC BY 4.0 |

Each gets its own `--out` folder with the same layout, so every later stage works the same regardless of
which dataset you picked. See [`docs/PIPELINE_REFERENCE.md`](docs/PIPELINE_REFERENCE.md#datasets-select---dataset)
for the exact screening rules, what ground truth each dataset provides, and citation details.

## Adding your own segmentation method

You do not need to touch the pipeline code. Produce one NIfTI label map per case, using the
CTSpinoPelvic1K label IDs (`spine_mesh_pipeline/labels.py`), named `<volume_id>_label.nii.gz`
(ideally on the same grid as the CT, `import` resamples it onto that grid automatically if not),
then:

```bash
python run_pipeline.py import --out my_first_run --method my_algorithm --source /path/to/label/maps
python run_pipeline.py mesh --out my_first_run --methods ground_truth my_algorithm
python run_pipeline.py evaluate --out my_first_run --methods ground_truth my_algorithm
```

It now appears in the report alongside every built-in method. To add a method that runs *inside* this
repo's PyTorch environment instead, add one function to `torch_segment.py` (see its existing backends)
and register it in `TORCH_MODELS` in `spine_mesh_pipeline/segmenters.py`.

## Notebooks

- **`label_and_cleaning_viewer.ipynb`**: two methods' labels side by side on the CT slices, plus a
  rotatable 3D view of what the cleaning step changed for one bone (also available as the standalone
  script `view_cleaning.py`).
- **`learn_biphasic_meshing.ipynb`**: an annotated, step-by-step build-up of the cortical/trabecular
  mesh in `biphasic.py`.
- **`colab/segment_on_colab.ipynb`**: run `nvsegment` / `totalsegmentator` on a free Colab GPU
  (CTSpinoPelvic1K cases only) and bring the results back with `import`.

## Outputs

```
<out>/
├── selected_manifest.csv, eligibility_audit.csv   which cases, and why they were chosen
├── source_nifti/                                  downloaded CTs and ground-truth labels
├── segmentations/<method>/<volume_id>_label.nii.gz
├── meshes/<method>/<case>/individual_stl/<bone>.stl   + spine_combined.stl, pelvis_combined.stl
├── meshes/<method>/<case>/biphasic/                   cortical/trabecular .inp / .msh / .vtu
└── evaluation/
    ├── report.html                                 open this first
    ├── mesh_quality.csv, segmentation_accuracy.csv  one row per mesh / structure
    └── summary_by_method*.csv, accuracy_by_method*.csv
```

`.inp` meshes open directly in Abaqus (element sets `cortical` / `trabecular`); `.vtu` / `.vtp` open in
ParaView. Full schema: [`docs/PIPELINE_REFERENCE.md`](docs/PIPELINE_REFERENCE.md#outputs-under---out).

## Troubleshooting

Issues that came up most often while building and running this on real machines:

- **A fresh `seg_torch` env throws `ModuleNotFoundError` or an `AttributeError` deep inside `monai` /
  `fire` / `ignite`.** `pip install -r requirements-torch.txt` once more, in a *clean* environment
  (`conda env remove -n seg_torch && conda create ...`) rather than reinstalling over one that has had
  packages added and removed by hand, a half-upgraded `torch` or `fire` install is the usual cause, and
  reinstalling over it does not reliably fix it.
- **`segment` says "retrying on CPU" with an MPS / float64 error.** Expected on Apple Silicon for some
  preprocessing ops; it is handled automatically (the case finishes on CPU, a few seconds slower to
  start). Check `segmentation_log.csv`'s `device` column, `cpu (mps failed)` means this happened.
- **`evaluate` gives identical numbers for two different methods.** `evaluate` now checks for this and
  prints a warning (and flags it in `report.html`) when two methods' label maps are byte-identical, or
  when a mesh was built from an older/different label map than the one currently on disk. Re-run `mesh
  --overwrite` for the affected method and evaluate again.
- **Segmentation is very slow.** CPU-only runs can take minutes to tens of minutes per case per method.
  Use a GPU (`--device cuda` / `mps`, the default `auto` picks one automatically), or
  `colab/segment_on_colab.ipynb` for a free GPU with no local setup.

## Roadmap / in development

- **Bone material properties from CT density** (the main item: see the callout at the top). Planned:
  an empirical HU → ash density → elastic modulus relationship (e.g. Rho / Carter–Hayes-style), applied
  per element of the biphasic mesh, exported as per-element Abaqus/FEBio material cards. Today,
  `biphasic.py` already samples each element's CT intensity as a first diagnostic step
  (`ct_at_elements()`), the density/modulus mapping and material-card export are not yet implemented.
- A multi-dataset version of an end-to-end, run-every-stage-and-inspect-the-output walkthrough notebook
  (an earlier single-dataset one was retired when multi-dataset support was added).
- Automated tests for the mesh-quality metrics and the cleaning pipeline (currently verified by hand
  against VTK's `vtkMeshQuality` and real cases, see [`docs/PIPELINE_REFERENCE.md`](docs/PIPELINE_REFERENCE.md)).

Contributions toward any of these are welcome: open an issue or a pull request.

## Before you trust the numbers

A few things worth checking before drawing conclusions from a run (full detail in
[`docs/PIPELINE_REFERENCE.md`](docs/PIPELINE_REFERENCE.md#notes)):

- **Laterality.** In this label convention, "left" follows the label name each method uses, it is
  consistent across methods, but verify against a viewer (e.g. 3D Slicer) before reporting left/right
  findings.
- **Sacrum / S1 boundary.** Datasets and models draw this boundary differently; `evaluate` also reports
  sacrum and S1 combined (`sacrum_with_S1`) for a fairer comparison.
- **Tetrahedral slivers.** The default fast tet mesher leaves a small fraction of low-quality elements.
  Use `--tet netgen` (slower) for meshes you intend to solve, not just screen.
- **`vbsegm2step`** segments vertebral bodies only, low recall against whole-vertebra ground truth is
  expected; read its precision and its score against `ground_truth_vb` instead.

## Citing this work

Research use of this pipeline requires citing it (see [`LICENSE`](LICENSE)). Cite it as given in
[`CITATION.cff`](CITATION.cff) (GitHub renders a ready-to-copy citation from this file once the repo is
pushed), or:

> Mithani, A. (2026). *CT Spine & Pelvis → 3D Bone Mesh Pipeline* [Computer software].

Please also cite the datasets and models you used. See
[`docs/PIPELINE_REFERENCE.md`](docs/PIPELINE_REFERENCE.md#datasets-select---dataset) for the exact
citations CTSpinoPelvic1K, VerSe 2020, TotalSegmentator, NV-Segment, and VBSegm2Step each ask for.

## License

This code is released for **research and academic use only**: free to use, modify, and share for
non-commercial research, provided you let the author know
([mcgillspacemed@gmail.com](mailto:mcgillspacemed@gmail.com)) and cite the work in anything you
publish. **Commercial use is not permitted** without the author's prior written approval. Contact
the same address to request it. Full terms: [`LICENSE`](LICENSE).

The datasets and model weights the code downloads keep their own, separate licenses, several are
themselves non-commercial or research-only. See [`LICENSE`](LICENSE) and
[`docs/PIPELINE_REFERENCE.md`](docs/PIPELINE_REFERENCE.md) before any use, commercial or otherwise.
