# Pipeline reference

> This is the detailed technical reference: the cleaning algorithm, the quality metrics and their
> thresholds, the accuracy metrics, and the full output layout. For installation and a copy-paste
> quickstart, start at [`../README.md`](../README.md) instead come back here for the "why".

Segments spine and pelvis CTs from three open datasets and compares segmentation approaches by the quality of the meshes they produce.
Every bone from every approach goes through the same cleaning, and every mesh is scored by the same metrics.
Each dataset gets its own `--out` folder with the same layout, so every later stage works the same on all of them.

## Datasets (`select --dataset`)

| `--dataset` | Source | Ground truth in the region (T12 → pelvis) | Notes |
|---|---|---|---|
| `ctspinopelvic1k` (default) | Hugging Face `OpenSpineConsortium/CTSpinoPelvic1K` | T12–L5, sacrum, S1, hips | CC BY-NC 4.0 |
| `verse2020` | [OSF t98fz](https://osf.io/t98fz) (VerSe 2020) | T12–L5 only (**no sacrum or hip labels**) | CC BY-SA 4.0; cite the three VerSe papers in the OSF readme |
| `totalseg_v3` | [Zenodo 22688904](https://zenodo.org/records/22688904) (TotalSegmentator dataset v3) | T12–L5, sacrum, hips (no S1 label) | CC BY 4.0; only the selected cases are read from the 37 GB zip (HTTP range requests). `--zip` uses a local copy instead |

Selection rules for the two new datasets follow the "normal" rules used for CTSpinoPelvic1K:

- every region bone is present and not cut by the scan edge;
- no L6 or T13 (TotalSegmentator has no T13 label, so T13 can't be screened there);
- no metal: at most 0.1 ml of voxels ≥ 2800 HU within 5 mm of the region bones;
- VerSe: Genant fracture grade 0 for every region vertebra (ungraded scans are rejected), and at least 20 mm of scan below L5. VerSe publishes no ages;
- TotalSegmentator: age ≥ 18, and no pathology located in the bones.

Every candidate and the reason it was accepted or rejected is in `eligibility_audit.csv`.

**Scans the models weren't trained on come first.** TotalSegmentator draws first from the 711 scans added in v3 (September 2026), which postdate the training of NV-Segment and vbsegm2step.
Recent TotalSegmentator model releases may have been trained on them.
VerSe draws first from vbsegm2step's held-out test split. NV-Segment-CTMR was trained partly on VerSe either way.
The manifest's `seen_in_training` column lists the models known to have trained on each selected scan.

## Approaches

| Approach | Where it runs | Label source |
|---|---|---|
| `ground_truth` | nowhere (dataset labels) | the dataset's labels (see above) |
| `ground_truth_vb` | nowhere (published labels) | vertebral bodies + sacrum from [VertebralBodiesCT-Labels](https://huggingface.co/datasets/fhofmann/VertebralBodiesCT-Labels), for VerSe and TotalSegmentator **v2** scans only |
| `nvsegment` | local MPS / CUDA / CPU, or Colab GPU (`colab/segment_on_colab.ipynb`, CTSpinoPelvic1K only) | NV-Segment-CTMR. **Non-commercial** weight license |
| `nvsegment_ct` | local MPS / CUDA / CPU | NV-Segment-CT, the CT-only sibling model from the same repo (same label IDs) |
| `totalsegmentator` | local, or Colab GPU | TotalSegmentator v2, `total` task, spine + pelvis ROI subset. CC BY 4.0 |
| `vbsegm2step` | local MPS / CUDA / CPU | [VBSegm2Step](https://github.com/fohofmann/vbsegm2step): two-step nnU-Net v2. **Vertebral bodies only** (T12–L5, no posterior arch) + sacrum; no hips or S1 |
| anything else | anywhere | `import --method <name> --source <dir or zip>` |

`vbsegm2step` STLs keep the vertebra names (`vertebra_L1.stl`, …) but are **vertebral bodies**: meshing expects genus 0 for them, not 1.
Against whole-vertebra `ground_truth`, their Dice and recall are low by construction. Read **precision** there, which is near 1 when the body sits in the right vertebra.
They are also scored against `ground_truth_vb` wherever those labels exist. Weights download on first use (ResEncL 5-fold ensemble ≈ 4.1 GB + neighbour model 0.5 GB; `--vbseg-variant ResEncM` is a single 0.8 GB model and about 5× faster).
Model weights live in `~/.cache/spine_mesh_pipeline/models`, outside the iCloud-synced project; override this with `SPINE_MODELS_DIR`.

## Notebooks

- **`label_and_cleaning_viewer.ipynb`**: view two methods' labels side by side on the CT, and the uncleaned vs. cleaned surface for one bone, with the voxels/area the cleaning step changed highlighted.
- **`learn_biphasic_meshing.ipynb`**: a worked, step-by-step walkthrough of `spine_mesh_pipeline/biphasic.py` (the cortical/trabecular two-material mesh), with inline explanations of each step.

There is currently no single notebook that runs the whole `select → segment → mesh → evaluate` pipeline end to end interactively (an earlier `pipeline_walkthrough.ipynb`, built before multi-dataset support, was retired. See the roadmap in the main [README](../README.md)). Until a multi-dataset version exists, drive the pipeline from the CLI as in [Quickstart](../README.md#quickstart) and use the two notebooks above for visual inspection of the results.

## Environments

- **Project venv (Python 3.14)**: `pip install -r requirements.txt`. This covers selection, meshing, evaluation and the report.
- **`seg_torch` conda env (Python 3.11)**: only needed to run models locally (PyTorch has no Python 3.14 build yet, which is what the project venv above uses):
  ```bash
  conda create -y -n seg_torch python=3.11 && conda activate seg_torch && pip install -r requirements-torch.txt
  ```

## Workflow

```bash
python run_pipeline.py select --out ctspinopelvic1k_100_normal --n 100      # reuses selected_manifest.csv if present
# Colab: upload torch_segment.py + selected_manifest.csv, run the notebook, download the zips
python run_pipeline.py import --out ctspinopelvic1k_100_normal --method nvsegment --source nvsegment_labels.zip
python run_pipeline.py import --out ctspinopelvic1k_100_normal --method totalsegmentator --source totalsegmentator_labels.zip
python run_pipeline.py mesh --out ctspinopelvic1k_100_normal --methods ground_truth nvsegment totalsegmentator --workers 3
python run_pipeline.py evaluate --out ctspinopelvic1k_100_normal --methods ground_truth nvsegment totalsegmentator
```

To run models locally instead of on Colab, list one or more (`--method` also still works):
`python run_pipeline.py segment --out ctspinopelvic1k_100_normal --methods nvsegment totalsegmentator vbsegm2step`.
They run in the order given, and each writes to `segmentations/<method>/`. If one method crashes, the rest still run; a summary then lists any missing cases, and the command exits with an error if any are missing or a method crashed.

Every stage resumes: finished cases are skipped unless you pass `--overwrite`. Use `--cases 0264 0535` or `--limit 5` to try a subset first.

## What happens to each bone (`mesh`)

1. **Voxel clean-up**: morphological closing (1 voxel), keep the largest face-connected component (islands removed), fill enclosed cavities. Labels under 0.5 ml are skipped as stray fragments.
2. **Surface**: marching cubes at iso-level 0.49 (just under 0.5, so ambiguous voxel diagonals don't punch spurious tunnels), in world mm, normals outward.
3. **Slicer smoothing**: `vtkWindowedSincPolyDataFilter`, 20 iterations, pass band 10^(−4·factor). These are the settings behind Slicer's *Smoothing factor* (default 0.5, `--smoothing`).
4. **Repair**: MeshFix (holes, non-manifold, self-intersections) when needed.
5. **Isotropic remeshing** to a target edge length equal to the CT in-plane voxel spacing (`--target-edge` to override). The remesher may move the surface by at most a quarter of an edge.
6. **Validation**: watertight, one shell, manifold, no self-intersections, outward normals.
7. **Tetrahedral check**: Gmsh HXT volume mesh bounded exactly by the STL (`--tet fast|relocate|netgen|none`).

## Metrics (per element; the CSV stores min / mean / p5 / p95 / max and the count failing)

| Metric | Triangles | Tetrahedra | Default limit |
|---|---|---|---|
| Skewness (equilateral-volume) | ✓ | ✓ | ≤ 0.85 tri, ≤ 0.95 tet |
| Aspect ratio (Verdict/VTK) | ✓ | ✓ | ≤ 5 tri, ≤ 10 tet |
| Size change (edge-length ratio to neighbours) | ✓ | ✓ | ≤ 2 |
| Scaled Jacobian (Verdict/VTK) | ✓ (signed vs. local normal) | ✓ | ≥ 0.2 |
| Inverted elements | folded triangles | SJ ≤ 0 | 0 |
| Resolution | fidelity HD95 to the unsmoothed segmentation surface | | ≤ 0.5 mm |

Thresholds are CLI options (`--tri-max-skewness`, `--min-scaled-jacobian`, …). The metric code matches VTK's `vtkMeshQuality` to machine precision.
`--save-element-metrics` writes every element's values to `.vtp`/`.vtu` for colouring in ParaView.

**Tet slivers:** the default `fast` tet mesher leaves roughly 0.003% of elements (tens per bone) with scaled Jacobian 0.15–0.2, so the strict tet criterion usually fails. That is a property of the mesher, not of the STL. `--tet netgen` clears it (all elements ≥ 0.25 in tests) but is about 25× slower, so use it for the final FE meshes.

## Accuracy vs ground truth (`evaluate`)

Voxel Dice per bone, plus HD95 and ASSD between each approach's cleaned mesh and the ground-truth cleaned mesh.
`sacrum_with_S1` scores sacrum and S1 together, because the dataset's S1 is a planar cut of the sacrum and the models cut it elsewhere.
Structures an approach finds outside the ground-truth field of view are listed as `not_in_ground_truth`.

## Outputs (under `--out`)

```
segmentations/<method>/<volume_id>_label.nii.gz
meshes/<method>/<NNN_volume_id>/individual_stl/<structure>.stl
meshes/<method>/<NNN_volume_id>/{spine,pelvis}_combined.stl     (multi-body; S1 is in the pelvis group)
meshes/<method>/<NNN_volume_id>/quality_rows.json
meshes/<method>/<NNN_volume_id>/biphasic/<bone>_biphasic.{inp,msh,vtu}   (`biphasic` command; .inp = Abaqus,
                                          element sets cortical/trabecular; .vtu = ParaView)
meshes/<method>/<NNN_volume_id>/biphasic/<bone>_trabecular_core.stl
evaluation/mesh_quality.csv            one row per mesh, all metrics + pass flags
evaluation/segmentation_accuracy.csv   one row per structure per case
evaluation/summary_by_method*.csv, accuracy_by_method*.csv
evaluation/biphasic_summary.csv        one row per bone (`biphasic` command): per-region volume, element
                                        quality, and mean CT intensity (HU) for cortical vs. trabecular
evaluation/report.html
```

## Notes

- **Laterality:** in every checked case, label 30 (`hip_left`) lies at patient-right according to the affine (RAS x < 0).
  TotalSegmentator and NV-Segment call the same bone "left", so comparisons are consistent. Check in Slicer before using left/right in a study.
- **Field of view:** `truncated_by_fov = True` marks bones cut by the scan edge (flat caps, not complete anatomy).
- **Licenses:** NV-Segment-CTMR weights are non-commercial. CTSpinoPelvic1K labels are CC BY-NC 4.0.
  NV-Segment-CTMR was trained partly on CTPelvic1K and VerSe, which may overlap this dataset, so its accuracy here may be optimistic.
