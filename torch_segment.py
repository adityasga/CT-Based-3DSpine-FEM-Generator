#!/usr/bin/env python3
"""Run a PyTorch segmentation model on CTSpinoPelvic1K CTs and write CTSpinoPelvic1K-ID label maps.

This file is deliberately standalone (numpy, nibabel, huggingface_hub + the model's own
package) so it can be uploaded on its own to Google Colab / a GPU box, or run locally from
the `seg_torch` conda env. The local mesh pipeline (run_pipeline.py) consumes its output.

Backends:
  nvsegment         NVIDIA NV-Segment-CTMR (MONAI bundle, https://github.com/NVIDIA-Medtech/NV-Segment-CTMR)
  nvsegment_ct      NVIDIA NV-Segment-CT: the CT-only model from the same repo (same label IDs)
  totalsegmentator  TotalSegmentator v2 "total" task restricted to spine + pelvis classes
  vbsegm2step       VBSegm2Step (https://github.com/fohofmann/vbsegm2step): nnU-Net v2 whole-spine model, then a
                    neighbour model walking vertebra by vertebra. VERTEBRAL BODIES ONLY (T1-L5) + sacrum; no
                    posterior arch, hips or S1. Runs on MPS / CUDA / CPU; weights download on first use.

CTs: read from --ct-root/<ct_file>. Only CTSpinoPelvic1K CTs are downloaded on demand (Hugging Face);
VerSe and TotalSegmentator CTs are downloaded by `run_pipeline.py select`.

Region: only the structures in --label-ids are segmented (default T12 to the pelvis:
T12, L1-L5, sacrum, S1, both hips); `--label-ids all` segments the whole spine.

Output (one file per case, same grid/affine as the CT):
  <out>/<volume_id>_label.nii.gz      uint8, CTSpinoPelvic1K label IDs (1-24, 26, 29, 30, 31)
  <out>/segmentation_log.csv          runtime / status per case

Examples:
  # Colab / GPU box (CTs are downloaded from Hugging Face on demand)
  python torch_segment.py nvsegment --manifest selected_manifest.csv --out nvsegment_labels --zip
  python torch_segment.py totalsegmentator --manifest selected_manifest.csv --out totalseg_labels --zip

  # local CPU, re-using CTs already downloaded by run_pipeline.py
  python torch_segment.py totalsegmentator --manifest ctspinopelvic1k_100_normal/selected_manifest.csv \
      --ct-root ctspinopelvic1k_100_normal/source_nifti --out ctspinopelvic1k_100_normal/segmentations/totalsegmentator
"""
import argparse
import csv
import os
import io
import shutil
import subprocess
import functools
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

# Ops that Apple's MPS backend lacks run on the CPU instead of failing; must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REPO = "OpenSpineConsortium/CTSpinoPelvic1K"
NVSEG_GIT = "https://github.com/NVIDIA-Medtech/NV-Segment-CTMR.git"

# CTSpinoPelvic1K id -> (structure name, NV-Segment-CTMR label id, TotalSegmentator class name).
# Must stay in sync with spine_mesh_pipeline/labels.py (checked by that module's self-test).
TARGETS = {}
for _i in range(1, 8):
    TARGETS[_i] = (f"vertebra_C{_i}", 57 - _i, f"vertebrae_C{_i}")
for _i in range(1, 13):
    TARGETS[7 + _i] = (f"vertebra_T{_i}", 50 - _i, f"vertebrae_T{_i}")
for _i in range(1, 6):
    TARGETS[19 + _i] = (f"vertebra_L{_i}", 38 - _i, f"vertebrae_L{_i}")
TARGETS.update({
    26: ("sacrum", 97, "sacrum"),
    29: ("vertebra_S1", 127, "vertebrae_S1"),
    30: ("hip_left", 95, "hip_left"),
    31: ("hip_right", 96, "hip_right"),
})
# Default region (CTSpinoPelvic1K IDs): T12, L1-L5, sacrum, S1, left hip, right hip.
# run_pipeline.py passes the region from spine_mesh_pipeline/labels.py (REGION_IDS) instead.
T12_TO_PELVIS = [19, 20, 21, 22, 23, 24, 26, 29, 30, 31]

def resolve_ct(ct_file, ct_root, revision, dataset = "ctspinopelvic1k"):
    local = Path(ct_root) / ct_file
    if local.exists():
        return local
    if dataset != "ctspinopelvic1k":
        raise FileNotFoundError(f"{local} missing; {dataset} CTs are downloaded by `run_pipeline.py select`")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=REPO, repo_type="dataset", filename=ct_file,
                                local_dir=str(ct_root), revision=revision))


def pick_device(requested):
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return "mps"
    return "cpu"


def save_label(labels, ct_img, path):
    out = nib.Nifti1Image(labels.astype(np.uint8), ct_img.affine)
    out.set_qform(ct_img.affine, code=int(ct_img.header["qform_code"]) or 1)
    out.set_sform(ct_img.affine, code=int(ct_img.header["sform_code"]) or 1)
    out.set_data_dtype(np.uint8)
    nib.save(out, str(path))


def check_grid(img, ct_img, what):
    if img.shape[:3] != ct_img.shape[:3] or not np.allclose(img.affine, ct_img.affine, atol=1e-3):
        raise RuntimeError(f"{what} output grid {img.shape} does not match CT grid {ct_img.shape}")


def ensure_nvseg_bundle(repo_dir, bundle_name):
    repo_dir = Path(repo_dir)
    if not (repo_dir / "NV-Segment-CTMR" / "configs" / "inference.json").exists():
        subprocess.run(["git", "clone", "--depth", "1", NVSEG_GIT, str(repo_dir)], check=True)
    return repo_dir / bundle_name


def run_nvsegment(ct_path, ct_img, work, args, device, bundle_name ="NV-Segment-CTMR"):
    bundle = ensure_nvseg_bundle(args.nvseg_repo, bundle_name).resolve()
    prompt = sorted(nv for _, nv, _ in TARGETS.values())
    cmd = [sys.executable, "-m", "monai.bundle", "run",
           "--config_file", "configs/inference.json",
           "--bundle_root", str(bundle),
           "--output_dir", str(work),
           "--input_dict", repr({"image": str(Path(ct_path).resolve()), "label_prompt": prompt})]
    if device in ("cpu", "mps"):  # the bundle only auto-selects CUDA; mixed precision is CUDA-only
        cmd += ["--device", f"$torch.device('{device}')", "--evaluator#amp", "False"]
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}  # run ops MPS lacks on the CPU
    subprocess.run(cmd, cwd=str(bundle), check=True, env=env)
    outputs = sorted(Path(work).rglob("*_trans.nii.gz"))
    if not outputs:
        raise RuntimeError("NV-Segment produced no *_trans.nii.gz output")
    pred_img = nib.load(str(outputs[0]))
    check_grid(pred_img, ct_img, "NV-Segment")
    pred = np.rint(np.asanyarray(pred_img.dataobj)).astype(np.int32)
    labels = np.zeros(ct_img.shape[:3], dtype=np.uint8)
    for sp_id, (_, nv_id, _) in TARGETS.items():
        labels[pred == nv_id] = sp_id
    return labels


def run_totalsegmentator(ct_path, ct_img, work, args, device):
    exe = Path(sys.executable).parent / "TotalSegmentator"
    exe = str(exe) if exe.exists() else shutil.which("TotalSegmentator")
    if not exe:
        raise RuntimeError("TotalSegmentator executable not found; pip install TotalSegmentator")
    cmd = [exe, "-i", str(ct_path), "-o", str(work), "--device", "gpu" if device == "cuda" else device,
           "--roi_subset", *[ts for _, _, ts in TARGETS.values()]]
    if args.fast:
        cmd.append("--fast")
    subprocess.run(cmd, check=True)
    labels = np.zeros(ct_img.shape[:3], dtype=np.uint8)
    for sp_id, (_, _, ts_name) in TARGETS.items():
        mask_path = Path(work) / f"{ts_name}.nii.gz"
        if not mask_path.exists():
            continue
        mask_img = nib.load(str(mask_path))
        check_grid(mask_img, ct_img, "TotalSegmentator")
        labels[np.asanyarray(mask_img.dataobj) > 0] = sp_id
    return labels

# vbsegm2step label -> CTSpinoPelvic1K id: T1-T12, L1-L5, sacrum (its L6 18, coccyx 20 and T13 21 have no target)
VBSEG_TO_SPINO = {**{i: 7 + i for i in range(1, 13)}, **{12 + i: 19 + i for i in range(1, 6)}, 19: 26}
# Hugging Face repo, local folder, nnU-Net trainer folder, folds. Task 601 = whole spine, 602 = neighbours.
VBSEG_MODELS = {
    "ResEncL": ("fhofmann/VertebralBodiesCT-ResEncL", "Dataset601_VertebralBodies_ResEncL",
                "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres", [f"fold_{i}" for i in range(5)]),
    "ResEncM": ("fhofmann/VertebralBodiesCT-ResEncM", "Dataset601_VertebralBodies_ResEncM",
                "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres", ["fold_all"]),
    "Neighbors": ("fhofmann/VertebralBodiesCT-Neighbors", "Dataset602_VertebralBodiesNeighbors",
                  "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres", ["fold_all"]),
}
VBSEG_MODELS_DIR = Path(os.environ.get("SPINE_MODELS_DIR", Path.home() / ".cache" / "spine_mesh_pipeline" / "models")) / "vbsegm2step"


def ensure_vbseg_model(models_dir, key):
    """Download only what nnU-Net inference reads (plans, dataset definition, final checkpoints)."""
    repo, folder, trainer, folds = VBSEG_MODELS[key]
    root = Path(models_dir) / folder
    files = [f"{trainer}/plans.json", f"{trainer}/dataset.json"] + [f"{trainer}/{f}/checkpoint_final.pth" for f in folds]
    missing = [f for f in files if not (root / f).exists()]
    if missing:
        from huggingface_hub import hf_hub_download
        print(f"    downloading {repo} ({len(missing)} files) to {root}", flush=True)
        for f in missing:
            hf_hub_download(repo, f, local_dir=str(root))
    return root


@functools.lru_cache(maxsize=None)
def vbseg_pipeline(models_dir, variant, device):
    """Load both nnU-Net models once per device; later cases reuse them."""
    import torch
    from vbsegm2step import Config, VBSegm2StepPipeline
    config = Config.from_overrides(model601=ensure_vbseg_model(models_dir, variant),
                                   model602=ensure_vbseg_model(models_dir, "Neighbors"), model601_variant=variant)
    config.DEVICE = torch.device(device)  # the package itself only picks CUDA or CPU
    return VBSegm2StepPipeline(config)


def run_vbsegm2step(ct_path, ct_img, work, args, device):
    import nibabel.processing
    # vbsegm2step reads with SimpleITK. Give it a copy whose qform and sform both hold nibabel's affine,
    # so both libraries place the CT, and therefore the labels, in the same world position.
    ct_copy = Path(work) / "ct.nii"
    img = nib.Nifti1Image(ct_img.dataobj, ct_img.affine, header=ct_img.header)
    img.set_qform(ct_img.affine, code=1)
    img.set_sform(ct_img.affine, code=1)
    nib.save(img, str(ct_copy))
    seg_path = Path(work) / "vbseg.nii.gz"
    result = vbseg_pipeline(args.vbseg_models, args.vbseg_variant, device).process_file(ct_copy, seg_path)
    if result is False or not seg_path.exists():
        raise RuntimeError("vbsegm2step found no processable vertebra or failed (see its log above)")
    pred_img = nib.load(str(seg_path))
    if pred_img.shape[:3] != ct_img.shape[:3] or not np.allclose(pred_img.affine, ct_img.affine, atol=1e-3):
        pred_img = nibabel.processing.resample_from_to(pred_img, ct_img, order=0)
    pred = np.rint(np.asanyarray(pred_img.dataobj)).astype(np.int32)
    labels = np.zeros(ct_img.shape[:3], dtype=np.uint8)
    for vb_id, sp_id in VBSEG_TO_SPINO.items():
        if sp_id in args.targets:
            labels[pred == vb_id] = sp_id
    return labels


BACKENDS = {"nvsegment": run_nvsegment,
            "nvsegment_ct": functools.partial(run_nvsegment, bundle_name="NV-Segment-CT"),
            "totalsegmentator": run_totalsegmentator,
            "vbsegm2step": run_vbsegm2step}


def segment(method, ct_path, ct_img, args, device):
    with tempfile.TemporaryDirectory(prefix=f"{method}_") as work:
        return BACKENDS[method](ct_path, ct_img, work, args, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("method", choices=sorted(BACKENDS))
    parser.add_argument("--manifest", required=True, help="CSV with volume_id and ct_file columns (selected_manifest.csv)")
    parser.add_argument("--out", required=True, help="output folder for <volume_id>_label.nii.gz")
    parser.add_argument("--ct-root", default="ct_cache", help="folder holding (or receiving) the CT files")
    parser.add_argument("--revision", default="main", help="dataset revision / commit hash")
    parser.add_argument("--cases", nargs="*", help="only these volume_ids")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--overwrite", action="store_true", help="re-run cases whose output already exists")
    parser.add_argument("--nvseg-repo", default="NV-Segment-CTMR", help="clone location of the NV-Segment repo")
    parser.add_argument("--fast", action="store_true", help="TotalSegmentator 3 mm model (much faster, coarser)")
    parser.add_argument("--vbseg-models", default=str(VBSEG_MODELS_DIR), help="vbsegm2step weights folder")
    parser.add_argument("--vbseg-variant", default="ResEncL", choices=["ResEncL", "ResEncM"],
                        help="vbsegm2step whole-spine model: ResEncL (5-fold ensemble, default) or ResEncM (1 model, faster)")
    parser.add_argument("--zip", default = True, action="store_true", help="zip the label maps into <out>.zip when done"),
    parser.add_argument("--label-ids", nargs="+", default=[str(i) for i in T12_TO_PELVIS],
                        help="CTSpinoPelvic1K IDs to segment (default T12 to pelvis), or `all`")
    args = parser.parse_args()
    ids = list(TARGETS) if args.label_ids == ["all"] else [int(i) for i in args.label_ids]
    args.targets = {i: TARGETS[i] for i in ids}

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    rows = list(csv.DictReader(open(args.manifest, newline="", encoding="utf-8")))
    if args.cases is not None:
        rows = [r for r in rows if r["volume_id"] in set(args.cases)]
    log_path = out / "segmentation_log.csv"
    new_log = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as log_file:
        log = csv.DictWriter(log_file, fieldnames=["volume_id", "method", "device", "status", "seconds", "structures"])#the titles of the features in the log file
        if new_log:
            log.writeheader()
        for index, row in enumerate(rows, 1):
            volume_id = row["volume_id"]
            target = out / f"{volume_id}_label.nii.gz"#generating the name of the label file corresponding to the CT file, to ensure consistency in naming between the various methods
            if target.exists() and not args.overwrite:#avoids regenerating the same label if already exists
                print(f"[{index}/{len(rows)}] {volume_id}: exists, skipping", flush=True)
                continue
            print(f"[{index}/{len(rows)}] {volume_id}: {args.method} on {device}", flush=True)
            start = time.time()
            entry = {"volume_id": volume_id, "method": args.method, "device": device, "status": "", "seconds": "", "structures": ""}
            try:
                ct_path = resolve_ct(row["ct_file"], args.ct_root, args.revision)
                ct_img = nib.load(str(ct_path))
                args.current_ct_file = row["ct_file"]
                try:
                    labels = segment(args.method, ct_path, ct_img, args, device)
                except Exception as exc:
                    if device != "mps":
                        raise
                    print(f"    MPS run failed ({type(exc).__name__}); retrying on CPU", flush=True)
                    labels = segment(args.method, ct_path, ct_img, args, "cpu")#generate the labels
                    entry["device"] = "cpu (mps failed)"
                save_label(labels, ct_img, target)
                found = [TARGETS[i][0] for i in np.unique(labels).tolist() if i in TARGETS]
                entry.update(status="completed", structures=";".join(found))
            except Exception as exc:
                entry["status"] = f"failed: {type(exc).__name__}: {exc}"
                print(f"    {entry['status']}", flush=True)
            entry["seconds"] = round(time.time() - start, 1)
            log.writerow(entry); log_file.flush()

    if args.zip:
        archive = out.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
            for path in sorted(out.glob("*_label.nii.gz")) + [log_path]:
                zf.write(path, path.name)
        print(f"Wrote {archive}")


if __name__ == "__main__":
    main()




