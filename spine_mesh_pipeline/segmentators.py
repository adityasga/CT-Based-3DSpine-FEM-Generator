import tempfile, zipfile
import subprocess
import shutil
import nibabel as nib
import nibabel.processing
import numpy as np
from pathlib import Path
from .find_directory import source_dir, segmentation_dir, label_path, MODELS_DIR
from .labels_specs import Manifest_name, GROUND_TRUTH, Torch_Models, EXPORT_LABELS, GROUND_TRUTH_VB, VERTEBRAL_BODY_METHODS, REFERENCES

TORCH_SCRIPT = Path(__file__).resolve().parent.parent / "torch_segment.py"

def references_for(method):
    """The label maps a method is scored against: vertebral-body methods also against the body reference."""
    return [GROUND_TRUTH, GROUND_TRUTH_VB] if method in VERTEBRAL_BODY_METHODS else [GROUND_TRUTH]

def run_torch_model(out, method, cases, torch_python, revision, device="auto", fast=False, overwrite=False, vbseg_variant="ResEncL"):
    """Run torch_segment.py in the PyTorch environment (MPS on Apple, CPU locally, or CUDA if present)."""
    if method not in Torch_Models:
        raise SystemExit(f"'{method}' is not a PyTorch backend ({', '.join(Torch_Models)}); use `import` instead.")
    if not torch_python:
        raise SystemExit("No PyTorch interpreter found: pass --torch-python or set SEG_TORCH_PYTHON.")
    if not cases:
        raise SystemExit("No cases matched --cases / --limit.")
    cmd = [torch_python, str(TORCH_SCRIPT), method,
           "--manifest", str(Path(out) / Manifest_name),
           "--ct-root", str(source_dir(out)),
           "--out", str(segmentation_dir(out, method)),
           "--revision", revision, "--device", device,
           "--cases", *[c["volume_id"] for c in cases],
           "--label-ids", *[str(i) for i in EXPORT_LABELS]]  # the region set in labels.py
    if fast:
        cmd.append("--fast")
    if overwrite:
        cmd.append("--overwrite")
    if method in ("nvsegment","nvsegment-CT"):
        legacy = Path(out) / "NV-Segment-CTMR"
        cmd += ["--nvseg-repo", str(legacy if legacy.exists() else MODELS_DIR / "NV-Segment-CTMR")]
    if method == "vbsegm2step":
        cmd += ["--vbseg-models", str(MODELS_DIR / "vbsegm2step"), "--vbseg-variant", vbseg_variant]
    # stream the child's output through print() so it shows up in Jupyter cells, not only the terminal
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait():
        raise subprocess.CalledProcessError(proc.returncode, cmd)

def import_label_maps(out, method, cases, source):
    """Copy <volume_id>_label.nii.gz files (folder or zip, e.g. from Colab) into segmentations/<method>/.

    Each map is checked against the ground-truth grid; a map on a different grid is resampled
    (nearest neighbour) onto it so all approaches share one voxel grid.
    """
    if method in REFERENCES:
        raise SystemExit(f"{method} labels come from the dataset and cannot be imported.")
    dest = segmentation_dir(out, method); dest.mkdir(parents=True, exist_ok=True)
    source = Path(source)
    with tempfile.TemporaryDirectory() as tmp:
        if source.suffix == ".zip":
            with zipfile.ZipFile(source) as zf:
                zf.extractall(tmp)
            source = Path(tmp)
        found = {p.name: p for p in source.rglob("*_label.nii.gz")}
        for log in source.rglob("segmentation_log.csv"):
            shutil.copy2(log, dest / "segmentation_log.csv")
        report = []
        for case in cases:
            name = f"{case['volume_id']}_label.nii.gz"
            if name not in found:
                report.append((case["volume_id"], "missing"))
                continue
            img = nib.load(str(found[name]))
            gt = nib.load(str(label_path(out, GROUND_TRUTH, case)))
            status = "imported"
            if img.shape[:3] != gt.shape[:3] or not np.allclose(img.affine, gt.affine, atol=1e-3):
                img = nibabel.processing.resample_from_to(img, gt, order=0)
                status = "imported_resampled_to_gt_grid"
            data = np.rint(np.asanyarray(img.dataobj)).astype(np.uint8)
            out_img = nib.Nifti1Image(data, gt.affine, header=gt.header)
            out_img.set_data_dtype(np.uint8)
            nib.save(out_img, str(dest / name))
            report.append((case["volume_id"], status))
    return report

def zip_label_maps(out, methods, cases, archive=None):
    """Put every method's label maps (+ segmentation logs) into one zip: <method>/<volume_id>_label.nii.gz."""
    archive = Path(archive) if archive else Path(out) / "segmentations" / "label_maps.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:  # .nii.gz is already compressed
        for method in methods:
            for case in cases:
                path = label_path(out, method, case)
                if path.exists():
                    zf.write(path, f"{method}/{case['volume_id']}_label.nii.gz")
                    count += 1
            log = segmentation_dir(out, method) / "segmentation_log.csv"
            if method != GROUND_TRUTH and log.exists():
                zf.write(log, f"{method}/segmentation_log.csv")
    print(f"Wrote {archive} ({count} label maps, {archive.stat().st_size / 1e6:.0f} MB)")
    return archive