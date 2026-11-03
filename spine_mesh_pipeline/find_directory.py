from pathlib import Path
from .labels_specs import GROUND_TRUTH, GROUND_TRUTH_VB
import os

MODELS_DIR = Path(os.environ.get("SPINE_MODELS_DIR", Path.home() / ".cache" / "spine_mesh_pipeline" / "models"))

def source_dir(out):
    return Path(out)/"source_nifti"

def case_dir_name(case):
    return f"{case['sample_index']:03d}_{case['volume_id']}"

def label_path(out, method, case):
    if method == GROUND_TRUTH:
        return source_dir(out) / case["label_file"]
    if method == GROUND_TRUTH_VB:
        return source_dir(out) / (case.get("label_vb_file") or f"labels_vb/{case['volume_id']}_label.nii.gz")
    return segmentation_dir(out, method) / f"{case['volume_id']}_label.nii.gz"

def segmentation_dir(out, method):
    return Path(out) / "segmentations" / method

def mesh_dir(out, method, case):
    return Path(out) / "meshes" / method / case_dir_name(case)


