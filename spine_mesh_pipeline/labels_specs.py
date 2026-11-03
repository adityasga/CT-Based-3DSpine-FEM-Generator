import os
from pathlib import Path
from spine_mesh_pipeline import totalseg, verse

DEFAULT_DATASET = "ctspinopelvic1k"
DATASETS = [DEFAULT_DATASET, verse.DATASET, totalseg.DATASET]
REPO = "OpenSpineConsortium/CTSpinoPelvic1K"
excluded_list ={
    25: "vertebra_L6_LSTV",
    28: "vertebra_T13",
    46: "rib_left_13",
    59: "rib_right_13",
    60: "rib_left_lumbar",
    61: "rib_right_lumbar",
    62: "hardware_unspecified",
    63: "hardware_cage",
    64: "hardware_screw_rod",
    65: "hardware_plate",
    66: "hardware_arthroplasty",
    67: "hardware_si_screw",
    68: "hardware_osteosynthesis",}

ALL_LABELS = {
    **{i:f"vertebra_C{i}" for i in range(1, 8)},#C1 to C7
    **{i:f"vertebra_T{i-7}" for i in range(8, 20)}, #T1 to T12
    **{i:f"vertebra_L{i-19}" for i in range(20, 25)},
    26:"sacrum", 29:"vertebra_S1", 30:"hip_left", 31:"hip_right"
}
# REGION: the structures every stage works on (segmentation prompts, meshing, evaluation, viewers).
# T12 down to the pelvis: T12, L1-L5, sacrum, S1, left hip, right hip. For the whole spine use
# REGION_IDS = list(ALL_LABELS).
REGION_IDS = [19, 20, 21, 22, 23, 24, 26, 29, 30, 31]
EXPORT_LABELS = {i: name for i, name in ALL_LABELS.items() if i in REGION_IDS}
NAME_TO_ID = {name: label_id for label_id, name in EXPORT_LABELS.items()} #Flips the LABELS dictionary

UNION_STRUCTURES = {"sacrum_with_S1": [26, 29]}
GROUPS = {
    "spine": list(range(1, 25)),
    "pelvis": [26, 29, 30, 31],
}

REGION_VERTEBRAE = [i for i in REGION_IDS if i <= 24]


EXPECTED_GENUS = {"cervical": 3, "thoracic": 1, "lumbar": 1, "hip": 1}

#Properties of the CT
MIN_SLICE_THICKNESS = 1
MAX_SLICE_THICKNESS = 1.5


#Naming and labelling conventions
Manifest_name = "selected_manifest.csv"
MANIFEST_COLUMNS = ["sample_index", "volume_id", "dataset", "ct_file", "label_file", "label_vb_file", "seen_in_training"]
ROWS_FILE = "quality_rows.json"
GROUND_TRUTH = "ground_truth"
GROUND_TRUTH_VB = "ground_truth_vb"
REFERENCES = (GROUND_TRUTH, GROUND_TRUTH_VB)
VERTEBRAL_BODY_METHODS = {"vbsegm2step", GROUND_TRUTH_VB}

Torch_Models = {
    "nvsegment": "NVIDIA NV-Segment-CTMR (VISTA3D architecture, MONAI bundle)",
    "nvsegment_ct": "NVIDIA NV-Segment-CT, the CT-only model (same repo and label IDs)",
    "totalsegmentator": "TotalSegmentator v2, 'total' task, spine + pelvis ROI subset",
    "vbsegm2step": "VBSegm2Step two-step nnU-Net v2 (vertebral bodies T1-L5 + sacrum only)",
}

def structure_class(name):
    """Coarse anatomical class used to group summary statistics."""
    if name.startswith("vertebra_C"):
        return "cervical"
    if name.startswith("vertebra_T"):
        return "thoracic"
    if name.startswith("vertebra_L"):
        return "lumbar"
    if name in ("sacrum", "vertebra_S1"):
        return "sacrum"
    return "hip"

def default_torch_python():
    candidates = [os.environ.get("SEG_TORCH_PYTHON", "")] + [
        str(Path.home() / base / "envs" / "seg_torch" / "bin" / "python")
        for env in ("seg_torch","spine-nav")
        for base in ("opt/anaconda3", "anaconda3", "miniconda3", "opt/miniconda3", "miniforge3")
    ]
    return next((c for c in candidates if c and Path(c).exists()), None)
