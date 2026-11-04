"""TotalSegmentator dataset v3 case selection (https://zenodo.org/records/22688904, CC BY 4.0).

1939 CTs with one mask per structure. Used here: vertebrae C1-L6, sacrum and both hips (v3 has no S1 and no
implant label). The 37 GB zip is never downloaded whole: its directory is read once (cached), then only the
candidates' masks and CTs are fetched with HTTP range requests. A local copy of the zip works the same way.

'Normal' rules: adult (age >= min_age), no pathology located in the bones (meta.csv), every region bone present
and not cut by the scan edge (T12-L5, sacrum, both hips), no L6, no metal near the bones. T13 has no label in
this dataset, so it cannot be screened.

Unseen first: the 711 scans added in v3 (Sept 2026) postdate NV-Segment's and vbsegm2step's training, so they
are drawn first (recent TotalSegmentator model releases may include them). Of the v2 scans, the training split
trained TotalSegmentator, and VertebralBodiesCT-Labels' training split trained vbsegm2step; these are recorded in
the manifest's seen_in_training column. Cite Wasserthal et al., Radiology: AI 2023.
"""
import csv
import random

import nibabel as nib
import numpy as np




from .screening import (METAL_HU, METAL_MAX_ML, bounding_box, fetch_vertebral_bodies, labels_on_grid, metal_ml,
                        on_grid, save_label_map, vertebral_body_index)
from .sources import RemoteZip

DATASET = "totalseg_v3"
ZIP_URL = "https://zenodo.org/records/22688904/files/Totalsegmentator_dataset_v300.zip?download=1"
V2_ZIP_URL = "https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip?download=1"
ROOT = "Totalsegmentator_dataset_v300/"
# CTSpinoPelvic1K id -> TotalSegmentator mask name
MASKS = {**{i: f"vertebrae_C{i}" for i in range(1, 8)}, **{7 + i: f"vertebrae_T{i}" for i in range(1, 13)},
         **{19 + i: f"vertebrae_L{i}" for i in range(1, 6)}, 25: "vertebrae_L6", 26: "sacrum", 30: "hip_left", 31: "hip_right"}
L6 = 25
FILES = {"ct.nii.gz", *(f"{name}.nii.gz" for name in MASKS.values())}


def open_archive(src, zip_source=ZIP_URL):
    return RemoteZip(zip_source, src / "totalseg_v3_index.json",
                     keep=lambda name: name.endswith("meta.csv") or name.rsplit("/", 1)[-1] in FILES)


def v2_ids(src):
    """Image ids of the v2 release (read from the v2 zip's directory once, then cached)."""
    archive = RemoteZip(V2_ZIP_URL, src / "totalseg_v2_index.json", keep=lambda name: name.endswith("/ct.nii.gz"))
    return {name.split("/")[-2] for name in archive.members}


def read_meta(src, archive):
    path = src / "totalseg_v3_meta.csv"
    if not path.exists():
        path.write_bytes(archive.read(ROOT + "meta.csv"))
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["image_id"]: row for row in csv.DictReader(f, delimiter=";")}


def _screen(src, archive, case_id, meta, required, vb_index, min_age, info):
    age = meta["age"]
    if not age or float(age) < min_age:
        return f"rejected_age_{age or 'unknown'}"
    if "bones" in meta["pathology_location"]:
        return f"rejected_bone_pathology_{meta['pathology']}"

    def load_mask(label_id):
        name = f"{ROOT}{case_id}/segmentations/{MASKS[label_id]}.nii.gz"
        img = nib.load(str(archive.extract(name, src / "totalseg_raw" / case_id / f"{MASKS[label_id]}.nii.gz")))
        return img, np.asanyarray(img.dataobj) > 0

    cut = []
    from .labels_specs import ALL_LABELS
    for label_id in required:
        m = load_mask(label_id)[1]
        if not m.any():
            return f"rejected_missing_{ALL_LABELS[label_id]}"
        if any(s.start == 0 or s.stop == n for s, n in zip(bounding_box(m), m.shape)):
            cut.append(ALL_LABELS[label_id])
    if load_mask(L6)[1].any():
        return "rejected_vertebra_L6_LSTV"
    if cut:
        return "rejected_cut_by_scan_edge_" + ";".join(cut)

    ct_img = nib.load(str(archive.extract(f"{ROOT}{case_id}/ct.nii.gz", src / "ct" / f"{case_id}.nii.gz")))
    gt = np.zeros(ct_img.shape[:3], dtype=np.uint8)
    for label_id in MASKS:
        if label_id == L6:
            continue
        img, m = load_mask(label_id)
        if not on_grid(img, ct_img):
            m = labels_on_grid(img, ct_img) > 0
        gt[m] = label_id
    metal = info["metal_ml"] = round(metal_ml(ct_img, np.isin(gt, required)), 3)
    if metal > METAL_MAX_ML:
        return f"rejected_metal_{metal:.2f}ml"
    save_label_map(gt, ct_img, src / "labels" / f"{case_id}_label.nii.gz")

    vb_row = vb_index.get(f"tseg-{case_id}")
    try:
        info["vertebral_body_labels"] = (fetch_vertebral_bodies(src, f"tseg-{case_id}", vb_row, ct_img, gt,
                                                                src / "labels_vb" / f"{case_id}_label.nii.gz")
                                         if vb_row else "none_published")
    except Exception as exc:
        info["vertebral_body_labels"] = f"error: {type(exc).__name__}: {exc}"
    return "accepted"


def select_cases(out, n, seed, zip_source=ZIP_URL, min_age=18.0):
    from .find_directory import source_dir
    from .labels_specs import REGION_IDS
    src = source_dir(out); src.mkdir(parents=True, exist_ok=True)
    archive = open_archive(src, zip_source)
    meta = read_meta(src, archive)
    released_in_v2 = v2_ids(src)
    vb_index = vertebral_body_index(src)
    required = [i for i in REGION_IDS if i in MASKS]

    def seen_by(case_id):
        if case_id not in released_in_v2:
            return []
        models = ["totalsegmentator"] if meta[case_id]["split"] == "train" else []
        return models + (["vbsegm2step"] if vb_index.get(f"tseg-{case_id}", {}).get("set") == "training" else [])

    candidates = sorted(meta)
    random.Random(seed).shuffle(candidates)
    candidates.sort(key=lambda case_id: case_id in released_in_v2)  # stable: v3 additions first

    selected, audit = [], []
    for case_id in candidates:
        if len(selected) == n:
            break
        row = meta[case_id]
        info = {"volume_id": case_id, "new_in_v3": case_id not in released_in_v2, "age": row["age"],
                "study_type": row["study_type"], "pathology": row["pathology"],
                "pathology_location": row["pathology_location"], "split": row["split"]}
        try:
            decision = _screen(src, archive, case_id, row, required, vb_index, min_age, info)
        except Exception as exc:
            decision = f"rejected_error_{type(exc).__name__}: {exc}"
        audit.append({**info, "decision": decision})
        if decision != "accepted":
            continue
        has_vb = str(info.get("vertebral_body_labels", "")).startswith("saved")
        selected.append({
            "volume_id": case_id, "dataset": DATASET, "ct_file": f"ct/{case_id}.nii.gz",
            "label_file": f"labels/{case_id}_label.nii.gz",
            "label_vb_file": f"labels_vb/{case_id}_label.nii.gz" if has_vb else "",
            "seen_in_training": ";".join(seen_by(case_id)),
        })
        print(f"[{len(selected)}/{n}] {case_id} accepted (new in v3: {info['new_in_v3']})", flush=True)
    from .case_extractor import write_selection
    write_selection(out, selected, audit, {
        "dataset": DATASET, "source": "https://zenodo.org/records/22688904 (TotalSegmentator dataset v3.0.0)",
        "license": "CC BY 4.0", "sample_size_requested": n, "sample_size_selected": len(selected), "seed": seed,
        "rules": [f"age >= {min_age}", "no pathology located in the bones",
                  "all region bones present and not cut by the scan edge", "no L6 (T13 has no label)",
                  f"<= {METAL_MAX_ML} ml of voxels >= {METAL_HU} HU within 5 mm of the region bones",
                  "scans added in v3 drawn first"],
        "ground_truth": "TotalSegmentator v3 masks merged into CTSpinoPelvic1K ids (no S1 label)",
        "vertebral_body_reference": "huggingface.co/datasets/fhofmann/VertebralBodiesCT-Labels (v2 scans only)",
    })
    if len(selected) < n:
        print(f"WARNING: only {len(selected)} TotalSegmentator scans passed the rules; requested {n}.")
    return selected
