"""VerSe 2020 case selection (https://osf.io/t98fz, CC BY-SA 4.0): whole-vertebra labels, no sacrum or hips.

Data: the challenge-structure copy (OSF component b2wxj), one folder per CT series holding the CT and its
vertebra mask. VerSe label ids are the CTSpinoPelvic1K ids (1-7 C1-C7, 8-19 T1-T12, 20-24 L1-L5, 25 L6,
28 T13), so the mask is the ground-truth label map unchanged. Sacrum and hips have no ground truth here.

'Normal' rules, as for CTSpinoPelvic1K: every region vertebra present and not cut by the scan edge, no L6 / T13,
no fracture of a region vertebra (Genant grade 0 in VerSe's fracture table; ungraded scans are rejected),
at least `min_caudal_mm` of scan below L5 so the top of the sacrum is in view, and no metal near the bones.
VerSe publishes no ages, so age is not screened.

Unseen first: scans in vbsegm2step's held-out test split (or not in its data at all) are drawn before its
training scans. NV-Segment-CTMR was trained partly on VerSe, so its accuracy here may be optimistic.
Cite the three VerSe papers listed in the OSF readme when using the data.
"""
import csv
import json
import random

import nibabel as nib
import numpy as np
from .screening import (METAL_HU, METAL_MAX_ML, bounding_box, caudal_margin_mm, fetch_vertebral_bodies,
                        labels_on_grid, metal_ml, save_label_map, vertebral_body_index)
from .sources import download, osf_list

DATASET = "verse2020"
OSF_API = "https://api.osf.io/v2/nodes/{}/files/osfstorage/"
SERIES_NODE = "b2wxj"     # VerSe 2020, challenge data structure (one folder per CT series)
FRACTURE_NODE = "4skx2"   # VerSe 2020, subject structure; holds fracture_data/Genant_Fractures_thoracolumbar_reference.csv
INDEX_NAME = "verse2020_index.json"


def series_index(src):
    """{volume_id: {"split", "folder"}} for every CT series (cached after the first run)."""
    path = src / INDEX_NAME
    if path.exists():
        return json.loads(path.read_text())
    index = {}
    for split, entry in osf_list(OSF_API.format(SERIES_NODE)).items():
        if entry["kind"] == "folder":
            for name, series in osf_list(entry["folder"]).items():
                index[name] = {"split": split, "folder": series["folder"]}
    path.write_text(json.dumps(index, indent=1))
    return index


def series_files(src, index, volume_id):
    """Download URLs of the CT and the vertebra mask of one series (the folder listing is cached in the index)."""
    entry = index[volume_id]
    if "files" not in entry:
        entry["files"] = {name: f["download"] for name, f in osf_list(entry["folder"]).items() if f["kind"] == "file"}
        (src / INDEX_NAME).write_text(json.dumps(index, indent=1))
    niftis = [name for name in entry["files"] if name.endswith(".nii.gz")]
    masks = [name for name in niftis if "_seg" in name or "_msk" in name]
    cts = [name for name in niftis if name not in masks]
    if len(masks) != 1 or len(cts) != 1:
        raise RuntimeError(f"expected one CT and one mask, found {niftis}")
    return entry["files"][cts[0]], entry["files"][masks[0]]


def fracture_grades(src):
    """{volume_id: {level: Genant grade}} for the graded levels of each scan."""
    path = src / "verse2020_genant_fractures.csv"
    if not path.exists():
        folder = osf_list(OSF_API.format(FRACTURE_NODE))["fracture_data"]["folder"]
        download(osf_list(folder)["Genant_Fractures_thoracolumbar_reference.csv"]["download"], path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {f"verse{row['ID']}": {level: int(float(grade)) for level, grade in row.items() if level != "ID" and grade.strip()}
                for row in csv.DictReader(f, delimiter=";")}


def vb_name(volume_id):
    return "verse-" + volume_id.removeprefix("verse")


def _screen(src, volume_id, index, grades, vb_index, min_caudal_mm, info):
    from .labels_specs import ALL_LABELS, excluded_list, REGION_VERTEBRAE
    graded = grades.get(volume_id)
    if graded is None:
        return "rejected_no_fracture_grading"
    levels = [ALL_LABELS[i].removeprefix("vertebra_") for i in REGION_VERTEBRAE]
    fractured = [f"{level}_grade{graded[level]}" for level in levels if graded.get(level, 0) > 0]
    if fractured:
        return "rejected_fracture_" + ";".join(fractured)

    ct_url, mask_url = series_files(src, index, volume_id)
    mask_img = nib.load(download(mask_url, src / "verse_raw" / f"{volume_id}_seg.nii.gz"))
    mask = np.rint(np.asanyarray(mask_img.dataobj)).astype(np.uint8)
    present = set(np.unique(mask).tolist()) - {0}
    info["labels_present"] = ";".join(ALL_LABELS.get(i, excluded_list.get(i, str(i))) for i in sorted(present))
    forbidden = [excluded_list[i] for i in excluded_list if i in present]
    if forbidden:
        return "rejected_" + ";".join(forbidden)
    missing = [ALL_LABELS[i] for i in REGION_VERTEBRAE if i not in present]
    if missing:
        return "rejected_missing_" + ";".join(missing)
    cut = [ALL_LABELS[i] for i in REGION_VERTEBRAE
           if any(s.start == 0 or s.stop == n for s, n in zip(bounding_box(mask == i), mask.shape))]
    if cut:
        return "rejected_cut_by_scan_edge_" + ";".join(cut)
    lowest = max(REGION_VERTEBRAE)
    margin = info["mm_below_lowest_vertebra"] = round(caudal_margin_mm(mask_img, mask == lowest), 1)
    if margin < min_caudal_mm:
        return f"rejected_only_{margin:.0f}mm_below_{ALL_LABELS[lowest]}"

    ct_img = nib.load(download(ct_url, src / "ct" / f"{volume_id}.nii.gz"))
    gt = labels_on_grid(mask_img, ct_img)
    metal = info["metal_ml"] = round(metal_ml(ct_img, np.isin(gt, REGION_VERTEBRAE)), 3)
    if metal > METAL_MAX_ML:
        return f"rejected_metal_{metal:.2f}ml"
    save_label_map(gt, ct_img, src / "labels" / f"{volume_id}_label.nii.gz")

    vb_row = vb_index.get(vb_name(volume_id))
    try:
        info["vertebral_body_labels"] = (fetch_vertebral_bodies(src, vb_name(volume_id), vb_row, ct_img, gt,
                                                                src / "labels_vb" / f"{volume_id}_label.nii.gz")
                                         if vb_row else "none_published")
    except Exception as exc:
        info["vertebral_body_labels"] = f"error: {type(exc).__name__}: {exc}"
    return "accepted"


def select_cases(out, n, seed, min_caudal_mm=20.0):
    from .find_directory import source_dir
    src = source_dir(out); src.mkdir(parents=True, exist_ok=True)
    index = series_index(src)
    grades = fracture_grades(src)
    vb_index = vertebral_body_index(src)

    def trained_vbseg(volume_id):
        return vb_index.get(vb_name(volume_id), {}).get("set") == "training"

    candidates = sorted(index)
    random.Random(seed).shuffle(candidates)
    candidates.sort(key=trained_vbseg)  # stable: unseen by vbsegm2step first, random order within each group

    selected, audit = [], []
    for volume_id in candidates:
        if len(selected) == n:
            break
        info = {"volume_id": volume_id, "split": index[volume_id]["split"],
                "in_vbsegm2step_training": trained_vbseg(volume_id)}
        try:
            decision = _screen(src, volume_id, index, grades, vb_index, min_caudal_mm, info)
        except Exception as exc:
            decision = f"rejected_error_{type(exc).__name__}: {exc}"
        audit.append({**info, "decision": decision})
        if decision != "accepted":
            continue
        has_vb = str(info.get("vertebral_body_labels", "")).startswith("saved")
        selected.append({
            "volume_id": volume_id, "dataset": DATASET, "ct_file": f"ct/{volume_id}.nii.gz",
            "label_file": f"labels/{volume_id}_label.nii.gz",
            "label_vb_file": f"labels_vb/{volume_id}_label.nii.gz" if has_vb else "",
            "seen_in_training": ";".join(["nvsegment"] + (["vbsegm2step"] if trained_vbseg(volume_id) else [])),
        })
        print(f"[{len(selected)}/{n}] {volume_id} accepted (vertebral-body labels: {info.get('vertebral_body_labels')})", flush=True)
    from .case_extractor import write_selection
    write_selection(out, selected, audit, {
        "dataset": DATASET, "source": "https://osf.io/t98fz (VerSe 2020; OSF component b2wxj)", "license": "CC BY-SA 4.0",
        "sample_size_requested": n, "sample_size_selected": len(selected), "seed": seed,
        "rules": ["all region vertebrae present and not cut by the scan edge", "no L6 / T13",
                  "Genant grade 0 for every region vertebra (ungraded scans rejected)",
                  f">= {min_caudal_mm} mm of scan below the lowest region vertebra",
                  f"<= {METAL_MAX_ML} ml of voxels >= {METAL_HU} HU within 5 mm of the region vertebrae",
                  "scans unseen by vbsegm2step drawn first"],
        "ground_truth": "VerSe vertebra masks (no sacrum or hip labels)",
        "vertebral_body_reference": "huggingface.co/datasets/fhofmann/VertebralBodiesCT-Labels (CC BY-SA 4.0)",
    })
    if len(selected) < n:
        print(f"WARNING: only {len(selected)} of {len(candidates)} VerSe series passed the rules; requested {n}.")
    return selected
