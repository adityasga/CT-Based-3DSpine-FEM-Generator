"""Voxel-level screening and label-map helpers shared by the VerSe and TotalSegmentator selections.

Also fetches the vertebral-body reference labels (VertebralBodiesCT-Labels, the training labels of vbsegm2step),
which exist for most VerSe scans and TotalSegmentator v2 scans. They are the fair reference for a method
that segments vertebral bodies only.
"""
import csv

import nibabel as nib
import nibabel.processing
import numpy as np
from huggingface_hub import hf_hub_download
from scipy import ndimage

METAL_HU = 2800          # cortical bone stays below ~2000 HU; implants saturate at 3071 or beyond
METAL_MAX_ML = 0.1       # more metal than this within METAL_MARGIN_MM of the bones rejects the scan
METAL_MARGIN_MM = 5.0

VB_REPO = "fhofmann/VertebralBodiesCT-Labels"
# VertebralBodiesCT / vbsegm2step label -> CTSpinoPelvic1K id: T1-T12, L1-L5, L6, sacrum, T13 (coccyx 20 has no id)
VB_TO_SPINO = {**{i: 7 + i for i in range(1, 13)}, **{12 + i: 19 + i for i in range(1, 6)}, 18: 25, 19: 26, 21: 28}
VB_MIN_CONTAINMENT = 0.8


def save_label_map(labels, ref_img, path):
    """uint8 label map on the grid (affine, qform/sform codes) of ref_img."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(labels.astype(np.uint8), ref_img.affine)
    out.set_qform(ref_img.affine, code=int(ref_img.header["qform_code"]) or 1)
    out.set_sform(ref_img.affine, code=int(ref_img.header["sform_code"]) or 1)
    out.set_data_dtype(np.uint8)
    nib.save(out, str(path))


def on_grid(img, ref_img):
    return img.shape[:3] == ref_img.shape[:3] and np.allclose(img.affine, ref_img.affine, atol=1e-3)


def labels_on_grid(img, ref_img):
    """Integer label array of img on ref_img's grid (nearest-neighbour resampling in world space if needed)."""
    if not on_grid(img, ref_img):
        img = nibabel.processing.resample_from_to(img, (ref_img.shape[:3], ref_img.affine), order=0)
    return np.rint(np.asanyarray(img.dataobj)).astype(np.uint8)


def bounding_box(mask, margin=(0, 0, 0)):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + 1 + np.asarray(margin), mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def metal_ml(ct_img, bone_mask, hu=METAL_HU, margin_mm=METAL_MARGIN_MM):
    """Volume (ml) of voxels >= hu within margin_mm of bone_mask: screws, rods, cages, arthroplasties.

    Only the neighbourhood of the bones counts, so vascular clips or dense contrast elsewhere do not reject a scan.
    """
    zooms = np.asarray(ct_img.header.get_zooms()[:3], dtype=float)
    box = bounding_box(bone_mask, np.ceil(margin_mm / zooms).astype(int))
    near = ndimage.distance_transform_edt(~bone_mask[box], sampling=zooms) <= margin_mm
    ct = np.asanyarray(ct_img.dataobj[box])
    return float(np.count_nonzero((ct >= hu) & near) * zooms.prod() / 1000.0)


def caudal_margin_mm(img, mask):
    """How far (mm, along the patient's head-feet axis) the scan extends below the lowest voxel of mask."""
    z = img.affine[2, :3] @ np.argwhere(mask).T + img.affine[2, 3]
    corners = np.array(np.meshgrid(*[[0, n - 1] for n in img.shape[:3]], indexing="ij")).reshape(3, -1)
    return float(z.min() - (img.affine[2, :3] @ corners + img.affine[2, 3]).min())


def vertebral_body_index(src):
    """{case name: data.csv row} of VertebralBodiesCT-Labels; `set` is training/testing for vbsegm2step."""
    path = hf_hub_download(VB_REPO, "data.csv", repo_type="dataset", local_dir=str(src / "vertebral_bodies_ct"))
    with open(path, newline="", encoding="utf-8") as f:
        return {row["case"].removesuffix(".nii.gz"): row for row in csv.DictReader(f)}


def fetch_vertebral_bodies(src, name, row, ct_img, gt, dest):
    """Save the vertebral-body labels of `name` (CTSpinoPelvic1K ids, CT grid) to dest; returns a status string.

    Each body must lie inside the ground-truth vertebra of the same level, so a level-numbering disagreement
    between the two label sets is caught instead of silently scoring the wrong vertebra.
    """
    folder = "test" if row["set"] == "testing" else "train"
    path = hf_hub_download(VB_REPO, f"{folder}/{name}.nii.gz", repo_type="dataset",
                           local_dir=str(src / "vertebral_bodies_ct"))
    vb = labels_on_grid(nib.load(path), ct_img)
    labels = np.zeros_like(vb)
    for vb_id, spino_id in VB_TO_SPINO.items():
        labels[vb == vb_id] = spino_id
    for label_id in np.unique(labels):
        if label_id == 0 or label_id == 26 or not (gt == label_id).any():
            continue
        body = labels == label_id
        inside = np.count_nonzero(body & (gt == label_id)) / np.count_nonzero(body)
        if inside < VB_MIN_CONTAINMENT:
            return f"not_used_level_mismatch_label_{label_id}_{inside:.2f}"
    save_label_map(labels, ct_img, dest)
    return f"saved_{row['set']}"
