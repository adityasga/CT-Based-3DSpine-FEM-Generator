from pathlib import Path
from .find_directory import source_dir
from .data_fetcher import fetch
from .labels_specs import excluded_list, MIN_SLICE_THICKNESS, MAX_SLICE_THICKNESS,REPO, Manifest_name, DEFAULT_DATASET
from .report import write_selection
import numpy as np
import nibabel as nib
import json
import random
import csv

def manifest_dataset(out):
    """Dataset of an existing selection (None if nothing was selected yet)."""
    if not (Path(out) / Manifest_name).exists():
        return None
    cases = load_cases(out)
    return cases[0]["dataset"] if cases else None

def is_false(value):
    return str(value).lower().strip() in {"false", "0", "no", "n"}

def eligible_metadata(record):
    #reject any non-fused, transitional, L6 or hardware record
    return(
        str(record.get("config", "")).strip().lower() == "fused"
        and str(record.get("lstv_class", "")).strip() in {"0", "0.0"}
        and is_false(record.get("lstv_confusion_zone", False))
        and is_false(record.get("has_l6", False))
        and is_false(record.get("hardware_labelled", False))
        and str(record.get("position")).strip().lower() in {"prone", "supine"}
        and MIN_SLICE_THICKNESS <= (record.get("slice_thickness")) <= MAX_SLICE_THICKNESS
        and is_false(record.get("has_lumbar_rib",False))
        and record.get("ct_file") and record.get("label_file")
    )
def voxel_exclusion(label_path):
    labels = np.asanyarray(nib.load(str(label_path)).dataobj)#nib used to load neuroimaging (NIFTI) files, dataobj returns an array proxy object (memory-saving feature)
    ids = set(np.unique(labels).astype(int).tolist())
    return {label_id: excluded_list[label_id] for label_id in excluded_list if label_id in ids}

def load_cases(out):
    path = Path(out)/Manifest_name
    if not path.exists():
        raise SystemExit(f"{path}not found - run 'python run_pipeline.py select --out{out}' first.")
    with path.open(newline="",encoding="utf-8") as f:
        return [
            {"sample_index": int(r["sample_index"]), "volume_id": r["volume_id"],
             "dataset": r.get("dataset") or DEFAULT_DATASET, "ct_file": r["ct_file"], "label_file": r["label_file"],
             "label_vb_file": r.get("label_vb_file") or "", "seen_in_training": r.get("seen_in_training") or ""}
            for r in csv.DictReader(f)
        ]
def select_cases(out,n,seed,revision):
    #screen out candidates based on labels and write the selection manifest.csv
    out = Path(out)
    src = source_dir(out)
    manifest_file = fetch("manifest.json", src, revision)
    records = json.loads(manifest_file.read_text(encoding="utf-8"))
    candidates = [r for r in records if eligible_metadata(r)]
    random.Random(seed).shuffle(candidates)

    selected, audit = [], []
    # Inspect labels before CT Download
    for record in candidates:
        volume_id = str(record.get("volume_id", record.get("token", "unknown")))
        try:
            label_path = fetch(record['label_file'], src, revision)
            excluded = voxel_exclusion(label_path)
            decision = "accepted" if not excluded else "rejected_voxels_" + ";".join(excluded.values())
        except Exception as e:
            label_path, excluded = None, {}
            decision = f"rejected_label_check_error_{type(e).__name__}: {e}"
        audit.append({
            "volume_id": volume_id, "ct_file": record.get("ct_file", ""), "label_file": record.get("label_file", ""),
            "lstv_class": record.get("lstv_class", ""), "castellvi_type": record.get("castellvi_type", ""),
            "hardware_labelled": record.get("hardware_labelled", ""),
            "forbidden_ids_found": ";".join(map(str, excluded)),
            "forbidden_structures_found": ";".join(excluded.values()), "decision": decision,
        })
        if decision == "accepted":
            selected.append((volume_id, record))
        if len(selected) == n:
            break
    if len(selected) < n:
        with (out / "eligibility_audit.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(audit[0]) if audit else [])
            writer.writeheader(); writer.writerows(audit)
        raise SystemExit(f"Only {len(selected)} candidates passed all exclusions; requested {n}.")

    rows = [{"volume_id": volume_id, "dataset": DEFAULT_DATASET, "ct_file": record["ct_file"],
             "label_file": record["label_file"]} for volume_id, record in selected]
    write_selection(out, rows, audit, {
        "dataset": DEFAULT_DATASET, "repo": REPO, "revision": revision, "sample_size": n, "seed": seed,
        "required_config": "fused",
        "metadata_exclusions": ["lstv_class != 0", "has_l6", "lstv_confusion_zone", "castellvi_type", "lstv_phenotype", "hardware_labelled"],
        "voxel_exclusions": excluded_list,
    })
    return load_cases(out)