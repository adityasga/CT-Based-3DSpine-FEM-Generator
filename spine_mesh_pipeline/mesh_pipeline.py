"""Mesh + quality stage and accuracy stage, one (method, case) job per worker process."""
import json
import hashlib
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import trimesh

from .find_directory import case_dir_name, mesh_dir
from .labels_specs import EXPECTED_GENUS, EXPORT_LABELS, GROUPS, UNION_STRUCTURES, structure_class, ROWS_FILE, GROUND_TRUTH, VERTEBRAL_BODY_METHODS, GROUND_TRUTH_VB, REFERENCES
from .mesh_quality import (distance_summary, save_element_metrics, summarize, surface_distances,
                           tet_metrics, tetrahedralize, triangle_metrics)
from .meshing import FragmentTooSmall, build_surface, inplane_spacing, structure_boxes, topology_report
from .segmentators import label_path, references_for

def fingerprint(labels):
    """Short hash of a label map's voxels: two methods with the same hash have identical segmentations."""
    return hashlib.md5(np.ascontiguousarray(labels).tobytes()).hexdigest()[:12]

def load_labels(path):
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if not np.issubdtype(data.dtype, np.integer):
        data = np.rint(data)
    return img, data.astype(np.uint8, copy=False)


def expected_genus(name, method):
    """Anatomical genus used for topology correction; a vertebral body (no arch, no foramen) is a solid, genus 0."""
    if method in VERTEBRAL_BODY_METHODS and name.startswith("vertebra_"):
        return 0
    return EXPECTED_GENUS.get(structure_class(name))


def mesh_structure(labels, box, label_id, name, img, target_edge, params, thresholds, tet_mode, save_dir, save_elements,
                   genus=None):
    row = {}
    t0 = time.time()
    surface = build_surface(labels, box, label_id, img.affine, target_edge, params, genus)
    v, f = surface.vertices, surface.faces
    row.update(surface.voxel_log)
    row["repair_steps"] = ";".join(surface.repair_steps)
    topo = topology_report(v, f)
    row.update({f"surf_{k}": val for k, val in topo.items()})
    tm = trimesh.Trimesh(v, f, process=False)
    row.update({
        "surf_watertight": bool(tm.is_watertight), "surf_winding_consistent": bool(tm.is_winding_consistent),
        "surf_outward_normals": bool(tm.volume > 0), "surf_volume_ml": float(tm.volume / 1000.0),
        "surf_area_mm2": float(tm.area), "surf_vertices": len(v), "surf_triangles": len(f),
    })
    tri = triangle_metrics(v, f)
    row.update({"surf_edge_mean_mm": float(tri["edge_length"].mean()), "surf_edge_min_mm": float(tri["edge_length"].min()),
                "surf_edge_max_mm": float(tri["edge_length"].max())})
    row.update(summarize("tri", tri, thresholds.limits("tri")))
    row["tri_inverted"] = int((tri["scaled_jacobian"] <= 0).sum())
    row["tri_min_angle_deg"] = float(tri["min_angle"].min())
    row.update(distance_summary("fidelity", *surface_distances(v, f, surface.reference_vertices, surface.reference_faces)))
    row["surface_seconds"] = round(time.time() - t0, 1)
    tm.export(str(save_dir / f"{name}.stl"))
    if save_elements:
        save_element_metrics(save_dir / "element_quality" / f"{name}_surface.vtp", v, f, tri, "tri")

    if tet_mode != "none":
        t0 = time.time()
        try:
            nodes, tets = tetrahedralize(v, f, tet_mode)
            tet = tet_metrics(nodes, tets)
            row.update({"tet_status": "ok", "tet_nodes": len(nodes), "tet_elements": len(tets),
                        "tet_inverted": int((tet["scaled_jacobian"] <= 0).sum()),
                        "tet_volume_ml": float(tet["volume"].sum() / 1000.0)})
            row.update(summarize("tet", tet, thresholds.limits("tet")))
            if save_elements:
                save_element_metrics(save_dir / "element_quality" / f"{name}_tet.vtu", nodes, tets, tet, "tet")
        except Exception as exc:
            row["tet_status"] = f"failed: {type(exc).__name__}: {exc}"
        row["tet_seconds"] = round(time.time() - t0, 1)
    return row, tm


def process_case(out, method, case, params, thresholds, tet_mode, save_elements, overwrite):
    """Mesh every exported structure of one case; results cached in quality_rows.json."""
    case_out = mesh_dir(out, method, case)
    rows_file = case_out / ROWS_FILE
    if rows_file.exists() and not overwrite:
        return json.loads(rows_file.read_text())
    src = label_path(out, method, case)
    if not src.exists():
        if method == GROUND_TRUTH_VB:  # no vertebral-body reference published for this scan
            return []
        return [{"method": method, "volume_id": case["volume_id"], "status": f"missing label map {src}"}]
    individual = case_out / "individual_stl"
    individual.mkdir(parents=True, exist_ok=True)
    for stale in list(individual.glob("*.stl")) + list(individual.glob("element_quality/*")) + list(case_out.glob("*_combined.stl")):
        stale.unlink()
    if save_elements:
        (individual / "element_quality").mkdir(exist_ok=True)
    img, labels = load_labels(src)
    target_edge = params.target_edge_mm or inplane_spacing(img)
    boxes = structure_boxes(labels)
    base = {"method": method, "volume_id": case["volume_id"], "case": case_dir_name(case),
            "voxel_spacing_mm": "x".join(f"{z:.3f}" for z in img.header.get_zooms()[:3]),
            "target_edge_mm": round(float(target_edge), 4),
            "label_file": str(src), "label_fingerprint": fingerprint(labels)}
    rows, groups = [], {group: [] for group in GROUPS}
    for label_id, name in EXPORT_LABELS.items():
        if label_id not in boxes:
            continue
        row = {**base, "structure": name, "structure_class": structure_class(name), "label_id": label_id}
        try:
            metrics, tm = mesh_structure(labels, boxes[label_id], label_id, name, img, target_edge, params,
                                         thresholds, tet_mode, individual, save_elements, expected_genus(name, method))
            row.update(metrics)
            row["status"] = "completed"
            for group, ids in GROUPS.items():
                if label_id in ids:
                    groups[group].append(tm)
        except FragmentTooSmall as exc:
            row["status"] = f"skipped_fragment: {exc}"
        except Exception as exc:
            row["status"] = f"failed: {type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc(limit=3)
        rows.append(row)
    for group, meshes in groups.items():
        if meshes:
            trimesh.util.concatenate(meshes).export(str(case_out / f"{group}_combined.stl"))
    rows_file.write_text(json.dumps(rows, indent=1))
    return rows


def case_accuracy(out, method, case, reference=GROUND_TRUTH):
    """Voxel Dice / precision / recall and mesh surface distances of one method's case against a reference.

    Precision (share of the prediction inside the reference structure) is what to read for a vertebral-body
    method against whole-vertebra ground truth: high means the body sits in the right vertebra, while its
    Dice and recall are low by construction.
    """
    gt_img, gt = load_labels(label_path(out, reference, case))
    pred_img, pred = load_labels(label_path(out, method, case))
    if gt.shape != pred.shape or not np.allclose(gt_img.affine, pred_img.affine, atol=1e-3):
        raise RuntimeError(f"prediction and {reference} are on different grids; re-import the label map")
    if np.array_equal(gt, pred):
        raise RuntimeError(f"{method} label map is identical to {reference} ({label_path(out, method, case)}); "
                           "the segmentation step did not produce its own output")
    voxel_ml = float(np.prod(gt_img.header.get_zooms()[:3])) / 1000.0
    gt_boxes, pred_boxes = structure_boxes(gt), structure_boxes(pred)
    structures = [(name, [label_id]) for label_id, name in EXPORT_LABELS.items()] + list(UNION_STRUCTURES.items())
    gt_mesh_dir = mesh_dir(out, reference, case) / "individual_stl"
    pred_mesh_dir = mesh_dir(out, method, case) / "individual_stl"
    rows = []
    for name, ids in structures:
        in_gt = any(i in gt_boxes for i in ids)
        in_pred = any(i in pred_boxes for i in ids)
        if not (in_gt or in_pred):
            continue
        boxes = [b[i] for b in (gt_boxes, pred_boxes) for i in ids if i in b]
        box = tuple(slice(min(b[k].start for b in boxes), max(b[k].stop for b in boxes)) for k in range(3))
        g, p = np.isin(gt[box], ids), np.isin(pred[box], ids)
        n_g, n_p, n_both = int(g.sum()), int(p.sum()), int(np.logical_and(g, p).sum())
        row = {"method": method, "reference": reference, "volume_id": case["volume_id"], "case": case_dir_name(case),
               "structure": name, "structure_class": structure_class(name) if len(ids) == 1 else "sacrum",
               "detection": "both" if in_gt and in_pred else ("missed" if in_gt else "not_in_ground_truth"),
               "volume_gt_ml": n_g * voxel_ml, "volume_pred_ml": n_p * voxel_ml,
               "dice": 2.0 * n_both / (n_g + n_p) if n_g + n_p else np.nan,
               "precision": n_both / n_p if n_p else np.nan, "recall": n_both / n_g if n_g else np.nan}
        row["volume_diff_pct"] = 100.0 * (n_p - n_g) / n_g if n_g else np.nan
        gt_stl, pred_stl = gt_mesh_dir / f"{name}.stl", pred_mesh_dir / f"{name}.stl"
        if len(ids) == 1 and gt_stl.exists() and pred_stl.exists():
            a, b = trimesh.load_mesh(str(pred_stl), process=False), trimesh.load_mesh(str(gt_stl), process=False)
            row.update(distance_summary("surface", *surface_distances(a.vertices, a.faces, b.vertices, b.faces)))
        rows.append(row)
    return rows


def _mesh_job(args):
    return process_case(*args)


def _accuracy_job(args):
    out, method, case, reference = args
    try:
        return case_accuracy(out, method, case, reference)
    except Exception as exc:
        return [{"method": method, "reference": reference, "volume_id": case["volume_id"], "case": case_dir_name(case),
                 "structure": "", "detection": f"failed: {type(exc).__name__}: {exc}"}]


def run_jobs(function, jobs, workers, describe):
    results = []
    if workers <= 1:
        for index, job in enumerate(jobs, 1):
            print(f"[{index}/{len(jobs)}] {describe(job)}", flush=True)
            results.extend(function(job))
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(function, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                results.extend(future.result())
                print(f"[{index}/{len(jobs)}] done {describe(job)}", flush=True)
            except Exception as exc:
                print(f"[{index}/{len(jobs)}] FAILED {describe(job)}: {type(exc).__name__}: {exc}", flush=True)
    return results


def mesh_all(out, methods, cases, params, thresholds, tet_mode, workers, save_elements, overwrite):
    jobs = [(out, m, c, params, thresholds, tet_mode, save_elements, overwrite) for c in cases for m in methods]
    (Path(out) / "meshes").mkdir(parents=True, exist_ok=True)
    (Path(out) / "meshes" / "mesh_params.json").write_text(json.dumps(
        {"mesh": asdict(params), "thresholds": asdict(thresholds), "tet_optimizer": tet_mode}, indent=2))
    return run_jobs(_mesh_job, jobs, workers, lambda j: f"{j[1]} / {case_dir_name(j[2])}")


def accuracy_all(out, methods, cases, workers):
    """Score every method against each reference it has (references_for) that is also in `methods`."""
    jobs = [(out, m, c, ref) for c in cases for m in methods if m not in REFERENCES
            for ref in references_for(m) if ref in methods
            and label_path(out, m, c).exists() and label_path(out, ref, c).exists()]
    return run_jobs(_accuracy_job, jobs, workers, lambda j: f"accuracy {j[1]} vs {j[3]} / {case_dir_name(j[2])}")


def check_inputs(out, methods, cases):
    """For every case and method: which label file is used, whether it duplicates another method's,
    and whether its meshes were built from that same label map (stale meshes give misleading results)."""
    rows = []
    for case in cases:
        seen = {}
        for method in methods:
            path = label_path(out, method, case)
            row = {"case": case_dir_name(case), "method": method, "label_file": str(path), "problem": ""}
            if not path.exists():
                if method == GROUND_TRUTH_VB:  # published for VerSe and TotalSegmentator v2 scans only
                    row["meshes"] = "no vertebral-body reference for this scan"
                else:
                    row["problem"] = "label map missing"
                rows.append(row)
                continue
            _, labels = load_labels(path)
            fp = row["label_fingerprint"] = fingerprint(labels)
            if fp in seen:
                row["problem"] = f"identical to {seen[fp]}"
            seen.setdefault(fp, method)
            rows_file = mesh_dir(out, method, case) / ROWS_FILE
            if not rows_file.exists():
                row["meshes"] = "not meshed"
            else:
                used = {r.get("label_fingerprint") for r in json.loads(rows_file.read_text())}
                row["meshes"] = ("up to date" if used == {fp} else "built before fingerprints were recorded - re-mesh to check"
                                 if used == {None} else "STALE: built from a different label map - re-mesh with overwrite")
                if row["meshes"].startswith("STALE"):
                    row["problem"] = (row["problem"] + "; " if row["problem"] else "") + "stale meshes"
            rows.append(row)
    return pd.DataFrame(rows)
