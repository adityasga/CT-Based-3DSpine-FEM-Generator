"""Visual checks: two methods' labels on the CT, and uncleaned vs cleaned geometry in 3D.

Neither label source is treated as the truth: the dataset's labels ("ground_truth" folder name)
are just another method, so comparisons report agreement, not accuracy.

  compare_labels(out, case, "ground_truth", "nvsegment")   slice viewer: both label maps + agreement map
  label_agreement(out, case, "ground_truth", "nvsegment")  per-structure agreement table
  cleaning_stages(out, method, case, bone)                 raw vs cleaned mask and mesh for one bone
  cleaning_summary(stages)                                 table: what cleaning changed, per method
  show_cleaning(stages)                                    rotatable 3D view with the changes highlighted
"""
from collections import Counter

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pyvista as pv
import trimesh
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from skimage.measure import marching_cubes

from .case_extractor import source_dir
from .labels_specs import EXPECTED_GENUS, EXPORT_LABELS, NAME_TO_ID, UNION_STRUCTURES, structure_class
from .meshing import (MeshParams, build_surface, clean_mask, mask_to_surface, orient_outward, pv_faces,
                      structure_boxes, topology_report)
from .segmentators import label_path
from .mesh_pipeline import load_labels, mesh_dir
from .mesh_quality import triangle_metrics

DISPLAY_NAME = {"ground_truth": "dataset labels"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]           # method A, method B, "different bone"
AGREE, BONE = "#c3c2b7", "#d8d2c4"
EXPORT_IDS = np.array(sorted(EXPORT_LABELS))
AXES = {"sagittal": 0, "coronal": 1, "axial": 2}     # axis held fixed, in RAS voxel order
LABEL_CMAP = ListedColormap(plt.cm.tab20(np.arange(32) % 20))
SEQUENTIAL = LinearSegmentedColormap.from_list("away", ["#e1e0d9", "#eb6834", "#8a3412"])


def display_name(method):
    return DISPLAY_NAME.get(method, method)


# ----------------------------------------------------------------------------------- labels on the CT
def load_case_volumes(out, case, methods):
    """CT and label maps reoriented to RAS (axis 0 = right->left, 1 = back->front, 2 = foot->head)."""
    ct_img = nib.as_closest_canonical(nib.load(str(source_dir(out) / case["ct_file"])))
    labels = {}
    for method in methods:
        lab = np.asanyarray(nib.as_closest_canonical(nib.load(str(label_path(out, method, case)))).dataobj)
        lab = np.rint(lab).astype(np.uint8)
        lab[~np.isin(lab, EXPORT_IDS)] = 0  # keep only meshed structures (drops ribs, femurs, ...)
        labels[method] = lab
    return ct_img.get_fdata(dtype=np.float32), labels, tuple(float(z) for z in ct_img.header.get_zooms()[:3])


def _plane(volume, axis, index):
    return np.take(volume, index, axis=axis).T  # rows = the later remaining axis, so head / front is up


def draw_label_slices(ct, lab_a, lab_b, zooms, plane, index, names):
    axis = AXES[plane]
    spacing = [z for i, z in enumerate(zooms) if i != axis]
    img, sa, sb = (_plane(v, axis, index) for v in (ct, lab_a, lab_b))
    agreement = np.zeros(sa.shape, np.uint8)
    agreement[(sa > 0) & (sa == sb)] = 1
    agreement[(sa > 0) & (sb == 0)] = 2
    agreement[(sa == 0) & (sb > 0)] = 3
    agreement[(sa > 0) & (sb > 0) & (sa != sb)] = 4
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))
    panels = [(sa, LABEL_CMAP, -0.5, 31.5, names[0]), (sb, LABEL_CMAP, -0.5, 31.5, names[1]),
              (agreement, ListedColormap([AGREE, SERIES[0], SERIES[1], SERIES[2]]), 0.5, 4.5, "agreement")]
    for ax, (overlay, cmap, lo, hi, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", origin="lower", vmin=-200, vmax=1200, aspect=spacing[1] / spacing[0])
        ax.imshow(np.ma.masked_equal(overlay, 0), cmap=cmap, vmin=lo, vmax=hi, alpha=0.5, origin="lower",
                  interpolation="nearest", aspect=spacing[1] / spacing[0])
        ax.set_title(f"{title} - {plane} {index}", fontsize=11)
        ax.axis("off")
    axes[2].legend(handles=[Patch(color=AGREE, label="both, same bone"), Patch(color=SERIES[0], label=f"only {names[0]}"),
                            Patch(color=SERIES[1], label=f"only {names[1]}"),
                            Patch(color=SERIES[2], label="both, different bone")],
                   loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.show()


def compare_labels(out, case, method_a="ground_truth", method_b="nvsegment"):
    """Slice viewer: pick a plane and scroll; method A, method B and their agreement side by side."""
    ct, labels, zooms = load_case_volumes(out, case, [method_a, method_b])
    a, b = labels[method_a], labels[method_b]
    names = (display_name(method_a), display_name(method_b))
    centre = [int(np.median(c)) for c in np.nonzero((a > 0) | (b > 0))]
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:  # no widgets: one static slice per plane
        for plane, axis in AXES.items():
            draw_label_slices(ct, a, b, zooms, plane, centre[axis], names)
        return
    plane = widgets.Dropdown(options=list(AXES), value="sagittal", description="plane")
    index = widgets.IntSlider(value=centre[0], min=0, max=ct.shape[0] - 1, description="slice",
                              continuous_update=False, layout=widgets.Layout(width="60%"))

    def on_plane(change):
        axis = AXES[change["new"]]
        index.max = ct.shape[axis] - 1
        index.value = centre[axis]

    plane.observe(on_plane, names="value")
    view = widgets.interactive_output(lambda plane, index: draw_label_slices(ct, a, b, zooms, plane, index, names),
                                      {"plane": plane, "index": index})
    display(widgets.HBox([plane, index]), view)


def label_agreement(out, case, method_a="ground_truth", method_b="nvsegment"):
    """Per structure: volume in each method, Dice overlap, and which label B gives A's voxels."""
    _, labels, zooms = load_case_volumes(out, case, [method_a, method_b])
    a, b = labels[method_a], labels[method_b]
    names = (display_name(method_a), display_name(method_b))
    voxel_ml = float(np.prod(zooms)) / 1000.0
    boxes_a, boxes_b = structure_boxes(a), structure_boxes(b)
    ids = sorted(set(boxes_a) | set(boxes_b))
    groups = [(EXPORT_LABELS[i], [i]) for i in ids] + [(n, g) for n, g in UNION_STRUCTURES.items() if set(g) & set(ids)]
    rows = []
    for name, group in groups:
        boxes = [bx[i] for bx in (boxes_a, boxes_b) for i in group if i in bx]
        box = tuple(slice(min(s[k].start for s in boxes), max(s[k].stop for s in boxes)) for k in range(3))
        in_a, in_b = np.isin(a[box], group), np.isin(b[box], group)
        n_a, n_b, n_both = int(in_a.sum()), int(in_b.sum()), int((in_a & in_b).sum())
        b_votes = Counter(b[box][in_a].tolist())
        b_votes.pop(0, None)
        top = b_votes.most_common(1)
        rows.append({
            "structure": name, f"{names[0]} (ml)": n_a * voxel_ml, f"{names[1]} (ml)": n_b * voxel_ml,
            "volume difference (% of mean)": 200.0 * (n_b - n_a) / (n_a + n_b) if n_a + n_b else np.nan,
            "agreement (Dice)": 2.0 * n_both / (n_a + n_b) if n_a + n_b else np.nan,
            f"{names[1]} label on {names[0]}'s voxels": EXPORT_LABELS.get(top[0][0], "-") if top else "-",
        })
    return pd.DataFrame(rows).round(3)


# ---------------------------------------------------------------------------- uncleaned vs cleaned 3D
def _raw_surface(mask, affine, offset):
    """Marching cubes at 0.5 with no clean-up: the geometry the original export script produced."""
    vertices, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5, method="lewiner")
    return orient_outward(nib.affines.apply_affine(affine, vertices + offset), faces.astype(np.int64))


def _signed_distance(points_from, vertices_to, faces_to):
    """Signed distance of each point to a closed surface (+ outside, - inside), in mm."""
    surface = pv.PolyData(vertices_to, pv_faces(faces_to))
    return np.asarray(pv.PolyData(points_from).compute_implicit_distance(surface)["implicit_distance"])


def cleaning_stages(out, method, case, bone, params=None):
    """Uncleaned vs cleaned mask and mesh of one bone, plus where they differ."""
    params = params or MeshParams()
    img, labels = load_labels(label_path(out, method, case))
    label_id = NAME_TO_ID[bone]
    boxes = structure_boxes(labels)
    if label_id not in boxes:
        raise ValueError(f"{bone} is not in the {method} labels for case {case['volume_id']}")
    box = boxes[label_id]
    expected = EXPECTED_GENUS.get(structure_class(bone))
    cleaned_mask, offset, voxel_log = clean_mask(labels, box, label_id, params, expected)
    pad = box[0].start - offset[0]
    raw_mask = np.pad(labels[box] == label_id, pad)

    raw_v, raw_f = _raw_surface(raw_mask, img.affine, offset)
    stl = mesh_dir(out, method, case) / "individual_stl" / f"{bone}.stl"
    if stl.exists():  # the mesh the pipeline actually exported
        tm = trimesh.load_mesh(str(stl), process=True)  # STL stores each triangle separately: merge shared vertices
        clean_v, clean_f = np.asarray(tm.vertices), np.asarray(tm.faces, dtype=np.int64)
    else:
        from .meshing import inplane_spacing
        target = params.target_edge_mm or inplane_spacing(img)
        surface = build_surface(labels, box, label_id, img.affine, target, params, expected)
        clean_v, clean_f = surface.vertices, surface.faces

    added, removed = cleaned_mask & ~raw_mask, raw_mask & ~cleaned_mask
    changes = {}
    for name, mask in (("added", added), ("removed", removed)):
        if mask.any():
            changes[name] = mask_to_surface(np.pad(mask, 1), img.affine, offset - 1)
    return {
        "method": method, "bone": bone, "voxel_log": voxel_log,
        "voxels_added": int(added.sum()), "voxels_removed": int(removed.sum()),
        "voxel_ml": abs(float(np.linalg.det(img.affine[:3, :3]))) / 1000.0,
        "raw": (raw_v, raw_f), "clean": (clean_v, clean_f), "changes": changes,
        # how far each uncleaned vertex is from the cleaned surface (big = cleaned away)
        "raw_to_clean_mm": np.abs(_signed_distance(raw_v, clean_v, clean_f)),
        # how far cleaning moved the surface: + outward (material added), - inward (removed)
        "clean_displacement_mm": _signed_distance(clean_v, raw_v, raw_f),
    }


def cleaning_summary(stages_list):
    """Uncleaned vs cleaned mesh, per method, as one table."""
    rows = {}
    for st in stages_list:
        for kind in ("raw", "clean"):
            v, f = st[kind]
            topo = topology_report(v, f)
            tri = triangle_metrics(v, f)
            tm = trimesh.Trimesh(v, f, process=False)
            rows[(display_name(st["method"]), "uncleaned" if kind == "raw" else "cleaned")] = {
                "shells": topo["shells"], "genus": topo["genus"], "boundary edges": topo["boundary_edges"],
                "non-manifold edges": topo["non_manifold_edges"], "self-intersecting faces": topo["self_intersecting_faces"],
                "watertight": bool(tm.is_watertight), "triangles": len(f), "volume (ml)": round(float(abs(tm.volume)) / 1000, 2),
                "min angle (deg)": round(float(tri["min_angle"].min()), 1), "max skewness": round(float(tri["skewness"].max()), 3),
                "voxels added / removed": f"{st['voxels_added']} / {st['voxels_removed']}" if kind == "clean" else "",
                "surface moved p95 / max (mm)": (f"{np.percentile(np.abs(st['clean_displacement_mm']), 95):.2f} / "
                                                 f"{np.abs(st['clean_displacement_mm']).max():.2f}") if kind == "clean" else "",
            }
    return pd.DataFrame(rows)


def _diverging(highlight_from_mm, clim_mm):
    """Blue (in) - grey - red (out), with a flat grey band for |movement| < highlight_from_mm."""
    half = min(0.49, highlight_from_mm / clim_mm / 2)
    return LinearSegmentedColormap.from_list(
        "in_out", [(0, "#2a78d6"), (0.5 - half, "#e1e0d9"), (0.5 + half, "#e1e0d9"), (1, "#e34948")])


def show_cleaning(stages_list, native_window=False, clim_mm=2.0, highlight_from_mm=0.5):
    """One row per method, three linked panels (rotate one, all follow):
      1. voxel changes: added (blue) / removed (orange) voxels inside the faint cleaned bone
      2. uncleaned mesh, coloured by distance to the cleaned mesh (orange = cleaned away)
      3. cleaned mesh, coloured by how far cleaning moved it (red = outward, blue = inward)
    Movements under highlight_from_mm stay grey: smoothing shifts the whole surface by ~0.2-0.4 mm when it
    removes the voxel staircase, and colouring that would hide the larger, local changes.
    native_window=True opens a separate rotatable window; otherwise the current Jupyter backend is used
    (rotatable in the notebook with pv.set_jupyter_backend("trame") after `pip install "pyvista[jupyter]"`).
    """
    rows = len(stages_list)
    plotter = pv.Plotter(shape=(rows, 3), window_size=(1500, 460 * rows), notebook=False if native_window else None)
    for r, st in enumerate(stages_list):
        name = display_name(st["method"])
        clean = pv.PolyData(st["clean"][0], pv_faces(st["clean"][1]))
        raw = pv.PolyData(st["raw"][0], pv_faces(st["raw"][1]))

        plotter.subplot(r, 0)
        plotter.add_text(f"{name}: voxel changes ({st['bone']})\nblue added {st['voxels_added']:,} | "
                         f"orange removed {st['voxels_removed']:,}", font_size=8)
        plotter.add_mesh(clean, color=BONE, opacity=0.25)
        for kind, color in (("added", SERIES[0]), ("removed", SERIES[1])):
            if kind in st["changes"]:
                v, f = st["changes"][kind]
                plotter.add_mesh(pv.PolyData(v, pv_faces(f)), color=color)

        plotter.subplot(r, 1)
        plotter.add_text(f"{name}: uncleaned mesh\ncolour = mm to cleaned mesh", font_size=8)
        plotter.add_mesh(raw, scalars=st["raw_to_clean_mm"], cmap=SEQUENTIAL, clim=[highlight_from_mm, clim_mm],
                         scalar_bar_args={"title": f"{name} mm away\n", "n_labels": 3, "fmt": "%.1f"})

        plotter.subplot(r, 2)
        plotter.add_text(f"{name}: cleaned mesh\ncolour = mm moved (+out / -in)", font_size=8)
        plotter.add_mesh(clean, scalars=st["clean_displacement_mm"], cmap=_diverging(highlight_from_mm, clim_mm),
                         clim=[-clim_mm, clim_mm],
                         scalar_bar_args={"title": f"{name} mm moved\n", "n_labels": 3, "fmt": "%.1f"})
    plotter.link_views()
    plotter.show()
    return plotter
