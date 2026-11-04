"""Label map -> clean, watertight, isotropically remeshed surface (one closed body per bone).

Per structure:
  1. voxel clean-up    closing (seal cracks / pinholes), optional opening (spurs), keep the
                       largest face-connected component (islands), fill enclosed cavities;
                       topology-guided closing removes tunnels beyond the bone's anatomical genus
  2. surface           marching cubes (Lewiner) in world coordinates, outward normals
  3. smoothing         3D Slicer closed-surface smoothing: vtkWindowedSincPolyDataFilter,
                       20 iterations, pass band 10^(-4 * smoothing_factor)
  4. repair            MeshFix (holes, non-manifold edges/vertices, self-intersections)
  5. remesh            isotropic explicit remeshing to a target edge length, with a cap on
                       how far the remesher may move the surface
  6. final repair      MeshFix again only if the remeshed surface is not clean
"""
from dataclasses import dataclass

import nibabel as nib
import numpy as np
import pymeshfix
import pymeshlab
import pyvista as pv
import vtk
from scipy import ndimage
from skimage.measure import euler_number, marching_cubes


@dataclass
class MeshParams:
    smoothing_factor: float = 0.5        # Slicer "Smoothing factor" (0 = none, 1 = very strong)
    closing_voxels: int = 1              # morphological closing radius (voxels)
    max_topology_closing_voxels: int = 3 # grow closing up to this radius to remove spurious tunnels (handles)
    opening_voxels: int = 0              # morphological opening radius; off by default (erodes thin laminae)
    fill_cavities: bool = True           # no internal voids / inner shells
    keep_largest: bool = True            # one body per bone
    target_edge_mm: float | None = None  # None -> mean in-plane voxel spacing of the CT
    remesh_iterations: int = 10
    max_remesh_deviation: float = 0.25   # remesher may move the surface <= this fraction of the target edge
    min_structure_ml: float = 0.5        # smaller labels are stray fragments (e.g. a mislabelled blob), not bones


# Just below 0.5: on a binary mask every ambiguous marching-cubes face has its saddle exactly at 0.5,
# and resolving those ties arbitrarily punches dozens of voxel-scale tunnels through thin bone
# (sacrum genus 42 -> 13). 0.49 resolves them consistently and moves the surface by 0.01 voxel.
ISO_LEVEL = 0.49


class FragmentTooSmall(ValueError):
    pass


@dataclass
class Surface:
    vertices: np.ndarray
    faces: np.ndarray
    reference_vertices: np.ndarray       # unsmoothed marching-cubes surface of the cleaned mask
    reference_faces: np.ndarray
    voxel_log: dict
    repair_steps: list


def pv_faces(faces):
    return np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).ravel()


def signed_volume(vertices, faces):
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def orient_outward(vertices, faces):
    if signed_volume(vertices, faces) < 0:
        faces = faces[:, ::-1].copy()
    return vertices, faces


def inplane_spacing(img):
    """Mean spacing of the two voxel axes that are not the cranio-caudal (slice) axis."""
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    slice_axis = int(np.argmax(np.abs(img.affine[2, :3]) / np.linalg.norm(img.affine[:3, :3], axis=0)))
    return float(np.mean(np.delete(zooms, slice_axis)))


def structure_boxes(labels):
    return {i + 1: box for i, box in enumerate(ndimage.find_objects(labels)) if box is not None}


def _voxel_clean(crop, pad, closing, params):
    face_conn = ndimage.generate_binary_structure(3, 1)
    mask = np.pad(crop, pad)
    if closing:
        mask = ndimage.binary_closing(mask, face_conn, iterations=closing)
    if params.opening_voxels:
        mask = ndimage.binary_opening(mask, face_conn, iterations=params.opening_voxels)
    components, n_components = ndimage.label(mask, structure=face_conn)
    log = {"components_found": int(n_components), "voxels_removed_islands": 0, "voxels_filled_cavities": 0}
    if params.keep_largest and n_components > 1:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        before = int(mask.sum())
        mask = components == int(np.argmax(sizes))
        log["voxels_removed_islands"] = before - int(mask.sum())
    if params.fill_cavities:
        before = int(mask.sum())
        mask = ndimage.binary_fill_holes(mask)
        log["voxels_filled_cavities"] = int(mask.sum()) - before
    return mask, log


def voxel_genus(mask):
    """Handles of a single-component, cavity-free mask (26-connectivity = marching cubes at ISO_LEVEL)."""
    return int(1 - euler_number(mask, connectivity=3))


def _topology_guided_clean(crop, pad, params, expected_genus):
    """Clean at increasing closing radii until the expected genus is reached or capped."""
    closing = params.closing_voxels
    mask, log = _voxel_clean(crop, pad, closing, params)
    initial_genus = genus = voxel_genus(mask)

    while (expected_genus is not None
           and genus > expected_genus
           and closing < params.max_topology_closing_voxels):
        closing += 1
        mask, log = _voxel_clean(crop, pad, closing, params)
        genus = voxel_genus(mask)

    return mask, log, initial_genus, closing


def clean_mask(labels, box, label_id, params, expected_genus=None):
    """Voxel clean-up. When the bone's anatomical genus is known, the closing radius is grown
    (up to max_topology_closing_voxels) until spurious tunnels through the bone are sealed."""
    pad = max(params.max_topology_closing_voxels, params.closing_voxels, params.opening_voxels) + 2
    crop = labels[box] == label_id
    offset = np.array([s.start for s in box]) - pad
    mask, log, initial_genus, closing = _topology_guided_clean(
        crop, pad, params, expected_genus
    )
    log.update({
        "voxel_genus_initial": initial_genus,
        "voxels_raw": int(crop.sum()), "voxels_clean": int(mask.sum()),
        "truncated_by_fov": any(s.start == 0 or s.stop == n for s, n in zip(box, labels.shape)),
        "closing_voxels_used": closing, "genus_expected": expected_genus,
    })
    return mask, offset, log


def mask_to_surface(mask, affine, offset):
    voxel_vertices, faces, _, _ = marching_cubes(
        mask.astype(np.float32), level=ISO_LEVEL, method="lewiner"
    )
    world_vertices = nib.affines.apply_affine(affine, voxel_vertices + offset)
    triangle_faces = faces.astype(np.int64)
    return orient_outward(world_vertices, triangle_faces)


def slicer_smooth(vertices, faces, smoothing_factor):
    """Same filter and parameter mapping as Slicer's labelmap -> closed-surface conversion."""
    if smoothing_factor <= 0:
        return vertices
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(pv.PolyData(vertices, pv_faces(faces)))
    smoother.SetNumberOfIterations(20)
    smoother.SetPassBand(10.0 ** (-4.0 * smoothing_factor))
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return np.asarray(pv.wrap(smoother.GetOutput()).points, dtype=float)


def topology_report(vertices, faces):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertices, faces))
    topo = ms.get_topological_measures()
    ms.compute_selection_by_self_intersections_per_face()
    return {
        "boundary_edges": int(topo["boundary_edges"]),
        "non_manifold_edges": int(topo["non_two_manifold_edges"]),
        "non_manifold_vertices": int(topo["non_two_manifold_vertices"]),
        "shells": int(topo["connected_components_number"]),
        "genus": int(topo["genus"]) if topo["genus"] is not None else -1,
        "holes": int(topo["number_holes"]),
        "self_intersecting_faces": int(ms.current_mesh().selected_face_number()),
    }


def is_clean(report):
    return (report["boundary_edges"] == 0 and report["non_manifold_edges"] == 0
            and report["non_manifold_vertices"] == 0 and report["shells"] == 1
            and report["self_intersecting_faces"] == 0)


def meshfix(vertices, faces):
    fixer = pymeshfix.MeshFix(np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int32))
    fixer.repair(joincomp=False, remove_smallest_components=True)
    return np.asarray(fixer.points, dtype=float), np.asarray(fixer.faces, dtype=np.int64)


def isotropic_remesh(vertices, faces, target_edge, iterations, max_deviation):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertices, faces))
    ms.meshing_isotropic_explicit_remeshing(
        iterations=iterations, targetlen=pymeshlab.PureValue(target_edge),
        featuredeg=180,  # bone is smooth: no crease edges to preserve
        checksurfdist=True, maxsurfdist=pymeshlab.PureValue(max_deviation),
    )
    mesh = ms.current_mesh()
    return np.asarray(mesh.vertex_matrix(), dtype=float), np.asarray(mesh.face_matrix(), dtype=np.int64)


def build_surface(labels, box, label_id, affine, target_edge, params, expected_genus=None):
    mask, offset, voxel_log = clean_mask(labels, box, label_id, params, expected_genus)
    volume_ml = voxel_log["voxels_clean"] * abs(np.linalg.det(affine[:3, :3])) / 1000.0
    if volume_ml < params.min_structure_ml:
        raise FragmentTooSmall(f"{volume_ml:.3f} ml < {params.min_structure_ml} ml")
    ref_v, ref_f = mask_to_surface(mask, affine, offset)
    vertices, faces = slicer_smooth(ref_v, ref_f, params.smoothing_factor), ref_f
    steps = []
    if not is_clean(topology_report(vertices, faces)):
        vertices, faces = meshfix(vertices, faces)
        steps.append("meshfix_after_smoothing")
    vertices, faces = isotropic_remesh(vertices, faces, target_edge, params.remesh_iterations,
                                       params.max_remesh_deviation * target_edge)
    if not is_clean(topology_report(vertices, faces)):
        vertices, faces = meshfix(vertices, faces)
        steps.append("meshfix_after_remesh")
    vertices, faces = orient_outward(vertices, faces)
    return Surface(vertices, faces, ref_v, ref_f, voxel_log, steps)
