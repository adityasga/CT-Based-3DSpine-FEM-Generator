"""Per-element mesh quality for surface triangles and volume tetrahedra.

Metric definitions (ideal element -> value):
  skewness          equilateral-volume skewness (ANSYS/Fluent): 1 - size / size of the equilateral
                    element with the same circumradius (area for triangles, volume for tets). 0 best, 1 degenerate.
  aspect_ratio      Verdict/VTK aspect ratio: triangle  l_max * (l0+l1+l2) / (4*sqrt(3)*A);
                    tet  l_max * (sum of face areas) / (6*sqrt(6)*V). 1 best.
  scaled_jacobian   Verdict/VTK scaled Jacobian (corner Jacobian normalised by its edge lengths). 1 best,
                    <= 0 inverted. Surface triangles are signed against the local surface normal, so a
                    folded-over triangle is negative.
  size_change       max ratio of element size (mean edge length) to that of any face/edge neighbour. 1 best.
Also reported: equiangle skewness and min/max angle for triangles.
"""
from dataclasses import dataclass

import numpy as np
import pyvista as pv
import trimesh

from .meshing import pv_faces

SQRT3 = np.sqrt(3.0)


@dataclass
class Thresholds:
    tri_max_skewness: float = 0.85
    tet_max_skewness: float = 0.95
    tri_max_aspect_ratio: float = 5.0
    tet_max_aspect_ratio: float = 10.0
    max_size_change: float = 2.0
    min_scaled_jacobian: float = 0.2
    max_fidelity_hd95_mm: float = 0.5

    def limits(self, kind):
        """metric -> (comparison, limit) for 'tri' or 'tet'."""
        return {
            "skewness": ("max", getattr(self, f"{kind}_max_skewness")),
            "aspect_ratio": ("max", getattr(self, f"{kind}_max_aspect_ratio")),
            "size_change": ("max", self.max_size_change),
            "scaled_jacobian": ("min", self.min_scaled_jacobian),
        }


def _neighbour_ratio(sizes, pairs):
    ratio = np.ones(len(sizes))
    if len(pairs):
        a, b = sizes[pairs[:, 0]], sizes[pairs[:, 1]]
        r = np.maximum(a, b) / np.maximum(np.minimum(a, b), 1e-300)
        np.maximum.at(ratio, pairs[:, 0], r)
        np.maximum.at(ratio, pairs[:, 1], r)
    return ratio


def triangle_metrics(vertices, faces):
    p = [vertices[faces[:, i]] for i in range(3)]
    edges = np.stack([np.linalg.norm(p[(i + 1) % 3] - p[i], axis=1) for i in range(3)], axis=1)  # e0=|p1-p0|...
    normal = np.cross(p[1] - p[0], p[2] - p[0])
    area = 0.5 * np.linalg.norm(normal, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        angles = []
        for i in range(3):
            u, w = p[(i + 1) % 3] - p[i], p[(i + 2) % 3] - p[i]
            cos = np.einsum("ij,ij->i", u, w) / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1))
            angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        angles = np.nan_to_num(np.stack(angles, axis=1), nan=0.0)
        circumradius = edges.prod(axis=1) / (4 * area)
        skewness = np.clip(1 - area / (3 * SQRT3 / 4 * circumradius ** 2), 0, 1)
        aspect = edges.max(axis=1) * edges.sum(axis=1) / (4 * SQRT3 * area)
        # Verdict: min over corners of 2A / (|e_i||e_j|), normalised by sin(60 deg)
        corner = np.stack([edges[:, 0] * edges[:, 2], edges[:, 0] * edges[:, 1], edges[:, 1] * edges[:, 2]], axis=1)
        scaled_jac = (2 * area / corner.max(axis=1)) * 2 / SQRT3
    skewness = np.nan_to_num(skewness, nan=1.0)
    aspect = np.nan_to_num(aspect, nan=np.inf, posinf=np.inf)
    scaled_jac = np.nan_to_num(scaled_jac, nan=0.0)

    # sign against the area-weighted vertex normals: folded/flipped triangles become negative
    vertex_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], normal)
    reference = vertex_normals[faces].sum(axis=1)
    scaled_jac = np.where(np.einsum("ij,ij->i", reference, normal) > 0, scaled_jac, -np.abs(scaled_jac))

    mesh = trimesh.Trimesh(vertices, faces, process=False)
    size_change = _neighbour_ratio(edges.mean(axis=1), np.asarray(mesh.face_adjacency))
    equiangle = np.maximum((angles.max(axis=1) - 60) / 120, (60 - angles.min(axis=1)) / 60)
    return {
        "skewness": skewness, "aspect_ratio": aspect, "scaled_jacobian": scaled_jac, "size_change": size_change,
        "equiangle_skewness": equiangle, "min_angle": angles.min(axis=1), "max_angle": angles.max(axis=1),
        "edge_length": edges.ravel(), "area": area,
    }


def tet_metrics(nodes, tets):
    P = nodes[tets]
    s0, s1, s2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 1], P[:, 0] - P[:, 2]
    s3, s4, s5 = P[:, 3] - P[:, 0], P[:, 3] - P[:, 1], P[:, 3] - P[:, 2]
    jac = np.einsum("ij,ij->i", s3, np.cross(s2, s0))  # = 6 * signed volume
    L2 = [np.einsum("ij,ij->i", s, s) for s in (s0, s1, s2, s3, s4, s5)]
    corner = np.sqrt(np.stack([L2[0] * L2[2] * L2[3], L2[0] * L2[1] * L2[4],
                               L2[1] * L2[2] * L2[5], L2[3] * L2[4] * L2[5]]))
    lengths = np.sqrt(np.stack(L2, axis=1))
    a, b, c = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]
    face_area = 0.5 * (np.linalg.norm(np.cross(a, b), axis=1) + np.linalg.norm(np.cross(a, c), axis=1)
                       + np.linalg.norm(np.cross(b, c), axis=1)
                       + np.linalg.norm(np.cross(P[:, 2] - P[:, 1], P[:, 3] - P[:, 1]), axis=1))
    volume = jac / 6.0
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled_jac = np.clip(np.sqrt(2) * jac / corner.max(axis=0), -1, 1)
        aspect = lengths.max(axis=1) * face_area / (6 * np.sqrt(6) * np.abs(volume))
        center = (L2[0][:, None] * np.cross(b, c) + np.einsum("ij,ij->i", b, b)[:, None] * np.cross(c, a)
                  + np.einsum("ij,ij->i", c, c)[:, None] * np.cross(a, b)) / (2 * jac[:, None])
        circumradius = np.linalg.norm(center, axis=1)
        skewness = np.clip(1 - np.abs(volume) / (8 * SQRT3 / 27 * circumradius ** 3), 0, 1)
    skewness = np.nan_to_num(skewness, nan=1.0)
    aspect = np.nan_to_num(aspect, nan=np.inf, posinf=np.inf)
    scaled_jac = np.nan_to_num(scaled_jac, nan=0.0)
    size_change = _neighbour_ratio(lengths.mean(axis=1), _tet_face_pairs(tets, len(nodes)))
    return {"skewness": skewness, "aspect_ratio": aspect, "scaled_jacobian": scaled_jac,
            "size_change": size_change, "volume": volume}


def _tet_face_pairs(tets, n_nodes):
    faces = np.sort(np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), axis=1)
    owner = np.tile(np.arange(len(tets)), 4)
    faces = faces.astype(np.int64)
    if n_nodes < 2 ** 21:
        keys = (faces[:, 0] << 42) | (faces[:, 1] << 21) | faces[:, 2]
    else:
        keys = np.unique(faces, axis=0, return_inverse=True)[1].ravel()
    order = np.argsort(keys, kind="stable")
    same = keys[order][1:] == keys[order][:-1]
    return np.stack([owner[order][:-1][same], owner[order][1:][same]], axis=1)


def tetrahedralize(vertices, faces, optimizer="fast"):
    """Gmsh HXT volume mesh bounded exactly by the given surface triangulation.

    optimizer: 'fast'     HXT + Gmsh default tet optimisation            (seconds)
               'relocate' + Relocate3D vertex smoothing                  (~3x slower)
               'netgen'   + Netgen optimiser: removes slivers            (~25x slower)
    """
    from gmsh import model, finalize, initialize, option
    initialize(interruptible=False)
    try:
        option.setNumber("General.Verbosity", 1)
        # Single-threaded on purpose: gmsh's OpenMP runtime segfaults once VTK/PyMeshLab (which bundle
        # their own) have run in the same process. Parallelism comes from the case-level worker pool.
        option.setNumber("General.NumThreads", 1)
        model.add("bone")
        surface = model.addDiscreteEntity(2)
        model.mesh.addNodes(2, surface, np.arange(1, len(vertices) + 1), np.asarray(vertices, float).ravel())
        model.mesh.addElementsByType(surface, 2, [], (np.asarray(faces) + 1).ravel())
        model.geo.addVolume([model.geo.addSurfaceLoop([surface])])
        model.geo.synchronize()
        option.setNumber("Mesh.Algorithm3D", 10)  # HXT
        option.setNumber("Mesh.Optimize", 1)
        model.mesh.generate(3)
        if optimizer == "relocate":
            model.mesh.optimize("Relocate3D")
        elif optimizer == "netgen":
            model.mesh.optimize("Netgen")
        _, _, element_nodes = model.mesh.getElements(3)
        tags, coords, _ = model.mesh.getNodes()
        index = np.zeros(int(tags.max()) + 1, dtype=np.int64)
        index[tags.astype(np.int64)] = np.arange(len(tags))
        tets = index[np.asarray(element_nodes[0], dtype=np.int64).reshape(-1, 4)]
        return coords.reshape(-1, 3), tets
    finally:
        finalize()


def summarize(prefix, metrics, limits):
    """min / mean / p5 / p95 / max per metric, plus how many elements break the threshold."""
    row = {}
    for name, values in metrics.items():
        if name in ("edge_length", "area", "volume"):
            continue
        finite = values[np.isfinite(values)] if np.isfinite(values).any() else np.array([np.inf])
        row.update({
            f"{prefix}_{name}_min": float(finite.min()), f"{prefix}_{name}_mean": float(finite.mean()),
            f"{prefix}_{name}_p5": float(np.percentile(finite, 5)), f"{prefix}_{name}_p95": float(np.percentile(finite, 95)),
            f"{prefix}_{name}_max": float(values.max()),
        })
        if name in limits:
            kind, limit = limits[name]
            bad = values > limit if kind == "max" else values < limit
            row[f"{prefix}_{name}_n_fail"] = int(bad.sum())
            row[f"{prefix}_{name}_pct_fail"] = float(100.0 * bad.mean())
    return row


def surface_distances(vertices_a, faces_a, vertices_b, faces_b):
    """Unsigned point-to-surface distances a->b and b->a (evaluated at the vertices)."""
    a = pv.PolyData(vertices_a, pv_faces(faces_a))
    b = pv.PolyData(vertices_b, pv_faces(faces_b))
    d_ab = np.abs(np.asarray(a.compute_implicit_distance(b)["implicit_distance"]))
    d_ba = np.abs(np.asarray(b.compute_implicit_distance(a)["implicit_distance"]))
    return d_ab, d_ba


def distance_summary(prefix, d_ab, d_ba):
    both = np.concatenate([d_ab, d_ba])
    return {
        f"{prefix}_hd95_mm": float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95))),
        f"{prefix}_assd_mm": float(both.mean()),
        f"{prefix}_hdmax_mm": float(both.max()),
    }


def save_element_metrics(path, vertices, cells, metrics, cell_type):
    """Write a VTK file with every element's metrics as cell data (open in ParaView/Slicer)."""
    if cell_type == "tri":
        grid = pv.PolyData(vertices, pv_faces(cells))
    else:
        grid = pv.UnstructuredGrid(np.hstack([np.full((len(cells), 1), 4), cells]).ravel(),
                                   np.full(len(cells), pv.CellType.TETRA, dtype=np.uint8), vertices)
    for name, values in metrics.items():
        if len(values) == len(cells):
            grid.cell_data[name] = values
    grid.save(str(path))

