"""Biphasic (two-material) bone mesh: cortical shell + trabecular core, as one conforming tetrahedral mesh.

Why a fixed-thickness shell: the CT voxels here are ~0.8 mm, but the cortex of a vertebral body is only
~0.2-0.5 mm thick (thicker in the posterior elements), so it is not resolved well enough to threshold from
the CT intensity. The usual finite-element choice is used instead: the cortex is the layer of bone within
`cortical_thickness_mm` of the outer surface, and everything deeper is trabecular. Bone thinner than twice
the thickness (thin laminae, process tips) is cortex all the way through.

Steps, one function each:
  1. load_outer_surface   the cleaned, watertight STL from the mesh stage
  2. depth_map            fill the surface into a fine voxel grid; distance transform = depth below the surface
  3. core_surfaces        isosurface at depth = thickness (the outer surface moved inward), split into pieces
  4. tidy_core            remesh + repair each piece like the outer surface; measure the cortex thickness
  5. biphasic_tets        Gmsh: cortical = between outer and core surfaces, trabecular = inside the core,
                          sharing nodes on the interface; tagged as physical groups "cortical" / "trabecular"
  6. mesh_biphasic        runs 1-5, checks quality per region and CT intensity per region, saves the files
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pyvista as pv
import trimesh
from scipy import ndimage
from skimage.measure import marching_cubes

from .find_directory import case_dir_name, source_dir, mesh_dir
from .labels_specs import EXPORT_LABELS
from .meshing import is_clean, isotropic_remesh, meshfix, orient_outward, topology_report
from .mesh_pipeline import run_jobs
from .mesh_quality import Thresholds, summarize, tet_metrics

CORTICAL, TRABECULAR = 0, 1
REGION_NAMES = {CORTICAL: "cortical", TRABECULAR: "trabecular"}


@dataclass
class BiphasicParams:
    cortical_thickness_mm: float = 1.0   # shell thickness; see module docstring
    grid_spacing_mm: float = 0.3         # voxel size of the depth map (0.3 mm: ~8 s per vertebra, 0.1% volume error)
    min_core_ml: float = 0.05            # smaller trabecular pieces are left as cortex
    tet_optimizer: str = "fast"          # "fast" | "relocate" | "netgen", as in the mesh stage
    target_edge_mm: float | None = None  # core surface edge length; None = same as the outer surface


def default_bones():
    """Vertebrae in the current region. S1 is excluded: it is a planar cut of the sacrum, so a shell on its cut face would be wrong."""
    return [n for n in EXPORT_LABELS.values() if n.startswith("vertebra_") and n != "vertebra_S1"]


# ---------------------------------------------------------------------------------------------- step 1
def load_outer_surface(out, method, case, bone):
    stl = mesh_dir(out, method, case) / "individual_stl" / f"{bone}.stl"
    if not stl.exists():
        raise FileNotFoundError(f"{stl} not found - run the mesh stage for {method} first")
    tm = trimesh.load_mesh(str(stl), process=True)  # STL stores each triangle separately: merge shared vertices
    return np.asarray(tm.vertices, dtype=float), np.asarray(tm.faces, dtype=np.int64)


# ---------------------------------------------------------------------------------------------- step 2
def depth_map(vertices, faces, spacing):
    """Depth below the surface (mm) for every voxel of a fine grid inside the bone; 0 outside.

    Returns (depth[x, y, z], origin) where origin is the world position of voxel [0, 0, 0].
    """
    surface = pv.PolyData(vertices, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
    image = surface.voxelize_binary_mask(spacing=spacing)                  # 1 inside the surface, 0 outside
    nx, ny, nz = image.dimensions
    inside = np.asarray(image.point_data["mask"]).reshape(nz, ny, nx).transpose(2, 1, 0) > 0  # VTK is x-fastest
    inside = np.pad(inside, 2)                                             # guarantee an outside border
    origin = np.asarray(image.origin) - 2 * spacing
    # distance from each inside voxel centre to the nearest outside voxel centre; the surface itself lies
    # half a voxel beyond the last inside centre, so depth below the surface = distance - spacing / 2
    depth = ndimage.distance_transform_edt(inside, sampling=spacing) - spacing / 2
    return np.where(inside, depth, 0.0), origin


# ---------------------------------------------------------------------------------------------- step 3
def core_surfaces(depth, origin, spacing, thickness, min_core_ml):
    """Isosurface depth = thickness, split into separate pieces; tiny pieces are dropped (stay cortical)."""
    if depth.max() <= thickness:
        return []                                                          # bone thinner than 2 x thickness everywhere
    vertices, faces, _, _ = marching_cubes(depth, level=thickness, spacing=(spacing,) * 3)
    core = trimesh.Trimesh(vertices + origin, faces)
    pieces = []
    for piece in core.split(only_watertight=False):
        v, f = orient_outward(np.asarray(piece.vertices), np.asarray(piece.faces, dtype=np.int64))
        if abs(piece.volume) / 1000.0 >= min_core_ml:
            pieces.append((v, f))
    return pieces


# ---------------------------------------------------------------------------------------------- step 4
def tidy_core(vertices, faces, target_edge, outer_vertices, outer_faces):
    """Remesh and repair one core piece, then measure the cortex: distance from the core to the outer surface."""
    vertices, faces = isotropic_remesh(vertices, faces, target_edge, 10, 0.25 * target_edge)
    if not is_clean(topology_report(vertices, faces)):
        vertices, faces = meshfix(vertices, faces)
    vertices, faces = orient_outward(vertices, faces)
    outer = pv.PolyData(outer_vertices, np.hstack([np.full((len(outer_faces), 1), 3), outer_faces]).ravel())
    signed = np.asarray(pv.PolyData(vertices).compute_implicit_distance(outer)["implicit_distance"])
    if (signed >= 0).any():
        raise RuntimeError("trabecular core pokes through the outer surface; use a smaller grid spacing or thickness")
    return vertices, faces, -signed                                        # local cortical thickness at each core vertex


# ---------------------------------------------------------------------------------------------- step 5
def biphasic_tets(outer, cores, optimizer="fast", save_prefix=None):
    """One conforming tet mesh: region CORTICAL between outer and cores, region TRABECULAR inside the cores.

    The core surfaces belong to both regions, so the two materials share nodes along the interface.
    save_prefix: write <prefix>.msh and <prefix>.inp (Abaqus; element sets CORTICAL / TRABECULAR).
    """
    import gmsh
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.option.setNumber("General.NumThreads", 1)  # see quality.tetrahedralize: OpenMP clash with VTK
        gmsh.model.add("biphasic")
        loops, next_node = [], 1
        for vertices, faces in [outer, *cores]:
            surface = gmsh.model.addDiscreteEntity(2)
            tags = np.arange(next_node, next_node + len(vertices))
            gmsh.model.mesh.addNodes(2, surface, tags, np.asarray(vertices, float).ravel())
            gmsh.model.mesh.addElementsByType(surface, 2, [], (np.asarray(faces) + next_node).ravel())
            next_node += len(vertices)
            loops.append(gmsh.model.geo.addSurfaceLoop([surface]))
        cortical = gmsh.model.geo.addVolume(loops)                         # outer loop, cores as holes
        trabecular = [gmsh.model.geo.addVolume([loop]) for loop in loops[1:]]
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(3, [cortical], name="cortical")
        if trabecular:
            gmsh.model.addPhysicalGroup(3, trabecular, name="trabecular")
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)
        if optimizer == "relocate":
            gmsh.model.mesh.optimize("Relocate3D")
        elif optimizer == "netgen":
            gmsh.model.mesh.optimize("Netgen")
        if save_prefix:
            gmsh.write(f"{save_prefix}.msh")
            gmsh.write(f"{save_prefix}.inp")
        tags, coords, _ = gmsh.model.mesh.getNodes()
        index = np.zeros(int(tags.max()) + 1, dtype=np.int64)
        index[tags.astype(np.int64)] = np.arange(len(tags))
        tets, region = [], []
        for volume, label in [(cortical, CORTICAL)] + [(v, TRABECULAR) for v in trabecular]:
            _, _, element_nodes = gmsh.model.mesh.getElements(3, volume)
            block = index[np.asarray(element_nodes[0], dtype=np.int64).reshape(-1, 4)]
            tets.append(block)
            region.append(np.full(len(block), label, dtype=np.int8))
        return coords.reshape(-1, 3), np.vstack(tets), np.concatenate(region)
    finally:
        gmsh.finalize()


# ---------------------------------------------------------------------------------------------- step 6
def ct_at_elements(out, case, nodes, tets):
    """CT intensity (HU) at each tet centroid, linearly interpolated: a check that the cortex is denser."""
    ct_path = source_dir(out) / case["ct_file"]
    if not ct_path.exists():
        return None
    img = nib.load(str(ct_path))
    voxel = nib.affines.apply_affine(np.linalg.inv(img.affine), nodes[tets].mean(axis=1))
    return ndimage.map_coordinates(img.get_fdata(dtype=np.float32), voxel.T, order=1)


def mesh_biphasic(out, method, case, bone, params=None, thresholds=None, overwrite=False):
    params, thresholds = params or BiphasicParams(), thresholds or Thresholds()
    folder = mesh_dir(out, method, case) / "biphasic"
    folder.mkdir(parents=True, exist_ok=True)
    prefix = folder / f"{bone}_biphasic"
    row = {"method": method, "volume_id": case["volume_id"], "case": case_dir_name(case), "structure": bone,
           **{f"param_{k}": v for k, v in asdict(params).items()}}

    outer = load_outer_surface(out, method, case, bone)
    target_edge = params.target_edge_mm or float(np.linalg.norm(outer[0][outer[1][:, 0]] - outer[0][outer[1][:, 1]], axis=1).mean())
    depth, origin = depth_map(*outer, params.grid_spacing_mm)
    cores, thickness = [], []
    for v, f in core_surfaces(depth, origin, params.grid_spacing_mm, params.cortical_thickness_mm, params.min_core_ml):
        v, f, local_thickness = tidy_core(v, f, target_edge, *outer)
        cores.append((v, f))
        thickness.append(local_thickness)
    if cores:
        trimesh.util.concatenate([trimesh.Trimesh(v, f, process=False) for v, f in cores]).export(str(folder / f"{bone}_trabecular_core.stl"))
        thickness = np.concatenate(thickness)
        row.update(core_pieces=len(cores), cortex_thickness_min_mm=float(thickness.min()),
                   cortex_thickness_mean_mm=float(thickness.mean()), cortex_thickness_max_mm=float(thickness.max()))

    nodes, tets, region = biphasic_tets(outer, cores, params.tet_optimizer, save_prefix=str(prefix))
    hu = ct_at_elements(out, case, nodes, tets)
    grid = pv.UnstructuredGrid(np.hstack([np.full((len(tets), 1), 4), tets]).ravel(),
                               np.full(len(tets), pv.CellType.TETRA, dtype=np.uint8), nodes)
    grid.cell_data["region"] = region                                      # 0 cortical, 1 trabecular
    scaled_jacobian = np.zeros(len(tets))
    for code, name in REGION_NAMES.items():
        part = region == code
        if not part.any():
            continue
        metrics = tet_metrics(nodes, tets[part])
        row.update({f"{name}_tets": int(part.sum()), f"{name}_volume_ml": float(metrics["volume"].sum() / 1000),
                    f"{name}_inverted": int((metrics["scaled_jacobian"] <= 0).sum())})
        row.update(summarize(f"{name}_tet", metrics, thresholds.limits("tet")))
        if hu is not None:
            row[f"{name}_mean_HU"] = float(hu[part].mean())
        scaled_jacobian[part] = metrics["scaled_jacobian"]
    grid.cell_data["scaled_jacobian"] = scaled_jacobian
    if hu is not None:
        grid.cell_data["HU"] = hu
    grid.save(str(prefix) + ".vtu")
    row["files"] = f"{prefix.name}.msh / .inp / .vtu, {bone}_trabecular_core.stl"
    row["status"] = "completed"
    return row


def _job(args):
    out, method, case, bone, params, thresholds = args
    try:
        return [mesh_biphasic(out, method, case, bone, params, thresholds)]
    except Exception as exc:
        return [{"method": method, "case": case_dir_name(case), "structure": bone,
                 "status": f"failed: {type(exc).__name__}: {exc}"}]


def biphasic_all(out, methods, cases, bones=None, params=None, thresholds=None, workers=2):
    """Biphasic meshes for every method x case x bone; summary in evaluation/biphasic_summary.csv."""
    params, thresholds = params or BiphasicParams(), thresholds or Thresholds()
    jobs = [(out, m, c, b, params, thresholds) for c in cases for m in methods for b in (bones or default_bones())
            if (mesh_dir(out, m, c) / "individual_stl" / f"{b}.stl").exists()]
    rows = run_jobs(_job, jobs, workers, lambda j: f"biphasic {j[1]} / {case_dir_name(j[2])} / {j[3]}")
    table = pd.DataFrame(rows)
    (Path(out) / "evaluation").mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(out) / "evaluation" / "biphasic_summary.csv", index=False)
    return table


def show_biphasic(vtu_path, native_window=False):
    """Tet mesh cut open by a draggable plane (drag the arrow / plane): orange = cortical, blue = trabecular."""
    from matplotlib.colors import ListedColormap
    grid = pv.read(str(vtu_path))
    plotter = pv.Plotter(window_size=(1000, 750), notebook=False if native_window else None)
    plotter.add_mesh_clip_plane(grid, normal="-x", scalars="region", cmap=ListedColormap(["#eb6834", "#2a78d6"]),
                                clim=[-0.5, 1.5], crinkle=True, show_edges=True, edge_color="#52514e", edge_opacity=0.3,
                                show_scalar_bar=False)
    plotter.add_text("orange = cortical shell   blue = trabecular core", font_size=10, color="black")
    plotter.view_yz()   # look straight at the cut face
    plotter.show()
    return plotter
