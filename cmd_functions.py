from pathlib import Path
import json
from spine_mesh_pipeline.case_extractor import load_cases, select_cases, manifest_dataset
from spine_mesh_pipeline import verse, totalseg
from spine_mesh_pipeline.segmentators import run_torch_model, import_label_maps, zip_label_maps
from spine_mesh_pipeline.find_directory import source_dir, mesh_dir, label_path
from spine_mesh_pipeline.labels_specs import ROWS_FILE, DEFAULT_DATASET, GROUND_TRUTH_VB, REFERENCES
from spine_mesh_pipeline.data_fetcher import fetch, fetch_ground_truth
from spine_mesh_pipeline.meshing import MeshParams
from spine_mesh_pipeline.report import build_reports
from spine_mesh_pipeline.mesh_quality import Thresholds
from spine_mesh_pipeline.mesh_pipeline import mesh_all, accuracy_all, check_inputs
from spine_mesh_pipeline.biphasic import BiphasicParams, biphasic_all

def pick_cases(args):
    cases = load_cases(args.out)
    if getattr(args, "cases", None):
        wanted = set(args.cases)
        cases = [c for c in cases if c["volume_id"] in wanted or f"{c['sample_index']:03d}" in wanted]
    if getattr(args, "limit", None):
        cases = cases[:args.limit]
    return cases

def cmd_select(args):
    manifest = Path(args.out)/"selected_manifest.csv"
    existing = manifest_dataset(args.out)
    if existing and existing != args.dataset:
        raise SystemExit(f"{args.out} holds a {existing} selection; give {args.dataset} its own --out folder.")
    if manifest.exists() and not args.reselect:
        cases = load_cases(args.out)
        print(f"Re-using{manifest}({len(cases)}cases); pass --reselect to draw a new sample.")
    elif args.dataset == verse.DATASET:
        verse.select_cases(args.out, args.n, args.seed, args.min_caudal_mm)
        cases = load_cases(args.out)
    elif args.dataset == totalseg.DATASET:
        totalseg.select_cases(args.out, args.n, args.seed, args.zip or totalseg.ZIP_URL, args.min_age)
        cases = load_cases(args.out)
    else:
        cases = select_cases(args.out,args.n, args.seed, args.revision)
    fetch_ground_truth(args.out,cases,args.revision)
    if args.dataset == DEFAULT_DATASET:  # the other datasets' labels and CTs are downloaded during selection
        fetch_ground_truth(args.out, cases, args.revision)
        if args.download_ct:
            for case in cases:
                fetch(case["ct_file"], source_dir(args.out), args.revision)
    print(f"{len(cases)} cases ready in {args.out}")

def cmd_segment(args):
    run_torch_model(args.out, args.method, pick_cases(args), args.torch_python, args.revision,
                    args.device, args.fast, args.overwrite,args.vbseg_variant)

def cmd_biphasic(args):
    params = BiphasicParams(cortical_thickness_mm=args.thickness, grid_spacing_mm=args.grid_spacing, tet_optimizer=args.tet)
    table = biphasic_all(args.out, args.methods, pick_cases(args), args.bones, params, thresholds_from(args), args.workers)
    print(f"{(table['status'] == 'completed').sum()}/{len(table)} bones done; summary: {args.out}/evaluation/biphasic_summary.csv")

def cmd_import(args):
    for volume_id, status in import_label_maps(args.out, args.method, pick_cases(args), args.source):
            print(f"{volume_id}: {status}")

def cmd_zip(args):
    zip_label_maps(args.out, args.methods, pick_cases(args), args.archive)


def thresholds_from(args):
    return Thresholds(args.tri_max_skewness, args.tet_max_skewness, args.tri_max_aspect_ratio, args.tet_max_aspect_ratio,
                      args.max_size_change, args.min_scaled_jacobian, args.max_fidelity_hd95)

def cmd_mesh(args):
    params = MeshParams(smoothing_factor=args.smoothing, closing_voxels=args.closing_voxels,
                        max_topology_closing_voxels=args.max_topology_closing,
                        opening_voxels=args.opening_voxels, target_edge_mm=args.target_edge,
                        remesh_iterations=args.remesh_iterations, min_structure_ml=args.min_structure_ml)
    rows = mesh_all(args.out, args.methods, pick_cases(args), params, thresholds_from(args), args.tet,
                    args.workers, args.save_element_metrics, args.overwrite)
    ok = sum(r.get("status") == "completed" for r in rows)
    print(f"Meshed {ok}/{len(rows)} structures. Next: python run_pipeline.py evaluate --out {args.out} --methods {' '.join(args.methods)}")

def cmd_evaluate(args):
    cases = pick_cases(args)
    inputs = check_inputs(args.out, args.methods, cases)
    problems = inputs[inputs["problem"] != ""]
    if len(problems):
        print("WARNING - these inputs make the comparison misleading:")
        print(problems[["case", "method", "problem", "label_file"]].to_string(index=False))
    quality_rows = []
    for method in args.methods:
        for case in cases:
            rows_file = mesh_dir(args.out, method, case) / ROWS_FILE
            if rows_file.exists():
                quality_rows.extend(json.loads(rows_file.read_text()))
            elif method != GROUND_TRUTH_VB or label_path(args.out, method, case).exists():
                print(f"no meshes for {method} / {case['volume_id']} - run the mesh stage first")
    if not quality_rows:
        raise SystemExit("Nothing to evaluate.")
    accuracy_rows = accuracy_all(args.out, args.methods, cases, args.workers) if set(REFERENCES) & set(args.methods) else []
    eval_dir = build_reports(args.out, quality_rows, accuracy_rows, thresholds_from(args), problems)
    print(f"Wrote {eval_dir / 'report.html'}")