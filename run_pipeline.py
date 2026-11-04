"""Compare segmentation approaches on CTSpinoPelvic1K by the quality of the meshes they produce.

Stages (each resumable; re-running skips finished work unless --overwrite):
  select    pick cases with the strict LSTV / L6 / T13 / hardware exclusions, download labels
  segment   run a PyTorch model locally (seg_torch conda env; CPU is slow - prefer Colab)
  import    bring in label maps produced elsewhere (Colab zip, other tools)
  mesh      clean every bone into a watertight STL + surface/tet element quality
  evaluate  accuracy vs ground truth, summary CSVs and evaluation/report.html

Typical run (ground truth + NV-Segment from Colab + TotalSegmentator):
  python run_pipeline.py select   --out ctspinopelvic1k_100_normal --n 100
  #   -> run colab/segment_on_colab.ipynb, download nvsegment_labels.zip / totalsegmentator_labels.zip
  python run_pipeline.py import   --out ctspinopelvic1k_100_normal --method nvsegment --source nvsegment_labels.zip
  python run_pipeline.py import   --out ctspinopelvic1k_100_normal --method totalsegmentator --source totalsegmentator_labels.zip
  python run_pipeline.py mesh     --out ctspinopelvic1k_100_normal --methods ground_truth nvsegment totalsegmentator
  python run_pipeline.py evaluate --out ctspinopelvic1k_100_normal --methods ground_truth nvsegment totalsegmentator

Several datasets in one command (segment, mesh, evaluate, biphasic, zip): list the folders, or use `all`:
  python run_pipeline.py segment  --out ctspinopelvic_normal totalsegmentator verse2020 --methods nvsegment vbsegm2step
  python run_pipeline.py mesh     --out all --methods ground_truth nvsegment vbsegm2step

Outputs:
  segmentations/<method>/<volume_id>_label.nii.gz
  meshes/<method>/<NNN_volume_id>/individual_stl/<structure>.stl, spine_combined.stl, pelvis_combined.stl
  evaluation/mesh_quality.csv, segmentation_accuracy.csv, summary_*.csv, accuracy_*.csv, report.html
"""

import argparse
from cmd_functions import cmd_select, cmd_segment, cmd_import, cmd_zip, cmd_mesh, cmd_evaluate, cmd_biphasic, run_on_folders
from spine_mesh_pipeline.mesh_quality import Thresholds
from spine_mesh_pipeline.labels_specs import default_torch_python, GROUND_TRUTH, Torch_Models, DEFAULT_DATASET, DATASETS

def add_thresholds(parser):
    t = Thresholds()
    g = parser.add_argument_group("pass/fail thresholds")
    g.add_argument("--tri-max-skewness", type=float, default=t.tri_max_skewness)
    g.add_argument("--tet-max-skewness", type=float, default=t.tet_max_skewness)
    g.add_argument("--tri-max-aspect-ratio", type=float, default=t.tri_max_aspect_ratio)
    g.add_argument("--tet-max-aspect-ratio", type=float, default=t.tet_max_aspect_ratio)
    g.add_argument("--max-size-change", type=float, default=t.max_size_change)
    g.add_argument("--min-scaled-jacobian", type=float, default=t.min_scaled_jacobian)
    g.add_argument("--max-fidelity-hd95", type=float, default=t.max_fidelity_hd95_mm, help="mm")

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest='command',required = True)

    def common(p, many_folders=True):
        if many_folders:
            p.add_argument('--out', nargs="+", default=["ctspinopelvic_normal"],
                           help="one or more project folders, or `all` for every folder with a selected_manifest.csv")
        else:
            p.add_argument('--out', default="ctspinopelvic_normal", help="project folder")
        p.add_argument("--cases", nargs="*", help ="only these volume_ids (or 3-digit sample indices)")
        p.add_argument("--limit", default = 500, type=int,help="only the first N selected cases")
        return p

    p = common(sub.add_parser("select",help="choose cases and download their labels"), many_folders=False)
    p.add_argument("--dataset", default=DEFAULT_DATASET, choices = DATASETS, help="choose between ctspinopelvic1k, verse2020 or totalseg_v3")
    p.add_argument("--n",type=int, default=1)
    p.add_argument("--seed", type=int, default=2026911)
    p.add_argument("--revision", default = 'main', help ="dataset commit hash to pin the release")
    p.add_argument("--reselect", action = "store_true", help = "draw a new sample even if manifest exists")
    p.add_argument("--download-ct", action = "store_true", help = "ctspinopelvic1k: also download the CTs (VerSe / TotalSegmentator CTs are always downloaded)" )
    p.add_argument("--min-caudal-mm", type=float, default=20.0, help="verse2020: scan needed below L5 (top of the sacrum)")
    p.add_argument("--min-age", type=float, default=18.0, help="totalseg_v3: youngest patient accepted")
    p.add_argument("--zip", help="totalseg_v3: local copy of Totalsegmentator_dataset_v300.zip (default: read from Zenodo)")
    p.set_defaults(func = cmd_select)

    p = common(sub.add_parser("segment", help="run a Pytorch segmentation model locally"))#need to run this on non-Mac platform
    p.add_argument("--methods","--method", dest="methods", nargs="+", required = True, choices = sorted (Torch_Models), help="one or more models; each writes to segmentations/<method>/")
    p.add_argument("--torch-python", default = default_torch_python(),help="python of the env with torch (default: seg_torch conda env")
    p.add_argument("--device", default = "auto", choices = ["auto", "cpu", "cuda","mps"])
    p.add_argument("--fast", action = "store_true", help = "TotalSegmentator 3mm model")
    p.add_argument("--vbseg-variant", default="ResEncL", choices=["ResEncL", "ResEncM"],
                   help="vbsegm2step whole-spine model: ResEncL (5-fold ensemble) or ResEncM (single model, ~5x faster)")
    p.add_argument("--revision", default = "main")
    p.add_argument("--overwrite", action = "store_true")
    p.set_defaults(func=cmd_segment)

    p = common(sub.add_parser("zip", help="zip all label maps of the given methods into one file"))
    p.add_argument("--methods", nargs="+", default=[GROUND_TRUTH, *Torch_Models])
    p.add_argument("--archive", help="output .zip (default <out>/segmentations/label_maps.zip)")
    p.set_defaults(func=cmd_zip)

    p = common(sub.add_parser("import", help="import label maps (folder or zip) for an approach"), many_folders=False)
    p.add_argument("--method", required=True, help="approach name, e.g. nvsegment, totalsegmentator, my_algo")
    p.add_argument("--source", required=True, help="folder or .zip containing <volume_id>_label.nii.gz")
    p.set_defaults(func=cmd_import)

    p = common(sub.add_parser("biphasic", help="cortical shell + trabecular core tet meshes (after `mesh`)"))
    p.add_argument("--methods", nargs="+", default=[GROUND_TRUTH])
    p.add_argument("--bones", nargs="+", help="default: the vertebrae in the region (S1 excluded)")
    p.add_argument("--thickness", type=float, default=1.0, help="cortical shell thickness, mm")
    p.add_argument("--grid-spacing", type=float, default=0.3, help="mm; finer = more accurate shell, slower")
    p.add_argument("--tet", default="fast", choices=["fast", "relocate", "netgen"])
    p.add_argument("--workers", type=int, default=2)
    add_thresholds(p)
    p.set_defaults(func=cmd_biphasic)

    p=common(sub.add_parser("mesh", help="clean meshes + element quality"))
    p.add_argument("--methods", nargs="+", default=[GROUND_TRUTH])
    p.add_argument("--workers", type=int, default=2, help="parallel cases (each uses ~1-3 GB RAM)")
    p.add_argument("--smoothing", type=float, default=0.5, help="Slicer smoothing factor 0-1")
    p.add_argument("--closing-voxels", type=int, default=1, help="morphological closing radius (0 = off)")
    p.add_argument("--max-topology-closing", type=int, default=3,
                   help="grow closing up to this radius until a bone has its anatomical genus (0 = off)")
    p.add_argument("--opening-voxels", type=int, default=0, help="morphological opening radius (0 = off)")
    p.add_argument("--target-edge", type=float, default=None, help="mm; default = CT in-plane voxel spacing")
    p.add_argument("--remesh-iterations", type=int, default=10)
    p.add_argument("--min-structure-ml", type=float, default=0.5, help="skip labels smaller than this (stray fragments)")
    p.add_argument("--tet", default="fast", choices=["fast", "relocate", "netgen", "none"],
                   help="tetrahedral check: fast (default), relocate, netgen (removes slivers, ~25x slower), none")
    p.add_argument("--save-element-metrics", action="store_true",
                   help="write per-element quality as .vtp/.vtu (ParaView) next to each STL")
    p.add_argument("--overwrite", action="store_true")
    add_thresholds(p)
    p.set_defaults(func=cmd_mesh)

    p = common(sub.add_parser("evaluate", help="accuracy vs ground truth + summaries + HTML report"))
    p.add_argument("--methods", nargs="+", default=[GROUND_TRUTH])
    p.add_argument("--workers", type=int, default=2)
    add_thresholds(p)
    p.set_defaults(func=cmd_evaluate)


    args=parser.parse_args()
    if isinstance(args.out, list):
        run_on_folders(args)
    else:
        args.func(args)

if __name__ == "__main__":
    main()
