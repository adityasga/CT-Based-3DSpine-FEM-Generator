"""Aggregate per-mesh rows into pass/fail flags, summary tables and a self-contained HTML report."""
import html
import json
import zlib
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .mesh_quality import Thresholds
from .labels_specs import MANIFEST_COLUMNS, Manifest_name, VERTEBRAL_BODY_METHODS

# Reference categorical palette, slots 1-3 validate all-pairs in light and dark; more
# approaches keep fixed slots (each strip has its own labelled row, so colour is secondary).
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

def write_selection(out, selected, audit, run_info):
    """Write selected_manifest.csv (rows of MANIFEST_COLUMNS, numbered in order), eligibility_audit.csv, run_info.json."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with (out / "eligibility_audit.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(key for row in audit for key in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(audit)
    with (out / Manifest_name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for index, row in enumerate(selected, 1):
            writer.writerow({column: row.get(column, "") for column in MANIFEST_COLUMNS} | {"sample_index": index})
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

def add_pass_flags(df, t):
    def col(name, default=np.nan):
        return df[name] if name in df else pd.Series(default, index=df.index)

    df["surface_valid"] = (
        col("surf_watertight", False).fillna(False).astype(bool) & col("surf_winding_consistent", False).fillna(False).astype(bool)
        & col("surf_outward_normals", False).fillna(False).astype(bool) & (col("surf_shells") == 1)
        & (col("surf_boundary_edges") == 0) & (col("surf_non_manifold_edges") == 0)
        & (col("surf_non_manifold_vertices") == 0) & (col("surf_self_intersecting_faces") == 0)
        & (col("tri_inverted") == 0)
    )
    df["surface_quality_pass"] = (
        (col("tri_skewness_max") <= t.tri_max_skewness) & (col("tri_aspect_ratio_max") <= t.tri_max_aspect_ratio)
        & (col("tri_size_change_max") <= t.max_size_change) & (col("tri_scaled_jacobian_min") >= t.min_scaled_jacobian)
    )
    df["resolution_pass"] = col("fidelity_hd95_mm") <= t.max_fidelity_hd95_mm
    tet_status = col("tet_status", None)
    tet_ran = tet_status.notna()
    df["tet_quality_pass"] = (
        tet_status.eq("ok") & (col("tet_inverted") == 0) & (col("tet_skewness_max") <= t.tet_max_skewness)
        & (col("tet_aspect_ratio_max") <= t.tet_max_aspect_ratio) & (col("tet_size_change_max") <= t.max_size_change)
        & (col("tet_scaled_jacobian_min") >= t.min_scaled_jacobian)
    ).where(tet_ran)
    expected = col("genus_expected")
    df["topology_ok"] = (col("surf_genus") <= expected).where(expected.notna())
    df["stl_pass"] = (df["surface_valid"] & df["topology_ok"].fillna(True).astype(bool)
                      & df["surface_quality_pass"] & df["resolution_pass"])
    df["all_pass"] = df["stl_pass"] & df["tet_quality_pass"].fillna(True).astype(bool)
    return df


def _pct(series):
    series = series.dropna()
    return float(100.0 * series.astype(bool).mean()) if len(series) else np.nan


def summarize_quality(df, keys):
    skipped = df["status"].astype(str).str.startswith("skipped")
    fragments = df[skipped].groupby(keys, dropna=False).size()
    df = df[~skipped]
    completed = df[df["status"] == "completed"]
    grouped = df.groupby(keys, dropna=False)
    out = grouped.size().rename("meshes").to_frame()
    out["completed"] = grouped["status"].apply(lambda s: int((s == "completed").sum()))
    out["fragments_skipped"] = fragments.reindex(out.index).fillna(0).astype(int)
    g = completed.groupby(keys, dropna=False)
    for flag in ("surface_valid", "topology_ok", "surface_quality_pass", "resolution_pass", "stl_pass",
                 "tet_quality_pass", "all_pass"):
        out[f"{flag}_pct"] = g[flag].apply(_pct)
    for column, stat in [("tri_skewness_max", "median"), ("tri_aspect_ratio_max", "median"), ("tri_scaled_jacobian_min", "median"),
                         ("tri_size_change_max", "median"), ("tri_min_angle_deg", "median"), ("fidelity_hd95_mm", "median"),
                         ("surf_triangles", "median"), ("surf_edge_mean_mm", "mean"), ("surf_genus", "median"),
                         ("voxels_removed_islands", "sum"), ("voxels_filled_cavities", "sum"),
                         ("tet_elements", "median"), ("tet_skewness_max", "median"), ("tet_aspect_ratio_max", "median"),
                         ("tet_scaled_jacobian_min", "median"), ("tet_size_change_max", "median"), ("tet_inverted", "sum"),
                         ("tet_scaled_jacobian_pct_fail", "mean"), ("surface_seconds", "sum"), ("tet_seconds", "sum")]:
        if column in completed:
            out[f"{column}_{stat}"] = g[column].agg(stat)
    return out.reset_index()


def summarize_accuracy(acc, keys):
    both = acc[acc["detection"] == "both"]
    grouped = acc.groupby(keys, dropna=False)
    out = grouped.size().rename("structures").to_frame()
    out["missed"] = grouped["detection"].apply(lambda s: int((s == "missed").sum()))
    out["not_in_ground_truth"] = grouped["detection"].apply(lambda s: int((s == "not_in_ground_truth").sum()))
    g = both.groupby(keys, dropna=False)
    out["dice_mean"] = g["dice"].mean()
    out["dice_median"] = g["dice"].median()
    out["dice_min"] = g["dice"].min()
    if "precision" in both:
        out["precision_mean"] = g["precision"].mean()
        out["recall_mean"] = g["recall"].mean()
    if "surface_hd95_mm" in both:
        out["surface_hd95_mm_median"] = g["surface_hd95_mm"].median()
        out["surface_assd_mm_mean"] = g["surface_assd_mm"].mean()
    out["volume_diff_pct_mean"] = g["volume_diff_pct"].mean()
    return out.reset_index()


# ----------------------------------------------------------------------------- HTML
def _nice_ticks(lo, hi, n=5):
    span = hi - lo if hi > lo else abs(hi) or 1.0
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = np.floor(lo / step) * step
    ticks = np.arange(start, hi + step * 0.5, step)
    return ticks[(ticks >= lo - 1e-9) & (ticks <= hi + 1e-9)]


def _fmt(value, digits=2):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def strip_chart(df, column, methods, title, subtitle, limit=None, limit_kind="max", digits=2):
    data = df[["method", "case", "structure", column]].copy()
    data[column] = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[column])
    rows = [m for m in methods if (data["method"] == m).any()]
    if not rows:
        return ""
    values = data[column].to_numpy()
    lo, hi = float(values.min()), float(values.max())
    if limit is not None:
        lo, hi = min(lo, limit), max(hi, limit)
    pad = (hi - lo) * 0.06 or 0.05
    lo, hi = lo - pad, hi + pad
    width, left, right, top, row_h = 520, 132, 16, 30, 54
    height = top + row_h * len(rows) + 34
    plot_w = width - left - right

    def x(v):
        return left + (v - lo) / (hi - lo) * plot_w

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">']
    if limit is not None:
        lx = x(limit)
        fx, fw = (lx, width - right - lx) if limit_kind == "max" else (left, lx - left)
        parts.append(f'<rect x="{fx:.1f}" y="{top - 6}" width="{max(fw, 0):.1f}" height="{row_h * len(rows) + 6}" class="fail-zone"/>')
    for tick in _nice_ticks(lo, hi):
        tx = x(tick)
        parts.append(f'<line x1="{tx:.1f}" x2="{tx:.1f}" y1="{top - 6}" y2="{top + row_h * len(rows)}" class="grid"/>')
        parts.append(f'<text x="{tx:.1f}" y="{top + row_h * len(rows) + 18}" class="tick" text-anchor="middle">{_fmt(float(tick), digits)}</text>')
    if limit is not None:
        lx = x(limit)
        parts.append(f'<line x1="{lx:.1f}" x2="{lx:.1f}" y1="{top - 10}" y2="{top + row_h * len(rows)}" class="limit"/>')
        parts.append(f'<text x="{lx:.1f}" y="{top - 14}" class="tick" text-anchor="middle">limit {_fmt(limit, digits)}</text>')
    for i, method in enumerate(rows):
        slot = methods.index(method) % len(SERIES_LIGHT) + 1
        cy = top + row_h * i + row_h / 2
        sub = data[data["method"] == method]
        med = float(sub[column].median())
        parts.append(f'<text x="{left - 14}" y="{cy - 3:.1f}" class="row-label" text-anchor="end">{html.escape(method)}</text>')
        parts.append(f'<text x="{left - 14}" y="{cy + 13:.1f}" class="row-sub" text-anchor="end">median {_fmt(med, digits)} · n={len(sub)}</text>')
        for _, r in sub.iterrows():
            jitter = (zlib.crc32(f"{r['case']}{r['structure']}".encode()) % 1000) / 1000 * 26 - 13
            tip = html.escape(f"{method} · {r['case']} · {r['structure']} · {_fmt(float(r[column]), 3)}")
            px = x(float(r[column]))
            parts.append(f'<g class="dot s{slot}" data-tip="{tip}"><circle cx="{px:.1f}" cy="{cy + jitter:.1f}" r="9" class="hit"/>'
                         f'<circle cx="{px:.1f}" cy="{cy + jitter:.1f}" r="4"/></g>')
        mx = x(med)
        parts.append(f'<line x1="{mx:.1f}" x2="{mx:.1f}" y1="{cy - 18:.1f}" y2="{cy + 18:.1f}" class="median"/>')
    parts.append("</svg>")
    return (f'<figure class="chart"><figcaption><strong>{html.escape(title)}</strong>'
            f'<span>{html.escape(subtitle)}</span></figcaption>{"".join(parts)}</figure>')


def pretty(column):
    text = column.replace("method", "approach").replace("structure_class", "region")
    for suffix, unit in (("_pct", " %"), ("_mm2", " (mm²)"), ("_mm", " (mm)"), ("_ml", " (ml)")):
        text = text.replace(suffix, unit)
    return text.replace("_", " ")


def table_html(df, columns, headers=None, digits=2):
    headers = headers or [pretty(c) for c in columns]
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in columns:
            v = r.get(c, np.nan)
            places = 1 if c.endswith("pct") else digits
            text = html.escape(str(v)) if isinstance(v, str) else _fmt(v if not isinstance(v, (np.floating,)) else float(v), places)
            cells.append(f'<td class="{"txt" if isinstance(v, str) else "num"}">{text}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


CSS = """
:root { color-scheme: light; --page:#f9f9f7; --surface-1:#fcfcfb; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --wash:#f0efec; --border:rgba(11,11,11,0.10);
  --s1:%s; --s2:%s; --s3:%s; --s4:%s; --s5:%s; --s6:%s; --s7:%s; --s8:%s; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --page:#0d0d0d; --surface-1:#1a1a19;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --grid:#2c2c2a; --axis:#383835; --wash:#262624; --border:rgba(255,255,255,0.10);
  --s1:%s; --s2:%s; --s3:%s; --s4:%s; --s5:%s; --s6:%s; --s7:%s; --s8:%s; } }
:root[data-theme="dark"] { color-scheme: dark; --page:#0d0d0d; --surface-1:#1a1a19; --text-primary:#ffffff; --text-secondary:#c3c2b7;
  --grid:#2c2c2a; --axis:#383835; --wash:#262624; --border:rgba(255,255,255,0.10);
  --s1:%s; --s2:%s; --s3:%s; --s4:%s; --s5:%s; --s6:%s; --s7:%s; --s8:%s; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--text-primary); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1080px; margin:0 auto; padding:32px 16px 64px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:40px 0 8px; } h3 { font-size:15px; margin:24px 0 8px; }
p, li { color:var(--text-secondary); max-width:78ch; }
.meta { color:var(--muted); font-size:13px; }
.card { background:var(--surface-1); border:1px solid var(--border); border-radius:12px; padding:16px; margin:12px 0; }
.table-wrap { overflow-x:auto; background:var(--surface-1); border:1px solid var(--border); border-radius:12px; }
table { border-collapse:collapse; width:100%%; font-size:13px; }
th, td { padding:6px 10px; border-bottom:1px solid var(--grid); white-space:nowrap; }
th { text-align:left; color:var(--text-secondary); font-weight:600; position:sticky; top:0; background:var(--surface-1); }
td.num { text-align:right; font-variant-numeric:tabular-nums; } tr:last-child td { border-bottom:0; }
.charts { display:grid; grid-template-columns:repeat(auto-fit,minmax(440px,1fr)); gap:12px; }
figure.chart { margin:0; background:var(--surface-1); border:1px solid var(--border); border-radius:12px; padding:14px 12px 6px; }
figcaption { display:flex; flex-direction:column; margin:0 4px 6px; } figcaption span { color:var(--text-secondary); font-size:12px; }
svg { width:100%%; height:auto; display:block; overflow:visible; }
.grid { stroke:var(--grid); stroke-width:1; } .limit { stroke:var(--text-secondary); stroke-width:1; }
.fail-zone { fill:var(--wash); } .median { stroke:var(--text-primary); stroke-width:2; stroke-linecap:round; }
.tick { fill:var(--muted); font-size:10.5px; font-variant-numeric:tabular-nums; }
.row-label { fill:var(--text-primary); font-size:12px; font-weight:600; } .row-sub { fill:var(--text-secondary); font-size:10.5px; }
.dot circle:not(.hit) { stroke:var(--surface-1); stroke-width:2; } .dot .hit { fill:transparent; }
.dot:hover circle:not(.hit) { r:6; }
.s1 circle:not(.hit){fill:var(--s1)} .s2 circle:not(.hit){fill:var(--s2)} .s3 circle:not(.hit){fill:var(--s3)} .s4 circle:not(.hit){fill:var(--s4)}
.s5 circle:not(.hit){fill:var(--s5)} .s6 circle:not(.hit){fill:var(--s6)} .s7 circle:not(.hit){fill:var(--s7)} .s8 circle:not(.hit){fill:var(--s8)}
#tip { position:fixed; pointer-events:none; background:var(--surface-1); color:var(--text-primary); border:1px solid var(--border);
  border-radius:8px; padding:6px 10px; font-size:12px; box-shadow:0 4px 16px rgba(0,0,0,.12); display:none; z-index:10; }
code { font-size:12px; }
@media (max-width:520px) { .charts { grid-template-columns:1fr; } }
""" % tuple(SERIES_LIGHT + SERIES_DARK + SERIES_DARK)

JS = """
const tip = document.getElementById('tip');
document.querySelectorAll('.dot').forEach(g => {
  g.addEventListener('mouseenter', () => { tip.textContent = g.dataset.tip; tip.style.display = 'block'; });
  g.addEventListener('mousemove', e => { tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY + 12) + 'px'; });
  g.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
});
"""


def write_report(path, quality, accuracy, by_method, by_class, acc_by_method, acc_by_class, t, params, problems=None):
    methods = list(dict.fromkeys(quality["method"])) if len(quality) else []
    done = quality[quality["status"] == "completed"]
    s = []
    s.append("<h1>Spine & pelvis mesh evaluation</h1>")
    if problems is not None and len(problems):
        s.append("<div class='card'><b>Warning: some inputs make this comparison misleading.</b> Fix these and re-run "
                 "segment / mesh before trusting the numbers.")
        s.append(table_html(problems, ["case", "method", "problem", "label_file"]) + "</div>")
    n_cases = done["case"].nunique() if len(done) else 0
    s.append(f'<p class="meta">{len(done):,} meshes from {n_cases} case{"s" if n_cases != 1 else ""} · '
             f'approaches: {", ".join(methods)} · smoothing factor {params.get("mesh", {}).get("smoothing_factor", "–")} · '
             f'tet optimiser {params.get("tet_optimizer", "–")}</p>')
    s.append("<h2>Pass criteria</h2><div class='card'><ul>"
             "<li><b>Valid surface</b>: watertight, consistently wound, outward normals, one shell, no boundary / non-manifold "
             "edges or vertices, no self-intersections, no inverted (folded) triangles.</li>"
             "<li><b>Anatomical genus</b>: no tunnels beyond the bone's real openings (1 for thoracic/lumbar vertebrae and hips, "
             "3 for cervical vertebrae, 0 for vertebral bodies; sacrum and S1 not checked).</li>"
             f"<li><b>Surface element quality</b> (every triangle): skewness ≤ {t.tri_max_skewness}, aspect ratio ≤ {t.tri_max_aspect_ratio}, "
             f"size change ≤ {t.max_size_change}, scaled Jacobian ≥ {t.min_scaled_jacobian}.</li>"
             f"<li><b>Volume element quality</b> (every tetrahedron): skewness ≤ {t.tet_max_skewness}, aspect ratio ≤ {t.tet_max_aspect_ratio}, "
             f"size change ≤ {t.max_size_change}, scaled Jacobian ≥ {t.min_scaled_jacobian}, no inverted elements.</li>"
             f"<li><b>Sufficient resolution</b>: edge length ≈ CT in-plane voxel spacing and 95th-percentile distance between the final "
             f"mesh and the unsmoothed segmentation surface ≤ {t.max_fidelity_hd95_mm} mm.</li></ul></div>")

    s.append("<h2>Mesh quality by approach</h2>")
    overview = [c for c in [
        ("method", "approach"), ("meshes", "meshes"), ("completed", "completed"), ("surface_valid_pct", "valid surface %"),
        ("topology_ok_pct", "anatomical genus %"), ("surface_quality_pass_pct", "triangle quality %"),
        ("resolution_pass_pct", "resolution %"), ("stl_pass_pct", "STL meets all %"),
        ("tet_quality_pass_pct", "tet quality %"), ("all_pass_pct", "STL + tets %"),
        ("tri_skewness_max_median", "worst skew (median)"), ("tri_aspect_ratio_max_median", "worst AR (median)"),
        ("tri_scaled_jacobian_min_median", "worst SJ (median)"), ("tri_size_change_max_median", "worst size change (median)"),
        ("fidelity_hd95_mm_median", "fidelity HD95 mm"), ("tet_inverted_sum", "inverted tets"),
        ("tet_scaled_jacobian_min_median", "worst tet SJ (median)")] if c[0] in by_method]
    s.append(table_html(by_method, [c for c, _ in overview], [h for _, h in overview]))
    s.append("<p>Medians are over meshes of each mesh's worst element. Full per-mesh rows: <code>mesh_quality.csv</code>.</p>")
    if params.get("tet_optimizer") in ("fast", "relocate") and "tet_scaled_jacobian_pct_fail_mean" in by_method:
        share = by_method["tet_scaled_jacobian_pct_fail_mean"].mean()
        s.append(f"<p>Tet quality uses the <b>{params['tet_optimizer']}</b> Gmsh optimiser, which leaves a few slivers per bone "
                 f"(on average {share:.4f}% of tets below the scaled-Jacobian limit), so the strict every-element tet criterion "
                 "rarely passes. That comes from the mesher, not the STL: rerun with <code>--tet netgen</code> for final FE meshes.</p>")

    if len(accuracy):
        s.append("<h2>Segmentation accuracy against ground truth</h2>"
                 "<p>Dice, precision and recall on the voxel label maps; surface distances between the cleaned meshes. "
                 "<code>sacrum_with_S1</code> scores sacrum and S1 together because the S1 cut plane is a convention that differs between sources.</p>")
        if set(accuracy["method"]) & VERTEBRAL_BODY_METHODS:
            s.append("<p><b>Vertebral-body approaches</b> (vbsegm2step) segment the body without the posterior arch. "
                     "Against <code>ground_truth</code> (whole vertebrae) their Dice and recall are low by construction; read "
                     "<b>precision</b> instead: the share of each predicted body inside the right ground-truth vertebra, so a "
                     "value near 1 means the right level was found. Rows with reference <code>ground_truth_vb</code> compare "
                     "against published vertebral-body labels (VerSe and TotalSegmentator v2 scans only).</p>")
        s.append(table_html(acc_by_method, [c for c in acc_by_method.columns]))
        s.append("<h3>By anatomical region</h3>")
        s.append(table_html(acc_by_class, [c for c in acc_by_class.columns]))

    s.append("<h2>Distributions</h2><p>One dot per mesh (its worst element); the black tick is the median; "
             "the shaded side of the limit line fails. Hover a dot for the case and structure.</p><div class='charts'>")
    charts = [
        ("tri_skewness_max", "Surface: worst skewness per mesh", "equilateral-area skewness, lower is better", t.tri_max_skewness, "max", 2),
        ("tri_aspect_ratio_max", "Surface: worst aspect ratio per mesh", "Verdict aspect ratio, 1 is ideal", t.tri_max_aspect_ratio, "max", 2),
        ("tri_scaled_jacobian_min", "Surface: worst scaled Jacobian per mesh", "≤ 0 would be an inverted triangle", t.min_scaled_jacobian, "min", 2),
        ("tri_size_change_max", "Surface: worst size change per mesh", "largest edge-length ratio between neighbours", t.max_size_change, "max", 2),
        ("fidelity_hd95_mm", "Resolution: surface fidelity HD95", "mm between final mesh and unsmoothed segmentation", t.max_fidelity_hd95_mm, "max", 2),
        ("tet_scaled_jacobian_min", "Volume: worst tet scaled Jacobian per mesh", "≤ 0 would be an inverted tetrahedron", t.min_scaled_jacobian, "min", 2),
        ("tet_skewness_max", "Volume: worst tet skewness per mesh", "equilateral-volume skewness", t.tet_max_skewness, "max", 2),
        ("tet_aspect_ratio_max", "Volume: worst tet aspect ratio per mesh", "Verdict aspect ratio, 1 is ideal", t.tet_max_aspect_ratio, "max", 1),
        ("tet_size_change_max", "Volume: worst tet size change per mesh", "largest edge-length ratio between face neighbours", t.max_size_change, "max", 2),
    ]
    for column, title, subtitle, limit, kind, digits in charts:
        if column in done:
            s.append(strip_chart(done, column, methods, title, subtitle, limit, kind, digits))
    if len(accuracy) and "dice" in accuracy:
        both = accuracy[accuracy["detection"] == "both"]
        # one strip per (approach, reference): "vbsegm2step vs ground_truth_vb" next to "vbsegm2step"
        both = both.assign(method=[m if r == "ground_truth" else f"{m} vs {r}" for m, r in zip(both["method"], both["reference"])])
        acc_methods = [label for m in methods for label in dict.fromkeys(both["method"]) if label.split(" vs ")[0] == m]
        s.append(strip_chart(both, "dice", acc_methods, "Accuracy: Dice per structure", "voxel overlap with ground truth, 1 is perfect"))
        if "surface_hd95_mm" in both:
            s.append(strip_chart(both, "surface_hd95_mm", acc_methods, "Accuracy: surface HD95 per structure", "mm between approach mesh and ground-truth mesh"))
    s.append("</div>")

    s.append("<h2>Mesh quality by anatomical region</h2>")
    s.append(table_html(by_class, [c for c in by_class.columns if c not in ("voxels_removed_islands_sum", "voxels_filled_cavities_sum")]))

    failing = done[~done["all_pass"]]
    if len(failing):
        reasons = []
        for _, r in failing.iterrows():
            why = [name for flag, name in (("surface_valid", "surface"), ("topology_ok", "genus"), ("surface_quality_pass", "triangles"),
                                           ("resolution_pass", "resolution"), ("tet_quality_pass", "tets"))
                   if r.get(flag) is False or r.get(flag) == 0]
            reasons.append(", ".join(why))
        failing = failing.assign(failed=reasons)
        s.append(f"<h2>Meshes that miss a criterion ({len(failing)})</h2>")
        s.append(table_html(failing.head(300), [c for c in ["method", "case", "structure", "failed", "tri_skewness_max", "tri_scaled_jacobian_min",
                                                            "tri_size_change_max", "fidelity_hd95_mm", "tet_scaled_jacobian_min",
                                                            "tet_scaled_jacobian_n_fail", "tet_inverted"] if c in failing]))
    failed_runs = quality[quality["status"] != "completed"]
    if len(failed_runs):
        s.append(f"<h2>Labels not meshed ({len(failed_runs)})</h2><p><code>skipped_fragment</code> = a label smaller than "
                 "the minimum structure volume (typically a stray blob given a vertebra label); anything else is a failure.</p>")
        s.append(table_html(failed_runs, [c for c in ["method", "case", "structure", "status"] if c in failed_runs]))

    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Spine Mesh Evaluation</title><style>{CSS}</style></head><body><main>{''.join(s)}</main>"
            f"<div id='tip'></div><script>{JS}</script></body></html>")
    Path(path).write_text(page, encoding="utf-8")


def build_reports(out, quality_rows, accuracy_rows, thresholds=None, problems=None):
    t = thresholds or Thresholds()
    eval_dir = Path(out) / "evaluation"; eval_dir.mkdir(parents=True, exist_ok=True)
    params_file = Path(out) / "meshes" / "mesh_params.json"
    params = json.loads(params_file.read_text()) if params_file.exists() else {}
    quality = add_pass_flags(pd.DataFrame(quality_rows), t)
    quality.to_csv(eval_dir / "mesh_quality.csv", index=False)
    by_method = summarize_quality(quality, ["method"])
    by_class = summarize_quality(quality, ["method", "structure_class"])
    by_method.to_csv(eval_dir / "summary_by_method.csv", index=False)
    by_class.to_csv(eval_dir / "summary_by_method_and_region.csv", index=False)
    accuracy = pd.DataFrame(accuracy_rows)
    acc_by_method = acc_by_class = pd.DataFrame()
    if len(accuracy) and "dice" in accuracy:
        accuracy["reference"] = accuracy["reference"].fillna("ground_truth") if "reference" in accuracy else "ground_truth"
        accuracy.to_csv(eval_dir / "segmentation_accuracy.csv", index=False)
        acc_by_method = summarize_accuracy(accuracy, ["method", "reference"])
        acc_by_class = summarize_accuracy(accuracy, ["method", "reference", "structure_class"])
        acc_by_method.to_csv(eval_dir / "accuracy_by_method.csv", index=False)
        acc_by_class.to_csv(eval_dir / "accuracy_by_method_and_region.csv", index=False)
    write_report(eval_dir / "report.html", quality, accuracy, by_method, by_class, acc_by_method, acc_by_class, t, params, problems)
    return eval_dir
