#!/usr/bin/env python3
"""
main_results.py
=============================

1) Figures:
   - For each dataset (mnist, fmnist), create ONE 1x3 figure:
       alpha order: 1.0, 0.5, 0.1
     Each subplot contains 3 curves:
       fedavg, softSGD, dflaa
     Adds zoom-in inset for alpha=1.0 and 0.5 (to separate close curves).

2) Tables (text + LaTeX in one .txt file):
   - Final mean accuracy (%): mean ± std across seeds
   - Time to reach 70/80/90 (%): mean ± std across seeds
   Uses summary_time_of_round.csv only.

Expected layout (same as your sweep):
  {results_root}/{dataset}/alpha_{alpha_tag}/{aggregation}/seed_{seed}/summary_time_of_round.csv

summary_time_of_round.csv columns:
  time_s, round, mean, p10, p50, p90

Run example:
  python main_results.py \
    --results-root all_exp_results \
    --datasets mnist \
    --alphas 1.0,0.5,0.1 \
    --aggregations fedavg,softSGD,dflaa \
    --out-dir icdcs_paper_results \
    --tables-out ICDCS_tables.txt
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# ----------------------------
# Helpers
# ----------------------------
def fmt_alpha(a: float) -> str:
    s = f"{a:.3f}".rstrip("0").rstrip(".")
    if "." not in s:
        s = s + ".0"
    return s.replace(".", "p")


def try_parse_alpha_from_dirname(name: str) -> Optional[float]:
    m = re.search(r"alpha[_-]([0-9]+(?:[p\.][0-9]+)?)", name.lower())
    if not m:
        return None
    token = m.group(1).replace("p", ".")
    try:
        return float(token)
    except Exception:
        return None


def list_seed_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    out = []
    for p in root.iterdir():
        if p.is_dir() and (p / "summary_time_of_round.csv").exists():
            out.append(p)
    return sorted(out)


def load_summary_time_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    cols = {c.lower(): c for c in arr.dtype.names}

    time_col = cols.get("time_s") or cols.get("time") or cols.get("t")
    mean_col = cols.get("mean") or cols.get("avg") or cols.get("accuracy") or cols.get("acc")
    if time_col is None or mean_col is None:
        raise ValueError(f"Missing expected columns in {path}. Found: {arr.dtype.names}")

    t = np.asarray(arr[time_col], dtype=float)
    y = np.asarray(arr[mean_col], dtype=float)

    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if len(t) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    order = np.argsort(t)
    t, y = t[order], y[order]
    t = np.maximum.accumulate(t)

    # normalize time to start at 0 for nice plots
    t = t - float(t[0])

    # auto-scale if accuracy is in [0,1]
    if np.nanmax(y) <= 1.5:
        y = y * 100.0

    return t, y


def resample_to_common_grid(curves: List[Tuple[np.ndarray, np.ndarray]], dt: float = 25.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    curves: list of (t, y) for multiple seeds
    Returns:
      grid_t, Y  where Y shape = (num_seeds, len(grid_t)), with NaNs for out-of-range.
    """
    valid = [(t, y) for (t, y) in curves if len(t) > 1 and len(y) > 1]
    if not valid:
        return np.array([], dtype=float), np.empty((0, 0), dtype=float)

    # common time horizon = min of max times across seeds (keeps fair overlap)
    t_max_common = min(float(np.max(t)) for (t, _) in valid)
    if not np.isfinite(t_max_common) or t_max_common <= 0:
        return np.array([], dtype=float), np.empty((0, 0), dtype=float)

    grid_t = np.arange(0.0, t_max_common + 1e-9, dt, dtype=float)

    Y = np.full((len(valid), len(grid_t)), np.nan, dtype=float)
    for i, (t, y) in enumerate(valid):
        # interpolate within [t[0], t[-1]]
        y_i = np.interp(grid_t, t, y)
        # if some seed has shorter than t_max_common (shouldn't happen by construction), still safe
        Y[i, :] = y_i

    return grid_t, Y


def mean_std_ignore_nan(vals: np.ndarray) -> Tuple[float, float]:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    mu = float(np.mean(v))
    sd = float(np.std(v, ddof=1) if len(v) > 1 else 0.0)
    return mu, sd


def first_crossing_time(t: np.ndarray, y: np.ndarray, thr: float) -> float:
    if len(t) == 0 or len(y) == 0:
        return float("nan")
    for i in range(len(y)):
        if y[i] >= thr:
            if i == 0:
                return float(t[0])
            t0, y0 = float(t[i - 1]), float(y[i - 1])
            t1, y1 = float(t[i]), float(y[i])
            if y1 == y0:
                return float(t1)
            a = (thr - y0) / (y1 - y0)
            return float(t0 + a * (t1 - t0))
    return float("nan")


def set_icdcs_style():
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 7,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 1.8,
    })


# ----------------------------
# Collect runs
# ----------------------------
@dataclass
class SeedRun:
    t: np.ndarray
    y: np.ndarray


def collect_seed_curves(
    results_root: Path,
    dataset: str,
    alpha: float,
    aggregation: str,
) -> List[SeedRun]:
    """
    Returns list of SeedRun for all seeds found.
    """
    ds_dir = results_root / dataset
    if not ds_dir.exists():
        return []

    # find alpha dir
    alpha_dir = ds_dir / f"alpha_{fmt_alpha(alpha)}"
    if not alpha_dir.exists():
        # fallback: search alpha_* dirs
        alpha_dirs = [p for p in ds_dir.iterdir() if p.is_dir() and p.name.lower().startswith("alpha")]
        alpha_dir = None
        for p in alpha_dirs:
            a = try_parse_alpha_from_dirname(p.name)
            if a is not None and abs(a - alpha) < 1e-9:
                alpha_dir = p
                break
        if alpha_dir is None:
            return []

    agg_dir = alpha_dir / aggregation
    if not agg_dir.exists():
        # allow case variants (softSGD vs softsgd)
        for p in alpha_dir.iterdir():
            if p.is_dir() and p.name.lower() == aggregation.lower():
                agg_dir = p
                break
    if not agg_dir.exists():
        return []

    runs: List[SeedRun] = []
    for seed_dir in list_seed_dirs(agg_dir):
        csv_path = seed_dir / "summary_time_of_round.csv"
        try:
            t, y = load_summary_time_csv(csv_path)
            if len(t) > 1:
                runs.append(SeedRun(t=t, y=y))
        except Exception:
            continue
    return runs


# ----------------------------
# Plotting
# ----------------------------
def add_zoom_inset(ax, t_common, mean_curves, zoom_alpha,
                   inset_box=(0.55, 0.22, 0.42, 0.42),  # (x0, y0, w, h) in AXES coords
                   x_frac=(0.70, 1.00),                 # zoom on last part of time
                   y_pad=0.6):
    """
    inset_box: axes-fraction bbox where the inset lives.
               Increase y0 to move it UP (e.g., 0.22 -> 0.32).
    x_frac:    portion of time range to show in zoom (fraction of max time).
    """

    tmax = float(np.max(t_common))
    x0 = x_frac[0] * tmax
    x1 = x_frac[1] * tmax

    # choose y-limits from curves within zoom window
    mask = (t_common >= x0) & (t_common <= x1)
    ys = []
    for _, y in mean_curves.items():
        if np.any(mask):
            ys.append(y[mask])
    y_min = float(np.nanmin(np.concatenate(ys))) - y_pad
    y_max = float(np.nanmax(np.concatenate(ys))) + y_pad

    # ---- inset position (THIS is what moves it) ----
    axins = inset_axes(
        ax,
        width="100%", height="100%",
        loc="lower left",
        bbox_to_anchor=inset_box,         # (x0,y0,w,h) in axes fraction
        bbox_transform=ax.transAxes,
        borderpad=0.0
    )

    # plot same curves inside inset
    for label, y in mean_curves.items():
        axins.plot(t_common, y, label=label)

    axins.set_xlim(x0, x1)
    axins.set_ylim(y_min, y_max)
    axins.grid(True, alpha=0.25)
    # axins.set_title("Zoom", fontsize=11)

    # optional: make tick labels smaller
    axins.tick_params(axis="both", labelsize=9)

    # draw the rectangle + connectors
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.35")

def _interp_to_grid(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Interpolate y(t) onto t_dst (assumes t_src sorted and increasing)."""
    if len(t_src) < 2 or len(y_src) < 2 or len(t_dst) == 0:
        return np.full_like(t_dst, np.nan, dtype=float)
    # clip destination grid to source bounds to avoid weird extrapolation
    t0, t1 = float(t_src[0]), float(t_src[-1])
    t_clip = np.clip(t_dst, t0, t1)
    return np.interp(t_clip, t_src, y_src)

def plot_dataset_1x3(
    results_root: Path,
    out_dir: Path,
    dataset: str,
    alphas: List[float],
    aggregations: List[str],
    agg_display: Dict[str, str],
    dt_resample: float = 25.0,
    zoom_alphas: Tuple[float, ...] = (1.0, 0.5),
):
    set_icdcs_style()

    fig, axes = plt.subplots(1, 3, figsize=(18, 3.0), sharey=True)
    # fig.suptitle(f"{dataset.upper()} (Time vs Mean Accuracy)", y=1.02, fontsize=16)

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]
        ax.set_title(rf"{dataset.upper()}  ($\alpha$={alpha:g})")
        ax.set_xlabel("Time (s)")
        if idx == 0:
            ax.set_ylabel("Mean accuracy (%)")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, 100)

        # ------------------------------------------------------------
        # 1) Build per-method mean curves on THEIR OWN grids first
        # ------------------------------------------------------------
        per_method = {}  # label -> (t_grid, y_mean)
        for agg in aggregations:
            runs = collect_seed_curves(results_root, dataset, alpha, agg)
            curves = [(r.t, r.y) for r in runs]
            t_grid, Y = resample_to_common_grid(curves, dt=dt_resample)

            if len(t_grid) == 0 or Y.size == 0:
                continue

            y_mean = np.nanmean(Y, axis=0)
            label = agg_display.get(agg.lower(), agg)
            per_method[label] = (t_grid, y_mean)

        if not per_method:
            ax.text(0.5, 0.5, "No data found", ha="center", va="center", transform=ax.transAxes)
            continue

        # ------------------------------------------------------------
        # 2) Create ONE shared time grid across methods (panel-level)
        #    Use the min of max-times so every curve is defined.
        # ------------------------------------------------------------
        t_max_shared = min(float(np.max(tg)) for (tg, _) in per_method.values() if len(tg) > 0)
        if not np.isfinite(t_max_shared) or t_max_shared <= 0:
            ax.text(0.5, 0.5, "Insufficient time range", ha="center", va="center", transform=ax.transAxes)
            continue

        t_common = np.arange(0.0, t_max_shared + 1e-9, float(dt_resample), dtype=float)

        # ------------------------------------------------------------
        # 3) Interpolate each method's mean curve onto shared grid
        # ------------------------------------------------------------
        mean_curves_aligned = {}
        for label, (tg, ym) in per_method.items():
            mean_curves_aligned[label] = _interp_to_grid(tg, ym, t_common)

        # Plot
        for label, y in mean_curves_aligned.items():
            ax.plot(t_common, y, label=label)

        ax.legend(loc="lower right", frameon=True)

        # Zoom inset for close curves (alpha=1.0 and 0.5 by default)
        if any(abs(alpha - za) < 1e-9 for za in zoom_alphas):
            add_zoom_inset(ax, t_common, mean_curves_aligned, zoom_alpha=alpha, inset_box=(0.55, 0.32, 0.42, 0.42))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_time_mean_1x3_zoom.png"
    out_path_pdf = out_dir / f"{dataset}_time_mean_1x3_zoom.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_path.resolve()}")


# ----------------------------
# Tables (merged from your previous table script idea)
# ----------------------------
@dataclass
class RunStats:
    final_acc: float
    t70: float
    t80: float
    t90: float


def pm_fmt(mu: float, sd: float, digits: int = 2) -> str:
    if not np.isfinite(mu):
        return "--"
    if not np.isfinite(sd):
        sd = 0.0
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(mu)}$\\pm${fmt.format(sd)}"


def build_tables_from_summaries(
    results_root: Path,
    datasets: List[str],
    alphas: List[float],
    aggregations: List[str],
    agg_display: Dict[str, str],
    out_txt: Path,
):
    # collect per-seed stats
    data: Dict[str, Dict[float, Dict[str, List[RunStats]]]] = {}

    for ds in datasets:
        data.setdefault(ds, {})
        for a in alphas:
            data[ds].setdefault(a, {})
            for agg in aggregations:
                data[ds][a].setdefault(agg.lower(), [])

                runs = collect_seed_curves(results_root, ds, a, agg)
                for r in runs:
                    final = float(r.y[-1]) if len(r.y) else float("nan")
                    data[ds][a][agg.lower()].append(
                        RunStats(
                            final_acc=final,
                            t70=first_crossing_time(r.t, r.y, 70.0),
                            t80=first_crossing_time(r.t, r.y, 80.0),
                            t90=first_crossing_time(r.t, r.y, 90.0),
                        )
                    )

    # plain text
    def fmt(mu, sd, digits=2):
        if not np.isfinite(mu):
            return "--"
        return f"{mu:.{digits}f} ± {sd:.{digits}f}"

    out = []
    out.append("==== Plain-text tables (quick view) ====\n")

    out.append("FINAL MEAN ACCURACY (mean ± std across seeds)\n")
    out.append(f"{'Algorithm':<12} {'alpha':>5} " + " ".join([f"{ds.upper():>16}" for ds in datasets]))
    out.append("-" * (12 + 1 + 5 + 1 + 17 * len(datasets)))

    for agg in aggregations:
        agg_key = agg.lower()
        name = agg_display.get(agg_key, agg)
        for a in alphas:
            row = f"{name:<12} {a:>5.1f} "
            cells = []
            for ds in datasets:
                runs = data.get(ds, {}).get(a, {}).get(agg_key, [])
                mu, sd = mean_std_ignore_nan(np.array([r.final_acc for r in runs], dtype=float))
                cells.append(f"{fmt(mu, sd, 2):>16}")
            out.append(row + " ".join(cells))
        out.append("")

    out.append("\nTIME TO REACH THRESHOLDS (seconds; mean ± std across seeds)\n")
    out.append(f"{'Dataset':<8} {'Algorithm':<12} {'alpha':>5} {'T@70':>12} {'T@80':>12} {'T@90':>12}")
    out.append("-" * 60)

    for ds in datasets:
        for agg in aggregations:
            agg_key = agg.lower()
            name = agg_display.get(agg_key, agg)
            for a in alphas:
                runs = data.get(ds, {}).get(a, {}).get(agg_key, [])
                mu70, sd70 = mean_std_ignore_nan(np.array([r.t70 for r in runs], dtype=float))
                mu80, sd80 = mean_std_ignore_nan(np.array([r.t80 for r in runs], dtype=float))
                mu90, sd90 = mean_std_ignore_nan(np.array([r.t90 for r in runs], dtype=float))
                out.append(
                    f"{ds.upper():<8} {name:<12} {a:>5.1f} "
                    f"{fmt(mu70, sd70, 0):>12} {fmt(mu80, sd80, 0):>12} {fmt(mu90, sd90, 0):>12}"
                )
        out.append("")

    # LaTeX tables
    out.append("\n==== LaTeX tables (paste into ICDCS template) ====\n")

    # LaTeX: final accuracy
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\caption{Final mean accuracy (\%) across seeds (mean$\pm$std).}")
    out.append(r"\label{tab:final_acc}")
    out.append(r"\small")
    out.append(r"\setlength{\tabcolsep}{6pt}")
    out.append(r"\renewcommand{\arraystretch}{1.15}")
    colspec = "l c " + " ".join(["c" for _ in datasets])
    out.append(r"\begin{tabular}{" + colspec + r"}")
    out.append(r"\toprule")
    header = ["\\textbf{Algorithm}", r"\textbf{$\alpha$}"] + [f"\\textbf{{{ds.upper()}}}" for ds in datasets]
    out.append(" & ".join(header) + r" \\")
    out.append(r"\midrule")
    for agg in aggregations:
        agg_key = agg.lower()
        name = agg_display.get(agg_key, agg)
        for a in alphas:
            row = [name, f"{a:.1f}"]
            for ds in datasets:
                runs = data.get(ds, {}).get(a, {}).get(agg_key, [])
                mu, sd = mean_std_ignore_nan(np.array([r.final_acc for r in runs], dtype=float))
                row.append(pm_fmt(mu, sd, digits=2))
            out.append(" & ".join(row) + r" \\")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    out.append("")

    # LaTeX: time to thresholds
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\caption{Time to reach target accuracy (seconds; mean$\pm$std across seeds).}")
    out.append(r"\label{tab:time_to_thr}")
    out.append(r"\small")
    out.append(r"\setlength{\tabcolsep}{5pt}")
    out.append(r"\renewcommand{\arraystretch}{1.15}")
    out.append(r"\begin{tabular}{l l c c c c}")
    out.append(r"\toprule")
    out.append(r"\textbf{Dataset} & \textbf{Algorithm} & \textbf{$\alpha$} & \textbf{T@70} & \textbf{T@80} & \textbf{T@90} \\")
    out.append(r"\midrule")
    for ds in datasets:
        first_ds = True
        for agg in aggregations:
            agg_key = agg.lower()
            name = agg_display.get(agg_key, agg)
            for a in alphas:
                runs = data.get(ds, {}).get(a, {}).get(agg_key, [])
                mu70, sd70 = mean_std_ignore_nan(np.array([r.t70 for r in runs], dtype=float))
                mu80, sd80 = mean_std_ignore_nan(np.array([r.t80 for r in runs], dtype=float))
                mu90, sd90 = mean_std_ignore_nan(np.array([r.t90 for r in runs], dtype=float))
                row = [
                    ds.upper() if first_ds else "",
                    name,
                    f"{a:.1f}",
                    pm_fmt(mu70, sd70, digits=0),
                    pm_fmt(mu80, sd80, digits=0),
                    pm_fmt(mu90, sd90, digits=0),
                ]
                out.append(" & ".join(row) + r" \\")
                first_ds = False
            out.append(r"\midrule")
        out.append(r"\midrule")
    out[-1] = r"\bottomrule"
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(out) + "\n")
    print(f"[OK] Wrote tables: {out_txt.resolve()}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=str, default="all_exp_results")
    ap.add_argument("--datasets", type=str, default="mnist,fmnist")
    ap.add_argument("--alphas", type=str, default="1.0,0.5,0.1")
    ap.add_argument("--aggregations", type=str, default="fedavg,softSGD,dflaa")

    ap.add_argument("--out-dir", type=str, default="all_exp_results/_icdcs")
    ap.add_argument("--tables-out", type=str, default="ICDCS_tables.txt")

    ap.add_argument("--dt-resample", type=float, default=25.0, help="Resample step (seconds) for seed aggregation.")
    ap.add_argument("--zoom-alphas", type=str, default="1.0,0.5", help="Which alphas get zoom inset.")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    aggregations = [a.strip() for a in args.aggregations.split(",") if a.strip()]
    zoom_alphas = tuple(float(x.strip()) for x in args.zoom_alphas.split(",") if x.strip())

    # stable alpha order (you requested 1.0,0.5,0.1)
    # if user passes different, keep their order
    agg_display = {
        "fedavg": "FedAvg",
        "softsgd": "Soft-DSGD",
        "dflaa": "DFL-AA",
    }
    # preserve user-provided capitalization in folder names but use display mapping
    agg_display_final = {a.lower(): agg_display.get(a.lower(), a) for a in aggregations}

    # figures
    for ds in datasets:
        plot_dataset_1x3(
            results_root=results_root,
            out_dir=out_dir,
            dataset=ds,
            alphas=alphas,
            aggregations=aggregations,
            agg_display=agg_display_final,
            dt_resample=float(args.dt_resample),
            zoom_alphas=zoom_alphas,
        )

    # tables
    tables_out = out_dir / args.tables_out
    build_tables_from_summaries(
        results_root=results_root,
        datasets=datasets,
        alphas=alphas,
        aggregations=aggregations,
        agg_display=agg_display_final,
        out_txt=tables_out,
    )


if __name__ == "__main__":
    main()
