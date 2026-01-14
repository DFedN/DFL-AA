#!/usr/bin/env python3
"""
main_results_comp_abl.py

Goal:
  ICDCS-style comparison plots over TIME for multiple alphas.
  Each subplot = one alpha, curves = aggregations.
  Mean across seeds.
  IMPORTANT: x-axis is COMMON TIME GRID across ALL curves in that subplot
             (across aggs AND seeds) so lines align and end at same x.

Expected layout:
  {root}/{dataset}/alpha_{alpha_tag}/{aggregation}/seed_{seed}/summary_time_of_round.csv

Example:
  ablation_results/fmnist/alpha_0p1/dflaa/seed_42/summary_time_of_round.csv

Run:
  python main_results_comp_abl.py \
    --root ablation_results \
    --dataset mnist \
    --alpha 0.1,0.5 \
    --aggregations dflaa,dflaa_s,dflaa_c,softSGD,softGSD_c \
    --out-dir icdcs_paper_results
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def fmt_alpha_dir(a: float) -> str:
    """0.5 -> 0p5, 0.1 -> 0p1"""
    s = f"{a:.3f}".rstrip("0").rstrip(".")
    if "." not in s:
        s = s + ".0"
    return s.replace(".", "p")


def parse_csv_floats_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def find_case_insensitive_dir(parent: Path, name: str) -> Optional[Path]:
    """Return child directory matching name ignoring case, else None."""
    if not parent.exists():
        return None
    p = parent / name
    if p.exists() and p.is_dir():
        return p
    target = name.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child
    return None


def list_seed_dirs(agg_dir: Path) -> List[Path]:
    if not agg_dir or not agg_dir.exists():
        return []
    out = []
    for p in agg_dir.iterdir():
        if p.is_dir() and (p / "summary_time_of_round.csv").exists():
            out.append(p)
    return sorted(out)


def load_summary_time_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads time_s and mean (accuracy) from summary_time_of_round.csv
    Columns accepted (case-insensitive):
      time: time_s / time / t
      mean: mean / avg / accuracy / acc
    """
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    cols = {c.lower(): c for c in arr.dtype.names}

    time_col = cols.get("time_s") or cols.get("time") or cols.get("t")
    mean_col = cols.get("mean") or cols.get("avg") or cols.get("accuracy") or cols.get("acc")

    if time_col is None or mean_col is None:
        raise ValueError(f"Missing columns in {path}. Found: {arr.dtype.names}")

    t = np.asarray(arr[time_col], dtype=float)
    y = np.asarray(arr[mean_col], dtype=float)

    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if len(t) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # sort + enforce monotonic time
    order = np.argsort(t)
    t, y = t[order], y[order]
    t = np.maximum.accumulate(t)

    # shift to start at 0 (each seed's local time axis)
    t = t - float(t[0])

    # if y in [0,1], convert to %
    if np.nanmax(y) <= 1.5:
        y = y * 100.0

    return t, y


def collect_seed_curves(root: Path, dataset: str, alpha: float, aggregation: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    alpha_dir = root / dataset / f"alpha_{fmt_alpha_dir(alpha)}"
    agg_dir = find_case_insensitive_dir(alpha_dir, aggregation)
    if agg_dir is None:
        return []

    curves = []
    for seed_dir in list_seed_dirs(agg_dir):
        csv_path = seed_dir / "summary_time_of_round.csv"
        try:
            t, y = load_summary_time_csv(csv_path)
            if len(t) > 1:
                curves.append((t, y))
        except Exception:
            # ignore broken / incomplete files silently (keeps script robust)
            pass
    return curves


def set_icdcs_single_column_style():
    # IEEE single-column is ~3.45in wide, so 2 columns ~6.9in
    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "lines.linewidth": 1.6,
    })


def style_for_agg(agg: str) -> Tuple[str, str, str]:
    """
    Returns: (color, linestyle, label)
    Ensure EVERY curve is distinguishable.
    """
    k = agg.strip().lower()

    # normalize common variants
    k = k.replace("softdsgd", "softsgd")

    style_map = {
        # DFL-AA family (greens, different line styles)
        "dflaa":    ("tab:green", "-",  "DFL-AA"),
        "dflaa_s":  ("tab:green", "--", "DFL-AA (w/o AoI decay)"),
        "dflaa_c":  ("tab:green", ":",  "DFL-AA (completeness-only)"),

        # Soft-DSGD family (oranges, different line styles)
        "softsgd":    ("tab:orange", "-",  "Soft-DSGD"),
        "softsgd_s":  ("tab:orange", "--", "Soft-DSGD (w/ AoI decay)"),
        "softgsd_c":  ("tab:orange", ":",  "Soft-DSGD (completeness-only)"),
    }

    if k in style_map:
        return style_map[k]
    # fallback if you pass a new name
    return ("tab:blue", "-", agg)


def compute_common_horizon_per_alpha(
    curves_by_agg: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]
) -> float:
    """
    For a given alpha:
      - For each aggregation, compute overlap horizon across seeds: agg_tmax = min(seed_max_t)
      - Then compute alpha_tmax = min(agg_tmax across aggregations)

    This guarantees:
      - no extrapolation for any seed
      - ALL curves in this subplot share the SAME x-range
    """
    agg_horizons = []
    for agg, curves in curves_by_agg.items():
        valid = [(t, y) for (t, y) in curves if len(t) > 1 and len(y) > 1]
        if not valid:
            continue
        # overlap across seeds for this agg
        agg_tmax = min(float(np.max(t)) for (t, _) in valid)
        if np.isfinite(agg_tmax) and agg_tmax > 0:
            agg_horizons.append(agg_tmax)

    if not agg_horizons:
        return 0.0
    return float(min(agg_horizons))


def mean_curve_on_grid(
    curves: List[Tuple[np.ndarray, np.ndarray]],
    grid_t: np.ndarray
) -> Optional[np.ndarray]:
    """
    Interpolate each seed curve onto grid_t and return mean across seeds.
    Assumes grid_t is within each seed's time domain (no extrapolation needed).
    """
    valid = [(t, y) for (t, y) in curves if len(t) > 1 and len(y) > 1]
    if not valid:
        return None

    Y = np.full((len(valid), len(grid_t)), np.nan, dtype=float)
    for i, (t, y) in enumerate(valid):
        # grid_t guaranteed <= max(t) by our horizon design
        Y[i, :] = np.interp(grid_t, t, y)

    return np.nanmean(Y, axis=0)


def plot_components_1xN(
    root: Path,
    out_dir: Path,
    dataset: str,
    alphas: List[float],
    aggregations: List[str],
    dt_resample: float = 25.0,
):
    set_icdcs_single_column_style()

    n = len(alphas)
    if n < 1:
        raise ValueError("Need at least one alpha")

    # width ~3.45in per column (IEEE single-column width)
    fig_w = 3.45 * n
    fig_h = 2.25
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), sharey=True)
    if n == 1:
        axes = [axes]

    # Reserve TOP space for legend (prevents legend/title overlap)
    fig.subplots_adjust(
        left=0.08, right=0.995,
        bottom=0.18, top=0.80,
        wspace=0.16
    )

    handles_for_legend: Dict[str, any] = {}
    any_data_any_subplot = False

    for ax, alpha in zip(axes, alphas):
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Time (s)")
        ax.set_title(rf"{dataset.upper()} ($\alpha$={alpha:g})", pad=2)

        # collect curves for this alpha
        curves_by_agg: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        for agg in aggregations:
            curves_by_agg[agg] = collect_seed_curves(root, dataset, alpha, agg)

        # compute common horizon (shared x for ALL curves in this subplot)
        tmax_common = compute_common_horizon_per_alpha(curves_by_agg)
        if tmax_common <= 0:
            ax.text(0.5, 0.5, "No data found", ha="center", va="center", transform=ax.transAxes)
            continue

        grid_t = np.arange(0.0, tmax_common + 1e-9, dt_resample, dtype=float)
        ax.set_xlim(0.0, float(grid_t[-1]))

        plotted_any = False
        for agg in aggregations:
            curves = curves_by_agg.get(agg, [])
            y_mean = mean_curve_on_grid(curves, grid_t)
            if y_mean is None:
                continue

            color, ls, label = style_for_agg(agg)
            line, = ax.plot(grid_t, y_mean, color=color, linestyle=ls, label=label)
            plotted_any = True
            any_data_any_subplot = True

            # collect legend handles once (by label)
            if label not in handles_for_legend:
                handles_for_legend[label] = line

        if not plotted_any:
            ax.text(0.5, 0.5, "No data found", ha="center", va="center", transform=ax.transAxes)

    axes[0].set_ylabel("Accuracy (%)")

    # Figure-level legend in reserved top strip (no overlap)
    if handles_for_legend and any_data_any_subplot:
        fig.legend(
            list(handles_for_legend.values()),
            list(handles_for_legend.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(4, len(handles_for_legend)),
            frameon=False,
            columnspacing=1.2,
            handlelength=2.2,
            borderaxespad=0.0,
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_tag = "_".join(fmt_alpha_dir(a) for a in alphas)
    stem = f"{dataset}_alphas_{alpha_tag}_components_1x{n}"
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {pdf_path}")
    print(f"[OK] Saved: {png_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="ablation_results")
    ap.add_argument("--dataset", type=str, default="fmnist")
    ap.add_argument("--alpha", type=str, default="0.1", help="comma-separated, e.g., 0.1,0.5")
    ap.add_argument("--aggregations", type=str, default="dflaa,dflaa_s,softSGD,softSGD_s")
    ap.add_argument("--dt-resample", type=float, default=25.0)
    ap.add_argument("--out-dir", type=str, default="icdcs_paper_results")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    dataset = args.dataset.strip().lower()
    alphas = parse_csv_floats_list(args.alpha)
    aggs = [x.strip() for x in args.aggregations.split(",") if x.strip()]

    plot_components_1xN(
        root=root,
        out_dir=out_dir,
        dataset=dataset,
        alphas=alphas,
        aggregations=aggs,
        dt_resample=float(args.dt_resample),
    )


if __name__ == "__main__":
    main()
