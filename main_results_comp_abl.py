"""
main_results_comp_abl.py

Generate figures from ablation results using summary of each experiment

Expected layout:
  {root}/{dataset}/alpha_{alpha_tag}/{aggregation}/seed_{seed}/summary_time_of_round.csv


Usage:
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

    order = np.argsort(t)
    t, y = t[order], y[order]
    t = np.maximum.accumulate(t)

    t = t - float(t[0])

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
            pass
    return curves


def set_icdcs_single_column_style():
    plt.rcParams.update({
        # ---- Font family ----
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",  
        "mathtext.rm": "STIXGeneral",
        "mathtext.it": "STIXGeneral:italic",
        "mathtext.bf": "STIXGeneral:bold",

        # ---- Embed fonts nicely in vector outputs ----
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",  

        # ---- Sizes ----
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 12,

        # ---- Lines, ticks, layout ----
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,

        # Clean export defaults
        "savefig.dpi": 300,  
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def style_for_agg(agg: str) -> Tuple[str, str, str]:
    """
    Returns: (color, linestyle, label)
    Ensure EVERY curve is distinguishable.
    """
    k = agg.strip().lower()

    k = k.replace("softdsgd", "softsgd")

    style_map = {
        # DFL-AA family (greens, different line styles)
        "dflaa":    ("tab:green", "-",  "DFL-AA"),
        "dflaa_s":  ("tab:green", "--", "DFL-AA (w/o AoI)"),
        "dflaa_c":  ("tab:green", ":",  "DFL-AA (o comp)"),

        # Soft-DSGD family (oranges, different line styles)
        "softsgd":    ("tab:orange", "-",  "Soft-DSGD"),
        "softsgd_s":  ("tab:orange", "--", "Soft-DSGD (w AoI)"),
        "softsgd_c":  ("tab:orange", ":",  "Soft-DSGD (w comp)"),
    }

    if k in style_map:
        return style_map[k]
    return ("tab:blue", "-", agg)


def compute_common_horizon_per_alpha(
    curves_by_agg: Dict[str, List[Tuple[np.ndarray, np.ndarray]]]
) -> float:
    
    agg_horizons = []
    for agg, curves in curves_by_agg.items():
        valid = [(t, y) for (t, y) in curves if len(t) > 1 and len(y) > 1]
        if not valid:
            continue
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

    fig_w = 5 * n
    fig_h = 3
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), sharey=True)
    if n == 1:
        axes = [axes]

    fig.subplots_adjust(
        left=0.08, right=0.995,
        bottom=0.18, top=0.80,
        wspace=0.16
    )

    handles_for_legend: Dict[str, any] = {}
    any_data_any_subplot = False

    for ax, alpha in zip(axes, alphas):
        ax.grid(True, alpha=0.25)
        ax.set_ylim(30, 100)
        ax.set_xlabel("Time (s)")
        ax.set_title(rf"{dataset.upper()} (Dirichlet: {alpha:g})", pad=2)

        curves_by_agg: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        for agg in aggregations:
            curves_by_agg[agg] = collect_seed_curves(root, dataset, alpha, agg)

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

            if label not in handles_for_legend:
                handles_for_legend[label] = line

        if not plotted_any:
            ax.text(0.5, 0.5, "No data found", ha="center", va="center", transform=ax.transAxes)

    axes[0].set_ylabel("Accuracy (%)")

    if handles_for_legend and any_data_any_subplot:
        fig.legend(
            list(handles_for_legend.values()),
            list(handles_for_legend.keys()),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(5, len(handles_for_legend)),
            frameon=False,
            columnspacing=1.2,
            handlelength=2.8,
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
