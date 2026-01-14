#!/usr/bin/env python3
"""
main_results_loss_abl.py

Expected layout:
  {root}/loss_{loss}/{dataset}/alpha_{alpha_tag}/{aggregation}/seed_{seed}/summary_time_of_round.csv

Example:
  ablation_results_loss/loss_6.5/mnist/alpha_0p5/softSGD/seed_42/summary_time_of_round.csv

Produces:
  out_dir/{dataset}_loss_ablation_2x2.pdf  (and .png)


python main_results_loss_abl.py \
  --root ablation_results_loss \
  --dataset mnist \
  --losses 6.5,4.5 \
  --alphas 0.5,0.1 \
  --aggregations dflaa,softSGD,fedavg \
  --out-dir icdcs_paper_results


"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
def fmt_alpha_dir(a: float) -> str:
    """0.5 -> 0p5, 0.1 -> 0p1"""
    s = f"{a:.3f}".rstrip("0").rstrip(".")
    if "." not in s:
        s = s + ".0"
    return s.replace(".", "p")


def list_seed_dirs(agg_dir: Path) -> List[Path]:
    if not agg_dir.exists():
        return []
    out = []
    for p in agg_dir.iterdir():
        if p.is_dir() and (p / "summary_time_of_round.csv").exists():
            out.append(p)
    return sorted(out)


def load_summary_time_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Reads time_s and mean from summary_time_of_round.csv"""
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
    t = t - float(t[0])  # normalize start at 0

    # auto-scale if y in [0,1]
    if np.nanmax(y) <= 1.5:
        y = y * 100.0

    return t, y


def resample_to_common_grid(curves: List[Tuple[np.ndarray, np.ndarray]], dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    curves: list of (t,y) per seed
    Returns:
      grid_t, Y  where Y shape=(num_seeds, len(grid_t))
    """
    valid = [(t, y) for (t, y) in curves if len(t) > 1 and len(y) > 1]
    if not valid:
        return np.array([], dtype=float), np.empty((0, 0), dtype=float)

    t_max_common = min(float(np.max(t)) for (t, _) in valid)
    if not np.isfinite(t_max_common) or t_max_common <= 0:
        return np.array([], dtype=float), np.empty((0, 0), dtype=float)

    grid_t = np.arange(0.0, t_max_common + 1e-9, dt, dtype=float)

    Y = np.full((len(valid), len(grid_t)), np.nan, dtype=float)
    for i, (t, y) in enumerate(valid):
        Y[i, :] = np.interp(grid_t, t, y)

    return grid_t, Y


def collect_seed_curves(root: Path, loss: float, dataset: str, alpha: float, aggregation: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Reads all seeds for a given (loss, dataset, alpha, aggregation)
    """
    loss_dir = root / f"loss_{loss:g}" / dataset / f"alpha_{fmt_alpha_dir(alpha)}" / aggregation
    curves = []
    for seed_dir in list_seed_dirs(loss_dir):
        csv_path = seed_dir / "summary_time_of_round.csv"
        try:
            t, y = load_summary_time_csv(csv_path)
            if len(t) > 1:
                curves.append((t, y))
        except Exception:
            pass
    return curves


def set_icdcs_single_column_style():
    # Single-column IEEE/ICDCS-ish: ~3.5 inches wide
    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "lines.linewidth": 1.3,
    })


# ----------------------------
# Plot
# ----------------------------
def plot_2x2_loss_alpha(
    root: Path,
    out_dir: Path,
    dataset: str,
    losses: List[float],
    alphas: List[float],
    aggregations: List[str],
    dt_resample: float = 25.0,
):
    assert len(losses) == 2 and len(alphas) == 2, "This script expects exactly 2 losses and 2 alphas for a 2x2 plot."

    set_icdcs_single_column_style()

    # Colors as requested
    color_map = {
        "fedavg": "tab:blue",
        "softsgd": "tab:orange",
        "dflaa": "tab:green",
    }
    label_map = {
        "fedavg": "FedAvg",
        "softsgd": "softSGD",
        "dflaa": "DFL-AA",
    }

    # figure size for single-column fit
    fig, axes = plt.subplots(
        2, 2,
        figsize=(3.45, 2.9),   # tweak if needed
        sharex="col",
        sharey=True,
    )

    # precompute so we can add one global legend
    legend_handles = None
    legend_labels = None

    for r, loss in enumerate(losses):
        for c, alpha in enumerate(alphas):
            ax = axes[r, c]
            ax.grid(True, alpha=0.25)
            ax.set_ylim(0, 100)

            # panel title (compact)
            actual_loss = [37.8, 26.8]
            ax.set_title(rf"$\alpha$={alpha:g}, loss={actual_loss[r]:g}%")

            # plot each aggregation
            plotted_any = False
            for agg in aggregations:
                key = agg.lower()
                curves = collect_seed_curves(root, loss, dataset, alpha, agg)
                if not curves:
                    continue

                t_grid, Y = resample_to_common_grid(curves, dt=dt_resample)
                if len(t_grid) == 0 or Y.size == 0:
                    continue

                y_mean = np.nanmean(Y, axis=0)

                h = ax.plot(
                    t_grid, y_mean,
                    label=label_map.get(key, agg),
                    color=color_map.get(key, None),
                )[0]
                plotted_any = True

            if not plotted_any:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

            # axis labels (only where needed)
            if c == 0:
                ax.set_ylabel("Accuracy (%)")
            if r == 1:
                ax.set_xlabel("Time (s)")

            # store legend from first non-empty axis
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    # Global legend (top, small, saves subplot space)
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.tight_layout(pad=0.5, rect=[0, 0, 1, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{dataset}_loss_ablation_2x2.pdf"
    png_path = out_dir / f"{dataset}_loss_ablation_2x2.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {pdf_path}")
    print(f"[OK] Saved: {png_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="ablation_results_loss")
    ap.add_argument("--dataset", type=str, default="mnist")
    ap.add_argument("--losses", type=str, default="6.5,4.5")
    ap.add_argument("--alphas", type=str, default="0.5,0.1")
    ap.add_argument("--aggregations", type=str, default="dflaa,softSGD,fedavg")
    ap.add_argument("--dt-resample", type=float, default=25.0)
    ap.add_argument("--out-dir", type=str, default="icdcs_paper_results")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)

    losses = [float(x.strip()) for x in args.losses.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    aggs = [x.strip() for x in args.aggregations.split(",") if x.strip()]

    plot_2x2_loss_alpha(
        root=root,
        out_dir=out_dir,
        dataset=args.dataset.strip().lower(),
        losses=losses,
        alphas=alphas,
        aggregations=aggs,
        dt_resample=float(args.dt_resample),
    )


if __name__ == "__main__":
    main()
