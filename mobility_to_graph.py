#!/usr/bin/env python3
"""
mobility_to_graph.py
====================

Create a GIF showing proximity-based connectivity over time from a mobility CSV
(e.g., Gauss–Markov output).

Accepts BOTH time column names:
  - time_sec  (preferred)
  - t         (what your Gauss-Markov script likely uses)

Required columns (aliases supported):
  - time:     time_sec or t or time or timestamp
  - node id:  node_id or id or node
  - position: x_m/y_m  (or x/y)

Example:
  python mobility_to_graph.py \
    --csv_path mobility.csv \
    --radius_m 500 \
    --area_m 2000 \
    --fps 10 \
    --out_gif network.gif
"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import imageio.v2 as imageio


# ----------------------------- Column normalization -----------------------------

def _pick_col(df: pd.DataFrame, candidates: List[str], what: str) -> str:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"CSV missing required '{what}' column. Tried: {candidates}. Found: {list(df.columns)}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    time_col = _pick_col(df, ["time_sec", "t", "time", "timestamp", "ts"], "time")
    node_col = _pick_col(df, ["node_id", "id", "node", "nid"], "node_id")
    x_col = _pick_col(df, ["x_m", "x"], "x")
    y_col = _pick_col(df, ["y_m", "y"], "y")

    out = df.rename(columns={
        time_col: "time_sec",
        node_col: "node_id",
        x_col: "x_m",
        y_col: "y_m",
    }).copy()

    out["time_sec"] = pd.to_numeric(out["time_sec"], errors="raise").astype(int)
    out["node_id"] = pd.to_numeric(out["node_id"], errors="raise").astype(int)
    out["x_m"] = pd.to_numeric(out["x_m"], errors="raise").astype(float)
    out["y_m"] = pd.to_numeric(out["y_m"], errors="raise").astype(float)

    return out[["time_sec", "node_id", "x_m", "y_m"]]


# ----------------------------- Graph building per frame -----------------------------

def edges_within_radius(pos: np.ndarray, radius: float) -> Tuple[np.ndarray, float]:
    """
    pos: [N,2]
    Returns:
      edges: [E,2] int pairs (i,j) with i<j
      avg_degree
    """
    n = pos.shape[0]
    if n <= 1:
        return np.zeros((0, 2), dtype=int), 0.0

    # pairwise squared distances
    diff = pos[:, None, :] - pos[None, :, :]
    d2 = (diff ** 2).sum(axis=2)

    r2 = radius * radius
    adj = (d2 <= r2) & (~np.eye(n, dtype=bool))

    # upper triangle edges
    iu, ju = np.triu_indices(n, k=1)
    mask = adj[iu, ju]
    edges = np.stack([iu[mask], ju[mask]], axis=1).astype(int)

    E = edges.shape[0]
    avg_deg = (2.0 * E) / float(n)
    return edges, avg_deg


def fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    """
    Robust conversion for modern matplotlib (no tostring_rgb dependency).
    Returns RGB uint8 image [H,W,3].
    """
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)  # [H,W,4]
    rgb = rgba[..., :3].copy()
    return rgb


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--radius_m", type=float, required=True)
    ap.add_argument("--area_m", type=float, default=None,
                    help="If set, axes will be [0, area_m] x [0, area_m]. If omitted, inferred from data.")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=1, help="Use every k-th timestamp (1 = all).")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = no limit")
    ap.add_argument("--out_gif", type=str, required=True)

    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--figsize", type=float, default=6.5, help="Figure size in inches (square).")
    ap.add_argument("--label_nodes", action="store_true", help="Draw node id labels (ok for small N).")
    ap.add_argument("--node_size", type=float, default=30.0)
    ap.add_argument("--edge_lw", type=float, default=0.8)
    ap.add_argument("--edge_alpha", type=float, default=0.35)

    args = ap.parse_args()

    print(f"[INFO] Loaded: {args.csv_path}")
    df_raw = pd.read_csv(args.csv_path)
    df = normalize_columns(df_raw)

    # sort + basic stats
    df = df.sort_values(["time_sec", "node_id"]).reset_index(drop=True)
    times = np.array(sorted(df["time_sec"].unique()), dtype=int)
    node_ids = np.array(sorted(df["node_id"].unique()), dtype=int)
    N = len(node_ids)

    # Map node_id -> 0..N-1
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    df["_idx"] = df["node_id"].map(id_to_idx).astype(int)

    # infer bounds if needed
    if args.area_m is not None:
        xlim = (0.0, float(args.area_m))
        ylim = (0.0, float(args.area_m))
    else:
        xmin, xmax = float(df["x_m"].min()), float(df["x_m"].max())
        ymin, ymax = float(df["y_m"].min()), float(df["y_m"].max())
        pad_x = 0.05 * max(1.0, (xmax - xmin))
        pad_y = 0.05 * max(1.0, (ymax - ymin))
        xlim = (xmin - pad_x, xmax + pad_x)
        ylim = (ymin - pad_y, ymax + pad_y)

    # build a dense [T,N,2] position array (fast for frame loop)
    T = len(times)
    pos = np.full((T, N, 2), np.nan, dtype=float)

    # fill positions
    # For each time, we expect all N nodes. If missing, we keep NaN (node not shown).
    grouped = df.groupby("time_sec")
    t_to_row = {t: i for i, t in enumerate(times)}
    for t, g in grouped:
        r = t_to_row[int(t)]
        idx = g["_idx"].to_numpy()
        pos[r, idx, 0] = g["x_m"].to_numpy()
        pos[r, idx, 1] = g["y_m"].to_numpy()

    # frames selection
    sel = np.arange(0, T, max(1, args.stride), dtype=int)
    if args.max_frames and args.max_frames > 0:
        sel = sel[:args.max_frames]

    print(f"[INFO] Nodes: {N} | Frames: {len(sel)} | radius={args.radius_m:.1f}m")
    print(f"[INFO] Writing GIF: {args.out_gif} (fps={args.fps}, stride={args.stride})")

    fig = plt.figure(figsize=(args.figsize, args.figsize), dpi=args.dpi)
    ax = fig.add_subplot(111)

    with imageio.get_writer(args.out_gif, mode="I", fps=float(args.fps)) as writer:
        for fi, ti in enumerate(sel):
            tsec = int(times[ti])
            p = pos[ti]  # [N,2]

            # drop nodes with NaN (if any)
            valid = np.isfinite(p[:, 0]) & np.isfinite(p[:, 1])
            p_valid = p[valid]
            idx_valid = np.where(valid)[0]

            edges, avg_deg = edges_within_radius(p_valid, float(args.radius_m))
            E = edges.shape[0]

            ax.clear()
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")

            # edges as LineCollection
            if E > 0:
                segs = np.stack([p_valid[edges[:, 0]], p_valid[edges[:, 1]]], axis=1)  # [E,2,2]
                lc = LineCollection(segs, linewidths=args.edge_lw, alpha=args.edge_alpha)
                ax.add_collection(lc)

            # nodes
            ax.scatter(p_valid[:, 0], p_valid[:, 1], s=args.node_size)

            if args.label_nodes and N <= 80:
                for local_i, global_i in enumerate(idx_valid):
                    ax.text(p_valid[local_i, 0], p_valid[local_i, 1],
                            str(int(node_ids[global_i])), fontsize=7, ha="left", va="bottom")

            ax.set_title(f"t={tsec}s | edges={E} | avg_deg={avg_deg:.2f}")

            frame = fig_to_rgb(fig)
            writer.append_data(frame)

            if (fi + 1) % 50 == 0:
                print(f"[INFO] Rendered {fi+1}/{len(sel)} frames...")

    plt.close(fig)
    print(f"[DONE] Wrote GIF: {args.out_gif}")


if __name__ == "__main__":
    main()
