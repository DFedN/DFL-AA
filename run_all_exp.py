#!/usr/bin/env python3
"""
run_all_exp.py
============

Running point for all DFL Simulation experiments for given algo on given dataset

Usage: For all experiments

  python run_all_exp.py \
    --partitions-root dataset_partitions \
    --partition-pattern "{root}/{dataset}/alpha_{alpha_tag}" \
    --datasets mnist,fmnist,cifar10 \
    --alphas 0.1,0.5,1.0 \
    --aggregations fedavg,softSGD,dflaa
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import ray

from dfl_core import DFedNode 


# ----------------------------
# Partitions loader
# ----------------------------
def load_saved_partition(partition_dir: Path):
    partition_path = Path(partition_dir)
    with open(partition_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    client_data = []
    for node_id in sorted(
        metadata["partitions"].keys(),
        key=lambda x: int(x.split("_")[-1]) if "_" in x else int(x),
    ):
        files = metadata["partitions"][node_id]
        data = np.load(partition_path / files["data_file"])
        labels = np.load(partition_path / files["labels_file"])
        client_data.append({"data": data, "labels": labels})

    test_data = np.load(partition_path / metadata["test_data_file"])
    test_labels = np.load(partition_path / metadata["test_labels_file"])
    return client_data, test_data, test_labels


# ----------------------------
# Styling
# ----------------------------
def set_icdcs_style():
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "legend.fontsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 3.0,
        "figure.figsize": (10.5, 6.5),
    })


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------
# Metrics helpers
# ----------------------------
def _parse_log_round_time_val(log) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not log:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)
    r = np.array([int(x[0]) for x in log], dtype=int)
    t = np.array([float(x[1]) for x in log], dtype=float)
    v = np.array([float(x[2]) for x in log], dtype=float)
    order = np.argsort(r)
    return r[order], t[order], v[order]


def build_matrix_by_common_round(per_node: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]):
    max_rounds = []
    for _, (r, _, _) in per_node.items():
        if len(r) > 0:
            max_rounds.append(int(r.max()))
    if not max_rounds:
        raise RuntimeError("No node produced any accuracy logs.")

    R_common = int(min(max_rounds))
    node_ids = sorted(per_node.keys())
    N = len(node_ids)

    M = np.full((N, R_common), np.nan, dtype=float)
    for i, nid in enumerate(node_ids):
        r, _, a = per_node[nid]
        mask = (r >= 1) & (r <= R_common)
        rr = r[mask]
        aa = a[mask]
        for rr_i, aa_i in zip(rr.tolist(), aa.tolist()):
            M[i, rr_i - 1] = aa_i

    rounds = np.arange(1, R_common + 1, dtype=int)
    return rounds, M, R_common


def summary_stats(M: np.ndarray):
    mean = np.nanmean(M, axis=0)
    p10 = np.nanpercentile(M, 10, axis=0)
    p50 = np.nanpercentile(M, 50, axis=0)
    p90 = np.nanpercentile(M, 90, axis=0)
    return mean, p10, p50, p90


def time_of_round_axis(
    per_node: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    starts_epoch: Dict[int, float],
    rounds: np.ndarray,
):
    global_t0 = float(min(starts_epoch.values()))
    x = np.zeros_like(rounds, dtype=float)
    for j, rj in enumerate(rounds.tolist()):
        times_abs = []
        for nid, (r, t, _) in per_node.items():
            idx = np.where(r == rj)[0]
            if len(idx) == 0:
                continue
            tj = float(t[idx[-1]])
            times_abs.append(float(starts_epoch[nid]) + tj)
        x[j] = float(np.median(times_abs) - global_t0) if times_abs else np.nan
    x = np.maximum.accumulate(np.nan_to_num(x, nan=0.0))
    return x


def auc_trapz(y: np.ndarray, x: Optional[np.ndarray] = None) -> float:
    if len(y) == 0:
        return float("nan")
    if x is None:
        return float(np.trapezoid(y, dx=1.0))
    return float(np.trapezoid(y, x=x))


def first_crossing_time(x: np.ndarray, y: np.ndarray, thr: float) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    for i in range(len(y)):
        if y[i] >= thr:
            if i == 0:
                return float(x[0])
            x0, y0 = float(x[i - 1]), float(y[i - 1])
            x1, y1 = float(x[i]), float(y[i])
            if y1 == y0:
                return float(x1)
            alpha = (thr - y0) / (y1 - y0)
            return float(x0 + alpha * (x1 - x0))
    return float("nan")


def jain_index(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan")
    s = vals.sum()
    ss = (vals ** 2).sum()
    if ss <= 0:
        return float("nan")
    return float((s * s) / (len(vals) * ss))


def gini_coefficient(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan")
    if np.all(vals == 0):
        return 0.0
    v = np.sort(vals)
    n = len(v)
    cum = np.cumsum(v)
    return float(1.0 + 1.0 / n - 2.0 * np.sum(cum) / (n * cum[-1]))


def cvar_worst_alpha(errors: np.ndarray, alpha: float = 0.1) -> float:
    e = np.asarray(errors, dtype=float)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return float("nan")
    k = max(1, int(np.ceil(alpha * len(e))))
    worst = np.sort(e)[-k:]
    return float(np.mean(worst))


def network_metrics_from_histories(net_histories: Dict[int, dict]) -> Dict[str, float]:
    latencies = []
    losses = []
    total_chunks = 0
    received_chunks = 0
    total_tx = 0

    for _, h in net_histories.items():
        if not isinstance(h, dict):
            continue
        for _, per_recv in h.items():
            if not isinstance(per_recv, dict):
                continue
            for _, info in per_recv.items():
                try:
                    latencies.append(float(info.get("latency", 0.0)))
                    losses.append(float(info.get("data_loss", 0.0)))
                    tc = int(info.get("total_chunks", 0))
                    rc = int(info.get("received_chunks", 0))
                    total_chunks += max(0, tc)
                    received_chunks += max(0, rc)
                    total_tx += 1
                except Exception:
                    pass

    delivery_ratio = (received_chunks / total_chunks) if total_chunks > 0 else float("nan")
    return {
        "tx_count": float(total_tx),
        "mean_latency_s": float(np.mean(latencies)) if latencies else float("nan"),
        "p50_latency_s": float(np.median(latencies)) if latencies else float("nan"),
        "mean_packet_loss": float(np.mean(losses)) if losses else float("nan"),
        "p90_packet_loss": float(np.percentile(losses, 90)) if losses else float("nan"),
        "chunk_delivery_ratio": float(delivery_ratio),
    }

def per_round_network_stats(net_hist_one_node: Dict[int, Dict[str, Dict[str, float]]]):
    """
    net_hist_one_node[round][receiver_id] = {
        "data_loss": ..., "latency": ..., "total_chunks": ..., "received_chunks": ...
    }
    Returns dict: round -> stats dict (scalars)
    """
    out = {}
    for r, per_recv in (net_hist_one_node or {}).items():
        if not isinstance(per_recv, dict) or len(per_recv) == 0:
            continue

        lat = []
        pl = []
        comp = []
        tot_chunks = 0
        rec_chunks = 0

        for _, info in per_recv.items():
            try:
                lat.append(float(info.get("latency", 0.0)))
                pl.append(float(info.get("data_loss", 0.0)))
                tc = int(info.get("total_chunks", 0))
                rc = int(info.get("received_chunks", 0))
                tot_chunks += max(0, tc)
                rec_chunks += max(0, rc)
                if tc > 0:
                    comp.append(rc / tc)
            except Exception:
                pass

        out[int(r)] = {
            "tx_count": float(len(per_recv)),
            "mean_latency_s": float(np.mean(lat)) if lat else np.nan,
            "p50_latency_s": float(np.median(lat)) if lat else np.nan,
            "mean_pkt_loss": float(np.mean(pl)) if pl else np.nan,
            "p90_pkt_loss": float(np.percentile(pl, 90)) if pl else np.nan,
            "chunk_delivery_ratio": float(rec_chunks / tot_chunks) if tot_chunks > 0 else np.nan,
            "mean_completeness": float(np.mean(comp)) if comp else np.nan,
        }
    return out


def build_matrix_by_common_round_scalar(per_node_round_to_val: Dict[int, Dict[int, float]], R_common: int):
    """
    per_node_round_to_val[node_id][round] = scalar
    returns M shape (N, R_common) filled with NaN for missing.
    """
    node_ids = sorted(per_node_round_to_val.keys())
    N = len(node_ids)
    M = np.full((N, R_common), np.nan, dtype=float)
    for i, nid in enumerate(node_ids):
        rrmap = per_node_round_to_val[nid]
        for r in range(1, R_common + 1):
            if r in rrmap:
                M[i, r - 1] = float(rrmap[r])
    return node_ids, M



# ----------------------------
# Plot helpers
# ----------------------------
def plot_acc_vs_round(out_png: Path, rounds: np.ndarray, mean: np.ndarray, p10: np.ndarray, p90: np.ndarray):
    set_icdcs_style()
    fig, ax = plt.subplots()
    ax.fill_between(rounds, p10, p90, alpha=0.20, label="p10–p90")
    ax.plot(rounds, mean, label="Mean")
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_acc_vs_time_of_round(out_png: Path, x_time: np.ndarray, mean: np.ndarray, p10: np.ndarray, p90: np.ndarray):
    set_icdcs_style()
    fig, ax = plt.subplots()
    ax.fill_between(x_time, p10, p90, alpha=0.20, label="p10–p90")
    ax.plot(x_time, mean, label="Mean")
    ax.set_xlabel("Time since global start (s)  [median node reaching round r]")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Dataset-based configs
# ----------------------------
def dataset_hparams(dataset: str) -> Dict:
    d = dataset.lower()

    base = {
        "weight_decay": 0.0,
        "momentum": 0.0,
        "scheduler": None,
        "milestones": [],
        "gamma": 0.0,
    }

    if d in ("mnist",):
        return {
            **base,
            "model_name": "mnist_mlp",
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 64,
            "local_epochs": 3,
        }

    if d in ("fmnist", "fashionmnist", "fashion_mnist"):
        return {
            **base,
            "model_name": "fmnist_mnistnet", 
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 64,
            "local_epochs": 3,
        }

    if d in ("cifar10", "cifar", "cifar-10"):
        return {
            **base,
            "model_name": "resnet18",
            "optimizer": "sgd",
            "learning_rate": 0.1,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "batch_size": 128,
            "local_epochs": 1,
            "scheduler": "multistep",
            "milestones": [60, 120, 160],
            "gamma": 0.2,
        }

    raise ValueError(f"Unknown dataset: {dataset}")


def build_run_hp(
    dataset: str,
    runtime_s: int,
    seed: int,
    base_net: Dict,
) -> Dict:
    hp = {
        **base_net,
        "runtime": int(runtime_s),
        "seed": int(seed),

        "staleness_tau_sec": 30.0, # Testing defaults is 30.0 | can change this for tau specific experiments
        "min_completeness": 0.2,
    }
    hp.update(dataset_hparams(dataset))
    return hp


def fmt_alpha(a: float) -> str:
    s = f"{a:.3f}".rstrip("0").rstrip(".")
    if "." not in s:  
        s = s + ".0"  
    return s.replace(".", "p")


def resolve_partition_dir(pattern: str, partitions_root: Path, dataset: str, alpha: float) -> Path:
    return Path(pattern.format(root=str(partitions_root), dataset=dataset, alpha=alpha, alpha_tag=fmt_alpha(alpha)))


def resolve_out_dir(results_root: Path, dataset: str, alpha: float, aggregation: str, seed: int) -> Path:
    return results_root / dataset / f"alpha_{fmt_alpha(alpha)}" / aggregation / f"seed_{seed}"


# ----------------------------
# Single run
# ----------------------------
def run_one(
    dataset: str,
    alpha: float,
    aggregation: str,
    seed: int,
    num_nodes: int,
    runtime_s: int,
    mobility_csv: str,
    partition_dir: Path,
    out_dir: Path,
    cpus_per_actor: float,
    max_concurrency: int,
    base_net: Dict,
):
    out_dir = ensure_dir(out_dir)

    run_spec = {
        "dataset": dataset,
        "alpha": alpha,
        "aggregation": aggregation,
        "seed": seed,
        "num_nodes": num_nodes,
        "runtime_s": runtime_s,
        "mobility_csv": mobility_csv,
        "partition_dir": str(partition_dir),
    }
    with open(out_dir / "run_spec.json", "w") as f:
        json.dump(run_spec, f, indent=2)

    client_data, test_data, test_labels = load_saved_partition(partition_dir)
    if len(client_data) < num_nodes:
        raise RuntimeError(f"Partitions have {len(client_data)} clients, but --num-nodes={num_nodes}")

    total_train_len = int(sum(len(c["data"]) for c in client_data[:num_nodes]))

    client_data_info = []
    for i in range(num_nodes):
        client_data_info.append({
            "client_id": f"node_{i}",
            "data_length": int(len(client_data[i]["data"])),
            "unique_classes": int(len(np.unique(client_data[i]["labels"]))),
        })

    hp = build_run_hp(dataset=dataset, runtime_s=runtime_s, seed=seed, base_net=base_net)
    with open(out_dir / "hyperparams.json", "w") as f:
        json.dump(hp, f, indent=2)

    if ray.is_initialized():
        ray.shutdown()

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)

    # Create DFL Ray actors
    actors: Dict[int, ray.actor.ActorHandle] = {}
    for i in range(num_nodes):
        a = DFedNode.options(
            name=f"node_{i}",
            num_cpus=cpus_per_actor,
            max_concurrency=max_concurrency,
        ).remote(
            node_id=f"node_{i}",
            train_data=client_data[i]["data"],
            train_labels=client_data[i]["labels"],
            test_data=test_data,
            test_labels=test_labels,
            num_nodes=num_nodes,
            hyperparams=hp,
            total_train_data_length=total_train_len,
            client_data_info=client_data_info,
            dataset_name=dataset,
            aggregation=aggregation,
            mobility_csv=mobility_csv,
        )
        actors[i] = a

    # Run
    global_start_time = time.time()
    print(f"Global start time: {global_start_time}")

    # Start all nodes with global clock
    t0 = time.time()
    ray.get([actors[i].start.remote(global_start_time) for i in range(num_nodes)])
    t1 = time.time()

    # Collect logs
    acc_logs = ray.get([actors[i].get_accuracy_log_after_aggregation.remote() for i in range(num_nodes)])
    try:
        loss_logs = ray.get([actors[i].get_loss_log_after_aggregation.remote() for i in range(num_nodes)])
    except Exception:
        loss_logs = [None] * num_nodes

    aoi_logs = ray.get([actors[i].get_aoi_round_log.remote() for i in range(num_nodes)])
    rows = [r for per_node in aoi_logs for r in (per_node or [])]
    if rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_dir / "aoi_round_summary.csv", index=False)

    starts = ray.get([actors[i].get_start_time.remote() for i in range(num_nodes)])
    net_histories = ray.get([actors[i].get_network_history.remote() for i in range(num_nodes)])

    starts_epoch = {i: float(starts[i]) for i in range(num_nodes)}
    net_map = {i: net_histories[i] for i in range(num_nodes)}

    # Parse per-node logs
    per_node_acc: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i in range(num_nodes):
        r, t, a = _parse_log_round_time_val(acc_logs[i] or [])
        per_node_acc[i] = (r, t, a)

    # Compute metrics fairnessly (COnsider globally achived rounds to collect mean values)
    rounds, M, R_common = build_matrix_by_common_round(per_node_acc)
    mean, p10, p50, p90 = summary_stats(M)
    x_time = time_of_round_axis(per_node_acc, starts_epoch, rounds)

    per_node_net_stats = {}
    for i in range(num_nodes):
        rrstats = per_round_network_stats(net_map[i])
        per_node_net_stats[i] = rrstats

    def export_series(metric_key: str, fname_prefix: str):
        per_node_metric = {i: {r: s.get(metric_key, np.nan) for r, s in per_node_net_stats[i].items()}
                           for i in range(num_nodes)}
        _, Mx = build_matrix_by_common_round_scalar(per_node_metric, R_common)
        mean_x = np.nanmean(Mx, axis=0)
        p10_x = np.nanpercentile(Mx, 10, axis=0)
        p50_x = np.nanpercentile(Mx, 50, axis=0)
        p90_x = np.nanpercentile(Mx, 90, axis=0)

        np.savetxt(
            out_dir / f"{fname_prefix}_vs_round.csv",
            np.column_stack([rounds, mean_x, p10_x, p50_x, p90_x]),
            delimiter=",",
            header="round,mean,p10,p50,p90",
            comments="",
        )

        np.savetxt(
            out_dir / f"{fname_prefix}_vs_time_of_round.csv",
            np.column_stack([x_time, rounds, mean_x, p10_x, p50_x, p90_x]),
            delimiter=",",
            header="time_s,round,mean,p10,p50,p90",
            comments="",
        )

    export_series("tx_count", "net_tx_count")
    export_series("mean_latency_s", "net_mean_latency")
    export_series("mean_pkt_loss", "net_mean_pkt_loss")
    export_series("chunk_delivery_ratio", "net_chunk_delivery_ratio")
    export_series("mean_completeness", "net_mean_completeness")

    final_acc = float(mean[-1])
    max_acc = float(np.nanmax(mean))
    auc_rounds = auc_trapz(mean)
    auc_time = auc_trapz(mean, x=x_time)

    t50 = first_crossing_time(x_time, mean, 50.0)
    t60 = first_crossing_time(x_time, mean, 60.0)
    t70 = first_crossing_time(x_time, mean, 70.0)

    final_per_node = M[:, -1]
    metrics = {
        "dataset": dataset,
        "alpha": float(alpha),
        "aggregation": aggregation,
        "seed": int(seed),
        "num_nodes": int(num_nodes),
        "runtime_s": int(runtime_s),
        "wall_time_s": float(t1 - t0),
        "R_common_used": int(R_common),
        "final_acc_common_round": final_acc,
        "max_acc_common_round": max_acc,
        "auc_over_rounds": float(auc_rounds),
        "auc_over_time_of_round": float(auc_time),
        "time_to_50": float(t50),
        "time_to_60": float(t60),
        "time_to_70": float(t70),
        "final_p10_acc": float(np.nanpercentile(final_per_node, 10)),
        "final_p90_acc": float(np.nanpercentile(final_per_node, 90)),
        "final_min_acc": float(np.nanmin(final_per_node)),
        "final_var_acc": float(np.nanvar(final_per_node)),
        "final_jain_acc": float(jain_index(final_per_node)),
        "final_gini_acc": float(gini_coefficient(final_per_node)),
        "final_cvar10_error": float(cvar_worst_alpha(100.0 - final_per_node, alpha=0.1)),
    }

    # Network summary
    nm = network_metrics_from_histories(net_map)
    metrics.update({
        "network_tx_count": nm["tx_count"],
        "network_mean_latency_s": nm["mean_latency_s"],
        "network_p50_latency_s": nm["p50_latency_s"],
        "network_mean_packet_loss": nm["mean_packet_loss"],
        "network_p90_packet_loss": nm["p90_packet_loss"],
        "network_chunk_delivery_ratio": nm["chunk_delivery_ratio"],
    })

    with open(out_dir / "per_node_accuracy_long.csv", "w") as f:
        f.write("node_id,round,time_s,acc\n")
        for i in range(num_nodes):
            r, t, a = per_node_acc[i]
            for rr, tt, aa in zip(r.tolist(), t.tolist(), a.tolist()):
                f.write(f"{i},{rr},{tt:.9f},{aa:.9f}\n")

    with open(out_dir / "per_node_loss_long.csv", "w") as f:
        f.write("node_id,round,time_s,loss\n")
        for i in range(num_nodes):
            if not loss_logs or not loss_logs[i]:
                continue
            r, t, v = _parse_log_round_time_val(loss_logs[i])
            for rr, tt, vv in zip(r.tolist(), t.tolist(), v.tolist()):
                f.write(f"{i},{rr},{tt:.9f},{vv:.9f}\n")

    np.savetxt(
        out_dir / "summary_round.csv",
        np.column_stack([rounds, mean, p10, p50, p90]),
        delimiter=",",
        header="round,mean,p10,p50,p90",
        comments="",
    )
    np.savetxt(
        out_dir / "summary_time_of_round.csv",
        np.column_stack([x_time, rounds, mean, p10, p50, p90]),
        delimiter=",",
        header="time_s,round,mean,p10,p50,p90",
        comments="",
    )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"[{dataset} | alpha={alpha} | {aggregation} | seed={seed}]\n")
        f.write(f"  R_common_used:             {metrics['R_common_used']}\n")
        f.write(f"  Final acc (common round):  {metrics['final_acc_common_round']:.4f}\n")
        f.write(f"  Max acc (common round):    {metrics['max_acc_common_round']:.4f}\n")
        f.write(f"  AUC over rounds:           {metrics['auc_over_rounds']:.2f}\n")
        f.write(f"  AUC over time-of-round:    {metrics['auc_over_time_of_round']:.2f}\n")
        f.write(f"  Time to 50%:               {metrics['time_to_50']}\n")
        f.write(f"  Time to 60%:               {metrics['time_to_60']}\n")
        f.write(f"  Time to 70%:               {metrics['time_to_70']}\n")
        f.write("\n")
        f.write("  --- extra ICDCS metrics (accuracy-only) ---\n")
        f.write(f"  Final p10/p90:             {metrics['final_p10_acc']:.4f} / {metrics['final_p90_acc']:.4f}\n")
        f.write(f"  Final min/var:             {metrics['final_min_acc']:.4f} / {metrics['final_var_acc']:.6f}\n")
        f.write(f"  Final Jain/Gini:           {metrics['final_jain_acc']:.4f} / {metrics['final_gini_acc']:.4f}\n")
        f.write(f"  Final CVaR10(error):       {metrics['final_cvar10_error']:.4f}\n")
        f.write("\n")
        f.write("  --- network summary (from network_history) ---\n")
        f.write(f"  TX count:                  {metrics['network_tx_count']}\n")
        f.write(f"  Mean/p50 latency (s):      {metrics['network_mean_latency_s']:.6f} / {metrics['network_p50_latency_s']:.6f}\n")
        f.write(f"  Mean/p90 pkt loss:         {metrics['network_mean_packet_loss']:.4f} / {metrics['network_p90_packet_loss']:.4f}\n")
        f.write(f"  Chunk delivery ratio:      {metrics['network_chunk_delivery_ratio']:.4f}\n")

    plot_acc_vs_round(out_dir / "acc_vs_round.png", rounds, mean, p10, p90)
    plot_acc_vs_time_of_round(out_dir / "acc_vs_time_of_round.png", x_time, mean, p10, p90)

    
    ray.shutdown()
    return metrics


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default="mnist,fmnist,cifar10")
    ap.add_argument("--alphas", type=str, default="0.1,0.5,1.0")
    ap.add_argument("--aggregations", type=str, default="fedavg,softSGD,dflaa")
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--num-nodes", type=int, default=20)
    ap.add_argument("--runtime-s", type=int, default=3000)
    ap.add_argument("--mobility-csv", type=str, default="mobility.csv")

    ap.add_argument("--partitions-root", type=str, default="dataset_partitions")
    ap.add_argument(
        "--partition-pattern",
        type=str,
        default="{root}/{dataset}/alpha_{alpha}",
        help="Python format string. Available: {root},{dataset},{alpha},{alpha_tag}",
    )

    ap.add_argument("--results-root", type=str, default="all_exp_results")

    ap.add_argument("--cpus-per-actor", type=float, default=1.0)
    ap.add_argument("--max-concurrency", type=int, default=32)

    ap.add_argument("--skip-existing", action="store_true", help="Skip run if metrics.json exists.")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    aggregations = [a.strip() for a in args.aggregations.split(",") if a.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    partitions_root = Path(args.partitions_root)
    results_root = ensure_dir(Path(args.results_root))

    base_net = {
        "base_latency_ms": 2.0,
        "latency_per_km": 5.0,
        "base_loss_ms": 0.02,
        "loss_per_100m": 0.03,
        "out_of_range": 900.0,
        "chunk_size_kb": 4,
    }

    sweep_rows = []
    for dataset in datasets:
        for alpha in alphas:
            for aggregation in aggregations:
                for seed in seeds:
                    part_dir = resolve_partition_dir(
                        pattern=args.partition_pattern,
                        partitions_root=partitions_root,
                        dataset=dataset,
                        alpha=alpha,
                    )
                    out_dir = resolve_out_dir(
                        results_root=results_root,
                        dataset=dataset,
                        alpha=alpha,
                        aggregation=aggregation,
                        seed=seed,
                    )
                    ensure_dir(out_dir)

                    if args.skip_existing and (out_dir / "metrics.json").exists():
                        print(f"[SKIP] exists: {out_dir}")
                        continue

                    if not (part_dir / "metadata.json").exists():
                        print(f"partition not found: {part_dir} (no metadata.json) -> skipping")
                        continue

                    print(f"Partitions: {part_dir}")
                    print(f"Results:    {out_dir}")

                    try:
                        metrics = run_one(
                            dataset=dataset,
                            alpha=alpha,
                            aggregation=aggregation,
                            seed=seed,
                            num_nodes=args.num_nodes,
                            runtime_s=args.runtime_s,
                            mobility_csv=args.mobility_csv,
                            partition_dir=part_dir,
                            out_dir=out_dir,
                            cpus_per_actor=args.cpus_per_actor,
                            max_concurrency=args.max_concurrency,
                            base_net=base_net,
                        )
                        sweep_rows.append(metrics)
                    except Exception as e:
                        print(f"[FAIL] {dataset} alpha={alpha} agg={aggregation} seed={seed}: {e}")
                        with open(out_dir / "ERROR.txt", "w") as f:
                            f.write(str(e))
        
                        try:
                            if ray.is_initialized():
                                ray.shutdown()
                        except Exception:
                            pass

    
    if sweep_rows:
        summary_path = results_root / "SWEEP_SUMMARY.csv"
        keys = sorted({k for row in sweep_rows for k in row.keys()})
        with open(summary_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in sweep_rows:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
        print(f"\nSweep summary saved: {summary_path}")
    else:
        print("\n No runs completed (check partitions paths / patterns).")


if __name__ == "__main__":
    main()
