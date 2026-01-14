#!/usr/bin/env python3
"""
data_loader.py
=========================

Generate partitions for:
  - datasets: mnist, fmnist, cifar10
  - splits: iid or non_iid_dirichlet
  - alphas: e.g., 0.1, 0.5, 1.0
and save to:
  partitions_sweep/{dataset}/alpha_{alpha_tag}/

Where alpha_tag:
  0.1 -> 0p1, 0.5 -> 0p5, 1.0 -> 1p0

Each output folder contains:
  - metadata.json
  - node_{i}_data.npy
  - node_{i}_labels.npy
  - test_data.npy
  - test_labels.npy
  - distribution_heatmap.png (optional)

Usage:
  python data_loader.py \
    --datasets mnist,fmnist,cifar10 \
    --alphas 0.1,0.5,1.0 \
    --num-clients 20 \
    --split non_iid_dirichlet \
    --out-root dataset_partitions \
    --seed 42

Then run all:
  python run_all_exp.py \
    --partitions-root dataset_partitions \
    --partition-pattern "{root}/{dataset}/alpha_{alpha_tag}" \
    --datasets mnist,fmnist,cifar10 \
    --alphas 0.1,0.5,1.0 \
    --aggregations vanilla,softSGD,dflaa
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets


# ----------------------------
# Helpers
# ----------------------------
def alpha_tag(alpha: float) -> str:
    # 0.1 -> 0p1, 1.0 -> 1p0 (stable folder naming, no dots)
    s = f"{alpha:.3f}".rstrip("0").rstrip(".")
    if "." not in s:
        s = s + ".0"
    return s.replace(".", "p")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_dataset_raw(dataset_name: str, data_dir: str = "./data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Return raw arrays (no torchvision transforms applied):
      X_train, y_train, X_test, y_test, num_classes

    Shapes:
      mnist/fmnist: X = (N, 28, 28) uint8
      cifar10:      X = (N, 32, 32, 3) uint8
    """
    dn = dataset_name.lower()

    if dn in ("mnist",):
        train = datasets.MNIST(root=data_dir, train=True, download=True)
        test = datasets.MNIST(root=data_dir, train=False, download=True)
        X_train = train.data.numpy()
        y_train = np.array(train.targets)
        X_test = test.data.numpy()
        y_test = np.array(test.targets)
        return X_train, y_train, X_test, y_test, 10

    if dn in ("fmnist", "fashionmnist", "fashion_mnist"):
        train = datasets.FashionMNIST(root=data_dir, train=True, download=True)
        test = datasets.FashionMNIST(root=data_dir, train=False, download=True)
        X_train = train.data.numpy()
        y_train = np.array(train.targets)
        X_test = test.data.numpy()
        y_test = np.array(test.targets)
        return X_train, y_train, X_test, y_test, 10

    if dn in ("cifar10", "cifar", "cifar-10"):
        train = datasets.CIFAR10(root=data_dir, train=True, download=True)
        test = datasets.CIFAR10(root=data_dir, train=False, download=True)
        # CIFAR10 stores numpy already
        X_train = np.array(train.data)  # (N,32,32,3)
        y_train = np.array(train.targets, dtype=np.int64)
        X_test = np.array(test.data)
        y_test = np.array(test.targets, dtype=np.int64)
        return X_train, y_train, X_test, y_test, 10

    raise ValueError(f"Unknown dataset: {dataset_name}")


def partition_iid(y_train: np.ndarray, num_clients: int, rng: np.random.Generator) -> List[np.ndarray]:
    idx = rng.permutation(len(y_train))
    splits = np.array_split(idx, num_clients)
    return [s.astype(np.int64) for s in splits]


def partition_dirichlet(
    y_train: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha: float,
    rng: np.random.Generator,
    min_size: int = 10,
    max_tries: int = 200,
) -> List[np.ndarray]:
    """
    Classic per-class Dirichlet partitioning.
    Retries until every client has at least min_size samples (helps alpha=0.1 not create empty clients).
    """
    class_indices = [np.where(y_train == c)[0].astype(np.int64) for c in range(num_classes)]

    for _ in range(max_tries):
        client_bins = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            idx_c = class_indices[c].copy()
            rng.shuffle(idx_c)

            props = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            # Convert proportions to split points
            split_points = (np.cumsum(props) * len(idx_c)).astype(int)
            split_points[-1] = len(idx_c)

            start = 0
            for k in range(num_clients):
                end = split_points[k]
                if end > start:
                    client_bins[k].extend(idx_c[start:end].tolist())
                start = end

        sizes = [len(b) for b in client_bins]
        if min(sizes) >= min_size:
            out = []
            for k in range(num_clients):
                arr = np.array(client_bins[k], dtype=np.int64)
                rng.shuffle(arr)
                out.append(arr)
            return out

    # Fallback: return whatever we have (still shuffled)
    out = []
    for k in range(num_clients):
        arr = np.array(client_bins[k], dtype=np.int64)
        rng.shuffle(arr)
        out.append(arr)
    return out


def compute_and_print_stats(y_train: np.ndarray, client_indices: List[np.ndarray], num_classes: int, title: str):
    print(f"\n=== Data Distribution ({title}) ===")
    mat = np.zeros((len(client_indices), num_classes), dtype=int)

    for i, idx in enumerate(client_indices):
        labels = y_train[idx]
        u, c = np.unique(labels, return_counts=True)
        for uu, cc in zip(u.tolist(), c.tolist()):
            mat[i, int(uu)] = int(cc)
        total = int(len(labels))
        nonzero = int(np.count_nonzero(mat[i]))
        print(f"node_{i}: {total} samples | non-empty classes={nonzero} | class_counts={dict(zip(u.tolist(), c.tolist()))}")

    # Non-IID metric (variance of normalized class histograms)
    row_sums = mat.sum(axis=1, keepdims=True) + 1e-8
    norm = mat / row_sums
    class_variances = np.var(norm, axis=0)
    print(f"Avg class-distribution variance: {float(np.mean(class_variances)):.4f}")
    print(f"Classes/client (min/avg/max): {int((mat>0).sum(axis=1).min())}/{float((mat>0).sum(axis=1).mean()):.2f}/{int((mat>0).sum(axis=1).max())}")
    return mat


def plot_heatmap(counts: np.ndarray, out_png: Path, title: str):
    """
    Bubble heatmap: clients vs classes, bubble size ~ count.
    """
    num_clients, num_classes = counts.shape
    totals = counts.sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    vmax = int(counts.max()) if counts.size else 1

    for i in range(num_clients):
        for j in range(num_classes):
            s = int(counts[i, j])
            if s > 0:
                ax.scatter(
                    j, i,
                    s=s * 0.3,
                    c=[s],
                    cmap="RdYlGn",
                    vmin=0,
                    vmax=vmax,
                    alpha=0.75,
                    edgecolors="k",
                )

    ax.set_xticks(range(num_classes))
    ax.set_xticklabels([f"{c}" for c in range(num_classes)])
    ax.set_yticks(range(num_clients))
    ax.set_yticklabels([f"node_{i}" for i in range(num_clients)])
    ax.set_xlabel("Class")
    ax.set_ylabel("Client")
    ax.set_title(title)

    for i, t in enumerate(totals.tolist()):
        ax.text(num_classes - 0.2, i, f"Total {int(t)}", va="center", ha="left", fontsize=9)

    if ax.collections:
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("Samples")

    ax.set_xlim(-0.5, num_classes + 1.5)
    plt.tight_layout()
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_partition_folder(
    out_dir: Path,
    dataset: str,
    split: str,
    alpha: Optional[float],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    client_indices: List[np.ndarray],
):
    ensure_dir(out_dir)

    partitions: Dict[str, Dict[str, str]] = {}
    for i, idx in enumerate(client_indices):
        node_id = f"node_{i}"  # IMPORTANT: node_0..node_{N-1}
        data_file = f"{node_id}_data.npy"
        labels_file = f"{node_id}_labels.npy"

        np.save(out_dir / data_file, X_train[idx])
        np.save(out_dir / labels_file, y_train[idx])

        partitions[node_id] = {"data_file": data_file, "labels_file": labels_file}

    np.save(out_dir / "test_data.npy", X_test)
    np.save(out_dir / "test_labels.npy", y_test)

    metadata = {
        "dataset": dataset,
        "num_clients": len(client_indices),
        "data_split": split,
        "alpha": alpha,
        "partitions": partitions,
        "test_data_file": "test_data.npy",
        "test_labels_file": "test_labels.npy",
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default="mnist,fmnist,cifar10")
    ap.add_argument("--alphas", type=str, default="0.1,0.5,1.0")
    ap.add_argument("--num-clients", type=int, default=20)
    ap.add_argument("--split", type=str, default="non_iid_dirichlet", choices=["iid", "non_iid_dirichlet"])
    ap.add_argument("--out-root", type=str, default="partitions_sweep")
    ap.add_argument("--data-dir", type=str, default="./data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-size", type=int, default=10, help="Min samples per client for dirichlet (retry)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    datasets_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    out_root = ensure_dir(Path(args.out_root))
    rng = np.random.default_rng(args.seed)

    for ds in datasets_list:
        X_train, y_train, X_test, y_test, num_classes = load_dataset_raw(ds, data_dir=args.data_dir)

        for a in (alphas if args.split == "non_iid_dirichlet" else [None]):
            if args.split == "iid":
                client_idx = partition_iid(y_train, args.num_clients, rng)
                folder = out_root / ds / "iid"
                title = f"{ds} | iid"
            else:
                client_idx = partition_dirichlet(
                    y_train=y_train,
                    num_clients=args.num_clients,
                    num_classes=num_classes,
                    alpha=float(a),
                    rng=rng,
                    min_size=int(args.min_size),
                )
                folder = out_root / ds / f"alpha_{alpha_tag(float(a))}"
                title = f"{ds} | dirichlet alpha={a}"

            ensure_dir(folder)

            # stats + plot
            counts = compute_and_print_stats(y_train, client_idx, num_classes, title=title)
            save_partition_folder(
                out_dir=folder,
                dataset=ds,
                split=args.split,
                alpha=a,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                client_indices=client_idx,
            )

            if not args.no_plot:
                plot_heatmap(counts, folder / "distribution_heatmap.png", title=title)

            print(f"\n✅ Saved: {folder}")

    print("\n[OK] All partitions created.")


if __name__ == "__main__":
    main()
