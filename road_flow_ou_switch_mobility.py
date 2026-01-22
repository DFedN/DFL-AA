"""
road_flow_ou_switch_mobility.py
=====================================

Outputs: save to a csv file
time_sec,node_id,x_m,y_m,vx_mps,vy_mps,flow_vx_mps,flow_vy_mps,rel_vx_mps,rel_vy_mps,flow_id

Usage : Prepared for MNIST 3000s runtime:
  python road_flow_ou_switch_mobility.py \
    --nodes 20 --area_m 2000 --duration_s 3000 --dt_s 1 \
    --num_flows 4 --flow_heading_mode paired --flow_heading_deg 0 --flow_heading_spread_deg 90 \
    --flow_speed_kmph 35 --flow_speed_spread_kmph 8 \
    --tau_rel_s 600 --sigma_rel_mps 0.25 \
    --tau_switch_s 900 --switch_mode adjacent \
    --speed_scale 0.7 --boundary reflect --out_csv mobility.csv \
    --analyze_radius_m 400 --analyze_delta_s 60
"""

from __future__ import annotations
import argparse
import csv
import math
from typing import Tuple
import numpy as np


# ----------------------------
# Utilities
# ----------------------------
def wrap_angle(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2 * np.pi) - np.pi

def kmph_to_mps(x: float) -> float:
    return x * (1000.0 / 3600.0)

def deg_to_rad(x: float) -> float:
    return x * (math.pi / 180.0)

def apply_reflect_boundary(x, y, vx, vy, W, H):
    for _ in range(6):
        left = x < 0
        if np.any(left):
            x[left] = -x[left]
            vx[left] = -vx[left]
        right = x > W
        if np.any(right):
            x[right] = 2 * W - x[right]
            vx[right] = -vx[right]
        if not (np.any(x < 0) or np.any(x > W)):
            break

    for _ in range(6):
        bot = y < 0
        if np.any(bot):
            y[bot] = -y[bot]
            vy[bot] = -vy[bot]
        top = y > H
        if np.any(top):
            y[top] = 2 * H - y[top]
            vy[top] = -vy[top]
        if not (np.any(y < 0) or np.any(y > H)):
            break
    return x, y, vx, vy

def apply_wrap_boundary(x, y, W, H):
    return np.mod(x, W), np.mod(y, H)

def ou_step_exact(x: np.ndarray, dt: float, tau: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if tau <= 1e-9:
        return (sigma * rng.normal(0.0, 1.0, size=x.shape)).astype(np.float32)
    a = math.exp(-dt / tau)
    b = sigma * math.sqrt(max(0.0, 1.0 - a * a))
    return (a * x + b * rng.normal(0.0, 1.0, size=x.shape)).astype(np.float32)

def analyze_retention(positions: np.ndarray, times: np.ndarray, radius_m: float, delta_s: float) -> Tuple[float, float]:
    """
    Retention = |N_i(t) ∩ N_i(t+Δ)| / |N_i(t)|
    averaged across nodes and time.
    """
    T, N, _ = positions.shape
    if T < 2:
        return 0.0, 0.0
    dt = float(times[1] - times[0])
    step = int(round(delta_s / max(1e-9, dt)))
    if step <= 0 or step >= T:
        return 0.0, 0.0

    total_ret, total_deg, count = 0.0, 0.0, 0
    for t in range(0, T - step):
        P0 = positions[t]
        P1 = positions[t + step]
        D0 = np.linalg.norm(P0[None, :, :] - P0[:, None, :], axis=2)
        D1 = np.linalg.norm(P1[None, :, :] - P1[:, None, :], axis=2)

        for i in range(N):
            n0 = set(np.where((D0[i] <= radius_m) & (np.arange(N) != i))[0].tolist())
            n1 = set(np.where((D1[i] <= radius_m) & (np.arange(N) != i))[0].tolist())
            deg = len(n0)
            inter = len(n0.intersection(n1))
            total_ret += inter / max(1, deg)
            total_deg += deg
            count += 1

    return float(total_ret / max(1, count)), float(total_deg / max(1, count))


# ----------------------------
# Flow heading builders
# ----------------------------
def build_flow_headings(mode: str, F: int, base_heading: float, spread: float, rng: np.random.Generator) -> np.ndarray:
    
    if F <= 0:
        return np.zeros((0,), dtype=np.float32)

    if mode == "fixed":
        return np.full((F,), float(base_heading), dtype=np.float32)

    if mode == "random":
        return rng.uniform(-np.pi, np.pi, size=F).astype(np.float32)

    if mode == "circle":
        return wrap_angle((base_heading + (2 * np.pi) * (np.arange(F) / F)).astype(np.float32))

    if mode == "spread":
        if F == 1:
            return np.array([float(base_heading)], dtype=np.float32)
        offsets = np.linspace(-0.5 * spread, 0.5 * spread, F, endpoint=True).astype(np.float32)
        return wrap_angle((base_heading + offsets).astype(np.float32))

    if mode == "paired":
        headings = []
        K = (F + 1) // 2
        if K == 1:
            bases = np.array([float(base_heading)], dtype=np.float32)
        else:
            bases = (base_heading + np.linspace(-0.5 * spread, 0.5 * spread, K, endpoint=True)).astype(np.float32)
            bases = wrap_angle(bases)
        for b in bases:
            headings.append(float(b))
            if len(headings) < F:
                headings.append(float(wrap_angle(np.array([b + np.pi], dtype=np.float32))[0]))
            if len(headings) >= F:
                break
        return np.array(headings[:F], dtype=np.float32)

    raise ValueError(f"Unknown heading mode: {mode}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--nodes", type=int, default=20)
    ap.add_argument("--area_m", type=float, default=2000.0)
    ap.add_argument("--duration_s", type=int, default=3600)
    ap.add_argument("--dt_s", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boundary", choices=["reflect", "wrap"], default="reflect")
    ap.add_argument("--out_csv", type=str, default="mobility.csv")

    ap.add_argument("--num_flows", type=int, default=4)
    ap.add_argument("--flow_assign", choices=["round_robin", "random"], default="round_robin")

    ap.add_argument("--flow_heading_mode", choices=["fixed", "spread", "circle", "paired", "random"], default="paired")
    ap.add_argument("--flow_heading_deg", type=float, default=0.0)
    ap.add_argument("--flow_heading_spread_deg", type=float, default=90.0,
                    help="Used by spread/paired. For 'circle' this is ignored.")

    ap.add_argument("--flow_speed_kmph", type=float, default=35.0)
    ap.add_argument("--flow_speed_spread_kmph", type=float, default=8.0)

    ap.add_argument("--tau_flow_heading_s", type=float, default=1e9)
    ap.add_argument("--sigma_flow_heading_deg", type=float, default=0.0)
    ap.add_argument("--tau_flow_speed_s", type=float, default=1e9)
    ap.add_argument("--sigma_flow_speed_kmph", type=float, default=0.0)

    ap.add_argument("--tau_rel_s", type=float, default=600.0)
    ap.add_argument("--sigma_rel_mps", type=float, default=0.25)

    ap.add_argument("--tau_switch_s", type=float, default=900.0, help="Mean dwell time (seconds). <=0 disables.")
    ap.add_argument("--switch_mode", choices=["random", "adjacent"], default="adjacent")

    ap.add_argument("--speed_scale", type=float, default=0.7,
                    help="Multiply final velocity by this (slows motion without changing OU math).")
    ap.add_argument("--zero_drift", action="store_true",
                    help="Subtract mean velocity each step (safety against drift if assignment is unbalanced).")

    ap.add_argument("--analyze_radius_m", type=float, default=0.0)
    ap.add_argument("--analyze_delta_s", type=float, default=60.0)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    N = int(args.nodes)
    W = H = float(args.area_m)
    dt = float(args.dt_s)
    steps = int(round(args.duration_s / dt)) + 1

    F = max(1, int(args.num_flows))
    base_heading = deg_to_rad(args.flow_heading_deg)
    spread = deg_to_rad(args.flow_heading_spread_deg)

    base_speed = kmph_to_mps(args.flow_speed_kmph)
    speed_spread = kmph_to_mps(args.flow_speed_spread_kmph)

    flow_heading0 = build_flow_headings(args.flow_heading_mode, F, base_heading, spread, rng).astype(np.float32)

    if F == 1:
        flow_speed0 = np.array([float(base_speed)], dtype=np.float32)
    else:
        offsets = np.linspace(-0.5 * speed_spread, 0.5 * speed_spread, F, endpoint=True).astype(np.float32)
        flow_speed0 = np.clip((base_speed + offsets).astype(np.float32), 0.0, None)

    flow_heading = flow_heading0.copy()
    flow_speed = flow_speed0.copy()

    if args.flow_assign == "round_robin":
        flow_id = (np.arange(N) % F).astype(int)
    else:
        flow_id = rng.integers(0, F, size=N).astype(int)

    x = rng.uniform(0, W, size=N).astype(np.float32)
    y = rng.uniform(0, H, size=N).astype(np.float32)

    u = np.zeros((N, 2), dtype=np.float32)

    tau_switch = float(args.tau_switch_s)
    do_switch = tau_switch > 0.0 and math.isfinite(tau_switch)
    p_switch = (1.0 - math.exp(-dt / tau_switch)) if do_switch else 0.0

    do_analyze = args.analyze_radius_m > 0.0
    pos_hist = np.zeros((steps, N, 2), dtype=np.float32) if do_analyze else None
    t_hist = np.zeros((steps,), dtype=np.float32) if do_analyze else None

    sigma_flow_speed = kmph_to_mps(args.sigma_flow_speed_kmph)
    sigma_flow_heading = deg_to_rad(args.sigma_flow_heading_deg)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time_sec", "node_id",
            "x_m", "y_m",
            "vx_mps", "vy_mps",
            "flow_vx_mps", "flow_vy_mps",
            "rel_vx_mps", "rel_vy_mps",
            "flow_id",
        ])

        for step_i in range(steps):
            t = step_i * dt

            if do_switch and p_switch > 0:
                switches = rng.random(size=N) < p_switch
                if np.any(switches):
                    idxs = np.where(switches)[0]
                    for i in idxs:
                        cur = int(flow_id[i])
                        if F == 1:
                            continue
                        if args.switch_mode == "random":
                            choices = list(range(F))
                            choices.remove(cur)
                            flow_id[i] = int(rng.choice(choices))
                        else:  
                            flow_id[i] = (cur - 1) % F if (rng.random() < 0.5) else (cur + 1) % F

            if sigma_flow_speed > 0:
                s = (flow_speed - flow_speed0).astype(np.float32)
                s = ou_step_exact(s, dt, float(args.tau_flow_speed_s), float(sigma_flow_speed), rng)
                flow_speed = np.clip((flow_speed0 + s).astype(np.float32), 0.0, None)
            else:
                flow_speed = flow_speed0.copy()

            if sigma_flow_heading > 0:
                deltas = wrap_angle((flow_heading - flow_heading0).astype(np.float32))
                deltas = ou_step_exact(deltas, dt, float(args.tau_flow_heading_s), float(sigma_flow_heading), rng)
                flow_heading = wrap_angle((flow_heading0 + deltas).astype(np.float32))
            else:
                flow_heading = flow_heading0.copy()

            u = ou_step_exact(u, dt, float(args.tau_rel_s), float(args.sigma_rel_mps), rng)

            flow_v = np.zeros((N, 2), dtype=np.float32)
            for k in range(F):
                mask = (flow_id == k)
                if not np.any(mask):
                    continue
                vxk = float(flow_speed[k] * math.cos(float(flow_heading[k])))
                vyk = float(flow_speed[k] * math.sin(float(flow_heading[k])))
                flow_v[mask, 0] = vxk
                flow_v[mask, 1] = vyk

            v = (flow_v + u).astype(np.float32)

            if args.zero_drift:
                v = (v - v.mean(axis=0, keepdims=True)).astype(np.float32)

            v_eff = (float(args.speed_scale) * v).astype(np.float32)

            vx = v_eff[:, 0].copy()
            vy = v_eff[:, 1].copy()

            if do_analyze and pos_hist is not None and t_hist is not None:
                pos_hist[step_i, :, 0] = x
                pos_hist[step_i, :, 1] = y
                t_hist[step_i] = float(t)

            for i in range(N):
                w.writerow([
                    int(round(t)), i,
                    float(x[i]), float(y[i]),
                    float(vx[i]), float(vy[i]),
                    float(flow_v[i, 0]), float(flow_v[i, 1]),
                    float(u[i, 0]), float(u[i, 1]),
                    int(flow_id[i]),
                ])

            x = x + vx * dt
            y = y + vy * dt

            if args.boundary == "reflect":
                x, y, vx2, vy2 = apply_reflect_boundary(x, y, vx, vy, W, H)

                v_eff_ref = np.stack([vx2, vy2], axis=1).astype(np.float32)

                inv = 1.0 / max(1e-9, float(args.speed_scale))
                v_ref = (inv * v_eff_ref).astype(np.float32)
                if args.zero_drift:
                    v_ref = (v_ref - v_ref.mean(axis=0, keepdims=True)).astype(np.float32)
                    
                u = (v_ref - flow_v).astype(np.float32)
            else:
                x, y = apply_wrap_boundary(x, y, W, H)

    print(f"wrote {args.out_csv} | nodes={N} | steps={steps} | area={W}x{H}m | flows={F} | "
          f"heading_mode={args.flow_heading_mode} | switch_tau={tau_switch}s | speed_scale={args.speed_scale}")

    if do_analyze and pos_hist is not None and t_hist is not None:
        mean_ret, mean_deg = analyze_retention(pos_hist, t_hist, float(args.analyze_radius_m), float(args.analyze_delta_s))
        print(f"Analysing .... R={args.analyze_radius_m:.1f}m Δ={args.analyze_delta_s:.1f}s | mean_deg={mean_deg:.3f} | retention={mean_ret:.3f}")


if __name__ == "__main__":
    main()
