#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from lunar_lander import FPS
except Exception:
    FPS = 50.0

SOURCE_ORDER_WITH_ORIG = [
    "original_eval_common",
    "reduced_eval_common",
    "full_eval_pd_common",
    "full_eval_adaptive_gain_common",
    "full_eval_adaptive_bias_common",
]
SOURCE_ORDER_NO_ORIG = [
    "reduced_eval_common",
    "full_eval_pd_common",
    "full_eval_adaptive_gain_common",
    "full_eval_adaptive_bias_common",
]

DISPLAY_LABELS = {
    "original_eval_common": "Original",
    "reduced_eval_common": "Reduced",
    "full_eval_pd_common": "Transfer-PD",
    "full_eval_adaptive_gain_common": "Transfer-AdaptiveGain",
    "full_eval_adaptive_bias_common": "Transfer-AdaptiveBias",
}
SOURCE_COLORS = {
    "original_eval_common": "tab:blue",
    "reduced_eval_common": "tab:orange",
    "full_eval_pd_common": "tab:green",
    "full_eval_adaptive_gain_common": "tab:red",
    "full_eval_adaptive_bias_common": "tab:purple",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_regime_name(name: str) -> str:
    name = str(name).strip().lower()
    return "nominal" if name == "custom" else name


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def get_omega_final(row: pd.Series) -> float:
    if "omega_raw" in row.index and pd.notna(row["omega_raw"]):
        return float(row["omega_raw"])
    if "omega" in row.index and pd.notna(row["omega"]):
        return float(row["omega"])
    if "omega_scaled" in row.index and pd.notna(row["omega_scaled"]):
        return float(row["omega_scaled"]) * FPS / 20.0
    return float("nan")


def trajectory_common_metrics(df: pd.DataFrame, pad_x_threshold: float, vx_threshold: float, vy_threshold: float, omega_threshold: float) -> Dict[str, float]:
    final = df.iloc[-1]
    x = float(final["x"])
    vx = float(final["vx"])
    vy = float(final["vy"])
    omega = get_omega_final(final)
    leg_left = float(final.get("leg_left", 0.0))
    leg_right = float(final.get("leg_right", 0.0))
    crashed = bool(final.get("crashed", False))
    both_legs = bool((leg_left > 0.5) and (leg_right > 0.5))
    on_pad = bool(abs(x) <= pad_x_threshold)
    low_v = bool(abs(vx) <= vx_threshold and abs(vy) <= vy_threshold)
    low_omega = bool(np.isfinite(omega) and abs(omega) <= omega_threshold)
    common_success = bool((not crashed) and on_pad and both_legs and low_v and low_omega)
    touchdown_speed = float(math.sqrt(vx * vx + vy * vy))
    return_sum = float(df["reward"].sum()) if "reward" in df.columns else float("nan")
    return {
        "common_success": 1.0 if common_success else 0.0,
        "crash": 1.0 if crashed else 0.0,
        "on_pad": 1.0 if on_pad else 0.0,
        "both_legs": 1.0 if both_legs else 0.0,
        "low_velocity": 1.0 if low_v else 0.0,
        "low_omega": 1.0 if low_omega else 0.0,
        "final_x_abs": abs(x),
        "touchdown_speed": touchdown_speed,
        "episode_length": float(len(df)),
        "return_sum": return_sum,
    }


def aggregate_metrics(metric_rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(metric_rows[0].keys())
    out = {}
    for k in keys:
        vals = np.asarray([float(r[k]) for r in metric_rows], dtype=np.float64)
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


def compute_compare_dir_metrics(compare_dir: Path, pad_x_threshold: float, vx_threshold: float, vy_threshold: float, omega_threshold: float) -> Dict[str, Dict[str, float]]:
    metadata = load_json(compare_dir / "metadata.json")
    regime = normalize_regime_name(metadata.get("regime", compare_dir.name))
    traj_dir = compare_dir / "trajectories"
    system_patterns = {
        "reduced_eval_common": "reduced_init*_seed*.csv",
        "full_eval_pd_common": "pd_init*_seed*.csv",
        "full_eval_adaptive_gain_common": "adaptive_gain_init*_seed*.csv",
        "full_eval_adaptive_bias_common": "adaptive_bias_init*_seed*.csv",
    }
    out = {}
    for source_name, pattern in system_patterns.items():
        files = sorted(traj_dir.glob(pattern))
        if not files:
            continue
        metric_rows = []
        for path in files:
            df = read_csv(path)
            metric_rows.append(trajectory_common_metrics(df, pad_x_threshold, vx_threshold, vy_threshold, omega_threshold))
        agg = aggregate_metrics(metric_rows)
        agg["regime"] = regime
        agg["source"] = source_name
        agg["num_episodes"] = len(metric_rows)
        out[source_name] = agg
    return out


def compute_original_metrics(original_traj_root: Path, pad_x_threshold: float, vx_threshold: float, vy_threshold: float, omega_threshold: float) -> Dict[str, Dict[str, float]]:
    out = {}
    for regime_dir in sorted([p for p in original_traj_root.iterdir() if p.is_dir()]):
        regime = normalize_regime_name(regime_dir.name)
        files = sorted(regime_dir.glob("original_seed*.csv"))
        if not files:
            continue
        metric_rows = []
        for path in files:
            df = read_csv(path)
            metric_rows.append(trajectory_common_metrics(df, pad_x_threshold, vx_threshold, vy_threshold, omega_threshold))
        agg = aggregate_metrics(metric_rows)
        agg["regime"] = regime
        agg["source"] = "original_eval_common"
        agg["num_episodes"] = len(metric_rows)
        out[regime] = agg
    return out


def save_rows_csv(rows: List[dict], path: Path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_grouped_metric(rows: List[dict], metric_key: str, out_path: Path, include_original: bool):
    order = SOURCE_ORDER_WITH_ORIG if include_original else SOURCE_ORDER_NO_ORIG
    rows = [r for r in rows if r["source"] in order]
    regimes = sorted(set(r["regime"] for r in rows))
    x = np.arange(len(regimes))
    n = len(order)
    width = 0.80 / max(n, 1)

    value_map: Dict[str, Dict[str, float]] = {reg: {} for reg in regimes}
    for r in rows:
        value_map[r["regime"]][r["source"]] = float(r[metric_key])

    plt.figure(figsize=(max(12, 1.7 * len(regimes)), 6.5))
    for i, source in enumerate(order):
        offsets = x + (i - (n - 1) / 2.0) * width
        vals = [value_map[reg].get(source, np.nan) for reg in regimes]
        plt.bar(offsets, vals, width=width, label=DISPLAY_LABELS[source], color=SOURCE_COLORS[source])

    plt.xticks(x, regimes, rotation=20, ha="right")
    ylabel = {
        "common_success_mean": "Common success rate",
        "crash_mean": "Crash rate",
        "on_pad_mean": "On-pad rate",
        "final_x_abs_mean": "Mean final |x|",
        "touchdown_speed_mean": "Mean touchdown speed",
        "return_sum_mean": "Mean return (secondary)",
    }.get(metric_key, metric_key)
    plt.ylabel(ylabel)
    title_suffix = "with Original" if include_original else "without Original"
    plt.title(f"{ylabel} | {title_suffix}")
    if metric_key in {"common_success_mean", "crash_mean", "on_pad_mean"}:
        plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Compute common post-hoc metrics from saved trajectory CSVs.")
    p.add_argument("--compare_dirs", nargs="+", required=True)
    p.add_argument("--original_traj_root", type=str, default="")
    p.add_argument("--output_dir", type=str, default="common_metrics_from_trajectories")
    p.add_argument("--pad_x_threshold", type=float, default=0.20)
    p.add_argument("--vx_threshold", type=float, default=0.35)
    p.add_argument("--vy_threshold", type=float, default=0.35)
    p.add_argument("--omega_threshold", type=float, default=0.35)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    original_by_regime = {}
    if args.original_traj_root:
        original_by_regime = compute_original_metrics(
            Path(args.original_traj_root),
            args.pad_x_threshold,
            args.vx_threshold,
            args.vy_threshold,
            args.omega_threshold,
        )
    for cmp_dir_str in args.compare_dirs:
        cmp_dir = Path(cmp_dir_str)
        compare_metrics = compute_compare_dir_metrics(
            cmp_dir,
            args.pad_x_threshold,
            args.vx_threshold,
            args.vy_threshold,
            args.omega_threshold,
        )
        regime = None
        for agg in compare_metrics.values():
            regime = agg["regime"]
            rows.append(agg)
        if regime and regime in original_by_regime:
            rows.append(original_by_regime[regime])
    rows = sorted(rows, key=lambda r: (r["regime"], r["source"]))
    save_rows_csv(rows, output_dir / "common_metrics_summary.csv")
    with open(output_dir / "common_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with_original = output_dir / "with_original"
    without_original = output_dir / "without_original"
    with_original.mkdir(parents=True, exist_ok=True)
    without_original.mkdir(parents=True, exist_ok=True)
    for metric in [
        "common_success_mean",
        "crash_mean",
        "on_pad_mean",
        "final_x_abs_mean",
        "touchdown_speed_mean",
        "return_sum_mean",
    ]:
        plot_grouped_metric(rows, metric, with_original / f"{metric}_grouped_all_regimes.png", include_original=True)
        plot_grouped_metric(rows, metric, without_original / f"{metric}_grouped_all_regimes.png", include_original=False)
    print("Saved common metrics to:", output_dir)


if __name__ == "__main__":
    main()
