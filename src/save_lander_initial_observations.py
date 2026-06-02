from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train_reduced_order_lunar_lander import Config, make_env

try:
    from gymnasium.envs.box2d.lunar_lander import LunarLander
except Exception:
    from lunar_lander import LunarLander


def load_seeds(seed_file: str, episodes_csv: str, n: int):
    if seed_file:
        p = Path(seed_file)
        seeds = np.load(p).astype(np.int64)
        return seeds[:n]
    if episodes_csv:
        rows = []
        with Path(episodes_csv).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(int(row["reset_seed"]))
        return np.asarray(rows[:n], dtype=np.int64)
    raise ValueError("Provide either --seed_file or --episodes_csv.")


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed_file", default="")
    p.add_argument("--episodes_csv", default="")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_episodes", type=int, default=333)
    p.add_argument("--theta_limit_deg", type=float, default=20.0)
    p.add_argument("--gravity", type=float, default=-10.0)
    p.add_argument("--enable_wind", action="store_true")
    p.add_argument("--wind_power", type=float, default=0.0)
    p.add_argument("--turbulence_power", type=float, default=0.0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = load_seeds(args.seed_file, args.episodes_csv, args.num_episodes)

    full_env = LunarLander(
        render_mode=None,
        continuous=True,
        gravity=args.gravity,
        enable_wind=args.enable_wind,
        wind_power=args.wind_power,
        turbulence_power=args.turbulence_power,
    )

    cfg = Config()
    cfg.mode = "eval"
    cfg.theta_limit_deg = args.theta_limit_deg
    cfg.gravity = args.gravity
    cfg.enable_wind = args.enable_wind
    cfg.wind_power = args.wind_power
    cfg.turbulence_power = args.turbulence_power
    reduced_env = make_env(cfg, render_mode=None)

    official_rows = []
    reduced_rows = []

    for i, seed in enumerate(seeds):
        full_obs, _ = full_env.reset(seed=int(seed))
        red_obs, _ = reduced_env.reset(seed=int(seed))

        official_rows.append({
            "episode_index": i,
            "reset_seed": int(seed),
            "x": float(full_obs[0]),
            "y": float(full_obs[1]),
            "vx": float(full_obs[2]),
            "vy": float(full_obs[3]),
            "theta": float(full_obs[4]),
            "omega": float(full_obs[5]),
            "left_contact": float(full_obs[6]),
            "right_contact": float(full_obs[7]),
        })

        reduced_rows.append({
            "episode_index": i,
            "reset_seed": int(seed),
            "x": float(red_obs[0]),
            "y": float(red_obs[1]),
            "vx": float(red_obs[2]),
            "vy": float(red_obs[3]),
            "theta_star_previous": float(red_obs[4]) if len(red_obs) > 4 else 0.0,
            "left_contact": float(red_obs[-2]),
            "right_contact": float(red_obs[-1]),
        })

    full_env.close()
    reduced_env.close()

    official_csv = out_dir / "official_reset_initial_observations.csv"
    reduced_csv = out_dir / "reduced_reset_initial_observations.csv"

    with official_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(official_rows[0].keys()))
        writer.writeheader()
        writer.writerows(official_rows)

    with reduced_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(reduced_rows[0].keys()))
        writer.writeheader()
        writer.writerows(reduced_rows)

    official_summary = {}
    for key in ["x", "y", "vx", "vy", "theta", "omega"]:
        official_summary[key] = summarize([row[key] for row in official_rows])

    reduced_summary = {}
    for key in ["x", "y", "vx", "vy", "theta_star_previous"]:
        reduced_summary[key] = summarize([row[key] for row in reduced_rows])

    extra = {
        "num_episodes": int(len(seeds)),
        "official_abs_x_mean": float(np.mean(np.abs([r["x"] for r in official_rows]))),
        "official_abs_vx_mean": float(np.mean(np.abs([r["vx"] for r in official_rows]))),
        "official_abs_theta_mean": float(np.mean(np.abs([r["theta"] for r in official_rows]))),
        "fraction_abs_x_less_0p05": float(np.mean(np.abs([r["x"] for r in official_rows]) < 0.05)),
        "fraction_abs_vx_less_0p05": float(np.mean(np.abs([r["vx"] for r in official_rows]) < 0.05)),
        "fraction_abs_vx_greater_0p1": float(np.mean(np.abs([r["vx"] for r in official_rows]) > 0.1)),
    }

    summary = {
        "official_initial_observation_summary": official_summary,
        "reduced_initial_observation_summary": reduced_summary,
        "extra_checks": extra,
        "official_csv": str(official_csv),
        "reduced_csv": str(reduced_csv),
    }

    summary_path = out_dir / "initial_observation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        xs = [r["x"] for r in official_rows]
        vxs = [r["vx"] for r in official_rows]
        plt.figure(figsize=(6, 5))
        plt.scatter(xs, vxs, s=14, alpha=0.7)
        plt.axhline(0.0, linewidth=1)
        plt.axvline(0.0, linewidth=1)
        plt.xlabel("initial x")
        plt.ylabel("initial vx")
        plt.title("Official reset initial x-vx distribution")
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        plt.savefig(out_dir / "initial_x_vx_scatter.png", dpi=180)
        plt.close()
    except Exception:
        pass

    print(json.dumps(summary["extra_checks"], indent=2))
    print(f"Saved: {official_csv}")
    print(f"Saved: {reduced_csv}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
