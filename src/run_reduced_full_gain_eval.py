from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_reduced_order_lunar_lander import ActorCritic, Config, RunningMeanStd, evaluate, load_checkpoint, make_env


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in ["1", "true", "yes", "y"]:
        return True
    if x in ["0", "false", "no", "n"]:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {x}")


def resolve_checkpoint(path):
    p = Path(path)
    if p.is_dir():
        p = p / "final_model.pt"
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def tag_float(x):
    return ("%g" % float(x)).replace("-", "m").replace(".", "p")


def load_or_create_seeds(path, n, seed):
    p = Path(path)
    if p.exists():
        arr = np.load(p).astype(np.int64)
        if len(arr) < n:
            raise ValueError(f"Seed file has {len(arr)} seeds but {n} were requested: {p}")
        return arr[:n]
    p.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 2**31 - 1, size=n, dtype=np.int64)
    np.save(p, arr)
    with p.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_index", "reset_seed"])
        for i, s in enumerate(arr):
            writer.writerow([i, int(s)])
    return arr


def evaluate_reduced(args, checkpoint, seeds, out_dir):
    cfg = Config()
    cfg.mode = "eval"
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.theta_limit_deg = args.theta_limit_deg
    cfg.step_penalty = args.step_penalty
    cfg.timeout_penalty = args.timeout_penalty
    cfg.variation_lambda = args.variation_lambda
    cfg.final_eval_episodes = args.num_episodes
    cfg.deterministic_eval = args.deterministic_eval
    cfg.gamma = args.gamma

    probe_env = make_env(cfg, render_mode=None)
    obs, _ = probe_env.reset(seed=args.seed)
    obs_dim = len(obs)
    probe_env.close()

    model = ActorCritic(obs_dim, cfg.hidden_size, cfg.std_mode).to(args.device)
    obs_rms = RunningMeanStd((obs_dim,)) if cfg.obs_norm else None
    load_checkpoint(str(checkpoint), model, optimizer=None, obs_rms=obs_rms, device=args.device)
    model.eval()

    reduced_dir = out_dir / "reduced_eval"
    reduced_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate(cfg, model, obs_rms, seeds, run_dir=reduced_dir, prefix="reduced_eval", save_trajectory=False)
    (out_dir / "reduced_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_json_from_stdout(text):
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("Could not find JSON object in deployment output.")
    return json.loads(text[start:end + 1])


def run_deployment(args, checkpoint, deploy_script, zeta, wn, out_dir):
    run_name = f"{args.run_name}_zeta{tag_float(zeta)}_wn{tag_float(wn)}"
    cmd = [
        sys.executable,
        str(deploy_script),
        "--checkpoint", str(checkpoint),
        "--seed_file", args.seed_file,
        "--num_episodes", str(args.num_episodes),
        "--save_dir", str(args.deployment_save_dir),
        "--run_name", run_name,
        "--controller", args.controller,
        "--scenario", args.scenario,
        "--theta_limit_deg", str(args.theta_limit_deg),
        "--step_penalty", str(args.step_penalty),
        "--variation_lambda", str(args.variation_lambda),
        "--inner_wn", str(wn),
        "--inner_zeta", str(zeta),
        "--theta_dot_filter_alpha", str(args.theta_dot_filter_alpha),
        "--zero_side_on_contact", str(args.zero_side_on_contact).lower(),
        "--device", args.device,
        "--side_gain", str(args.side_gain),
        "--side_bias", str(args.side_bias),
        "--wind_power", str(args.wind_power),
        "--turbulence_power", str(args.turbulence_power),
        "--ref_wn", str(args.ref_wn),
        "--ref_zeta", str(args.ref_zeta),
    ]
    log_dir = out_dir / "deployment_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"
    result = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(result.stdout + "\nSTDERR:\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Deployment failed for zeta={zeta}, wn={wn}. See {log_path}")
    row = parse_json_from_stdout(result.stdout)
    row["sweep_zeta"] = float(zeta)
    row["sweep_wn"] = float(wn)
    row["sweep_run_name"] = run_name
    row["sweep_log_path"] = str(log_path)
    row["sweep_command"] = " ".join(cmd)
    return row


def write_tables(rows, out_dir):
    (out_dir / "deployment_gain_sweep_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    keys = sorted({k for row in rows for k in row.keys()})
    with (out_dir / "deployment_gain_sweep_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def value(row, key):
    v = row.get(key)
    if v is None or v == "":
        return float("nan")
    return float(v)


def plot_metric(rows, out_dir, metric, ylabel, filename, title, reduced_summary=None, reduced_key=None):
    zetas = sorted({value(row, "sweep_zeta") for row in rows})
    plt.figure(figsize=(8.5, 5.5))
    for zeta in zetas:
        group = sorted([row for row in rows if value(row, "sweep_zeta") == zeta], key=lambda r: value(r, "sweep_wn"))
        xs = [value(row, "sweep_wn") for row in group]
        ys = [value(row, metric) for row in group]
        plt.plot(xs, ys, marker="o", linewidth=2, label=f"zeta = {zeta:g}")
    if reduced_summary is not None and reduced_key is not None and reduced_key in reduced_summary:
        plt.axhline(float(reduced_summary[reduced_key]), linestyle="--", linewidth=2, label="reduced evaluation")
    plt.xlabel("inner-loop natural frequency omega_n")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=180)
    plt.savefig(out_dir / filename.replace(".png", ".pdf"))
    plt.close()


def make_plots(rows, reduced_summary, out_dir, title_prefix):
    plot_metric(rows, out_dir, "mean_discounted_common_task_return", "discounted common task return", "discounted_common_task_return_vs_wn.png", f"{title_prefix}: discounted common task return", reduced_summary, "mean_discounted_task_return")
    plot_metric(rows, out_dir, "mean_final_distance_to_pad", "final distance to pad", "final_distance_to_pad_vs_wn.png", f"{title_prefix}: final distance to pad", reduced_summary, "mean_final_distance_to_pad")
    plot_metric(rows, out_dir, "mean_discounted_theta_star_variation", "discounted theta-star variation", "discounted_theta_star_variation_vs_wn.png", f"{title_prefix}: discounted theta-star variation", reduced_summary, "mean_discounted_variation")
    plot_metric(rows, out_dir, "official_success_rate", "official success rate", "official_success_rate_vs_wn.png", f"{title_prefix}: official success rate", reduced_summary, "success_rate")
    plot_metric(rows, out_dir, "mean_mean_abs_theta_tracking_error", "mean absolute theta tracking error", "theta_tracking_error_vs_wn.png", f"{title_prefix}: theta tracking error", None, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", default="deployment_gain_eval")
    p.add_argument("--deployment_save_dir", default="")
    p.add_argument("--deploy_script", default="")
    p.add_argument("--controller", default="pd", choices=["pd", "mrc", "adaptive_bias", "adaptive_gain"])
    p.add_argument("--scenario", default="nominal", choices=["nominal", "wind", "bias", "wind_bias"])
    p.add_argument("--variation_lambda", type=float, default=0.0)
    p.add_argument("--theta_limit_deg", type=float, default=20.0)
    p.add_argument("--step_penalty", type=float, default=0.02)
    p.add_argument("--timeout_penalty", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--num_episodes", type=int, default=333)
    p.add_argument("--wn_values", type=float, nargs="+", default=[2, 4, 6, 8, 10, 12])
    p.add_argument("--zeta_values", type=float, nargs="+", default=[0.3, 0.7, 1.0])
    p.add_argument("--theta_dot_filter_alpha", type=float, default=0.85)
    p.add_argument("--zero_side_on_contact", type=str2bool, default=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--side_gain", type=float, default=1.0)
    p.add_argument("--side_bias", type=float, default=0.0)
    p.add_argument("--wind_power", type=float, default=15.0)
    p.add_argument("--turbulence_power", type=float, default=1.5)
    p.add_argument("--ref_wn", type=float, default=6.0)
    p.add_argument("--ref_zeta", type=float, default=1.0)
    p.add_argument("--deterministic_eval", type=str2bool, default=True)
    p.add_argument("--seed", type=int, default=20260528)
    args = p.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.deploy_script:
        args.deploy_script = str(Path(__file__).with_name("deploy_reduced_order_lunar_lander.py"))
    deploy_script = Path(args.deploy_script)
    if not deploy_script.exists():
        raise FileNotFoundError(f"Deployment script not found: {deploy_script}")

    if not args.deployment_save_dir:
        args.deployment_save_dir = str(out_dir / "deployment_runs")

    seeds = load_or_create_seeds(args.seed_file, args.num_episodes, args.seed)
    reduced_summary = evaluate_reduced(args, checkpoint, seeds, out_dir)

    rows = []
    for zeta in args.zeta_values:
        for wn in args.wn_values:
            print(f"running full deployment: zeta={zeta:g}, wn={wn:g}", flush=True)
            row = run_deployment(args, checkpoint, deploy_script, zeta, wn, out_dir)
            rows.append(row)
            write_tables(rows, out_dir)

    make_plots(rows, reduced_summary, out_dir, args.run_name)
    print(json.dumps({"episodes": args.num_episodes, "rows": len(rows), "output_dir": str(out_dir.resolve()), "reduced_summary": str((out_dir / "reduced_eval_summary.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
