from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def resolve_checkpoint(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_dir():
        p = p / "final_model.pt"
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p.resolve()


def tag_float(x: float) -> str:
    return ("%g" % float(x)).replace("-", "m").replace(".", "p")


def load_or_create_seeds(path: str, n: int, seed: int):
    p = Path(path)
    if p.exists():
        seeds = np.load(p).astype(np.int64)
        if len(seeds) < n:
            raise ValueError(f"Seed file {p} has only {len(seeds)} seeds, but {n} were requested.")
        return seeds[:n]
    p.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n, dtype=np.int64)
    np.save(p, seeds)
    with p.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_index", "reset_seed"])
        for i, s in enumerate(seeds):
            writer.writerow([i, int(s)])
    return seeds


def parse_json_from_stdout(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("Could not find JSON block in deployment output.")
    return json.loads(text[start:end + 1])


def build_reduced_config(args, lam_value: float):
    cfg = Config()
    cfg.mode = "eval"
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.theta_limit_deg = args.theta_limit_deg
    cfg.step_penalty = args.step_penalty
    cfg.timeout_penalty = args.timeout_penalty
    cfg.variation_lambda = lam_value
    cfg.final_eval_episodes = args.num_episodes
    cfg.deterministic_eval = args.deterministic_eval
    cfg.gamma = args.gamma
    return cfg


def evaluate_reduced(checkpoint: Path, args, lam_value: float, seeds, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_reduced_config(args, lam_value)
    probe_env = make_env(cfg, render_mode=None)
    obs, _ = probe_env.reset(seed=args.seed)
    obs_dim = len(obs)
    probe_env.close()

    model = ActorCritic(obs_dim, cfg.hidden_size, cfg.std_mode).to(args.device)
    obs_rms = RunningMeanStd((obs_dim,)) if cfg.obs_norm else None
    load_checkpoint(str(checkpoint), model, optimizer=None, obs_rms=obs_rms, device=args.device)
    model.eval()

    summary = evaluate(
        cfg,
        model,
        obs_rms,
        seeds,
        run_dir=out_dir,
        prefix="reduced_eval",
        save_trajectory=False,
    )
    (out_dir / "reduced_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_full_deployment(checkpoint: Path, args, lam_value: float, lam_label: str, wn: float, zeta: float, out_dir: Path):
    deploy_script = Path(args.deploy_script)
    run_name = f"{lam_label}_zeta{tag_float(zeta)}_wn{tag_float(wn)}"

    cmd = [
        sys.executable,
        str(deploy_script),
        "--checkpoint", str(checkpoint),
        "--seed_file", args.seed_file,
        "--num_episodes", str(args.num_episodes),
        "--save_dir", str(out_dir / "deployment_runs"),
        "--run_name", run_name,
        "--controller", args.controller,
        "--scenario", args.scenario,
        "--theta_limit_deg", str(args.theta_limit_deg),
        "--step_penalty", str(args.step_penalty),
        "--variation_lambda", str(lam_value),
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

    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_name}.log"

    result = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(result.stdout + "\nSTDERR:\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Deployment failed for lambda={lam_label}, zeta={zeta}, wn={wn}. See {log_path}")

    row = parse_json_from_stdout(result.stdout)
    row["lambda_label"] = lam_label
    row["lambda_value"] = float(lam_value)
    row["sweep_wn"] = float(wn)
    row["sweep_zeta"] = float(zeta)
    row["log_path"] = str(log_path)
    return row


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_csv(path: Path, rows):
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def metric_value(row, key):
    if key is None:
        return np.nan
    v = row.get(key, np.nan)
    try:
        return float(v)
    except Exception:
        return np.nan


def lambda_style_map(lam_labels):
    cmap = plt.get_cmap("tab10")
    return {lam: cmap(i % 10) for i, lam in enumerate(lam_labels)}


def zeta_style_map(zetas):
    styles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "d", "v", "P"]
    out = {}
    for i, zeta in enumerate(zetas):
        out[zeta] = {
            "linestyle": styles[i % len(styles)],
            "marker": markers[i % len(markers)],
        }
    return out


def plot_metric_by_zeta(full_rows, reduced_rows, metric_key, reduced_key, ylabel, filename_prefix, title_prefix, output_dir: Path):
    zetas = sorted({metric_value(r, "sweep_zeta") for r in full_rows})
    lam_labels = sorted({r["lambda_label"] for r in full_rows}, key=lambda x: float(x))
    colors = lambda_style_map(lam_labels)

    for zeta in zetas:
        zeta_rows = [r for r in full_rows if metric_value(r, "sweep_zeta") == zeta]
        plt.figure(figsize=(8.5, 5.5))

        for lam_label in lam_labels:
            group = sorted(
                [r for r in zeta_rows if r["lambda_label"] == lam_label],
                key=lambda r: metric_value(r, "sweep_wn")
            )
            xs = [metric_value(r, "sweep_wn") for r in group]
            ys = [metric_value(r, metric_key) for r in group]
            plt.plot(xs, ys, marker="o", linewidth=2, color=colors[lam_label], label=fr"$\lambda$ = {lam_label}")

            reduced_match = next((rr for rr in reduced_rows if rr["lambda_label"] == lam_label), None)
            if reduced_match is not None and reduced_key is not None and reduced_key in reduced_match:
                y0 = float(reduced_match[reduced_key])
                plt.axhline(y0, linestyle="--", linewidth=1.4, color=colors[lam_label], alpha=0.8)

        plt.xlabel(r"$\omega_n$ [rad/s]")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix} (zeta = {zeta:g})")
        plt.grid(True, alpha=0.35)
        plt.legend()
        plt.tight_layout()

        png_path = output_dir / f"{filename_prefix}_zeta_{tag_float(zeta)}.png"
        pdf_path = output_dir / f"{filename_prefix}_zeta_{tag_float(zeta)}.pdf"
        plt.savefig(png_path, dpi=180)
        plt.savefig(pdf_path)
        plt.close()


def plot_metric_master(full_rows, reduced_rows, metric_key, reduced_key, ylabel, filename_prefix, title_prefix, output_dir: Path):
    zetas = sorted({metric_value(r, "sweep_zeta") for r in full_rows})
    lam_labels = sorted({r["lambda_label"] for r in full_rows}, key=lambda x: float(x))
    colors = lambda_style_map(lam_labels)
    zstyles = zeta_style_map(zetas)

    plt.figure(figsize=(11, 7))

    for lam_label in lam_labels:
        reduced_match = next((rr for rr in reduced_rows if rr["lambda_label"] == lam_label), None)
        for zeta in zetas:
            group = sorted(
                [r for r in full_rows if r["lambda_label"] == lam_label and metric_value(r, "sweep_zeta") == zeta],
                key=lambda r: metric_value(r, "sweep_wn")
            )
            if not group:
                continue
            xs = [metric_value(r, "sweep_wn") for r in group]
            ys = [metric_value(r, metric_key) for r in group]
            plt.plot(
                xs,
                ys,
                linewidth=2,
                color=colors[lam_label],
                linestyle=zstyles[zeta]["linestyle"],
                marker=zstyles[zeta]["marker"],
                label=fr"$\lambda$={lam_label}, $\zeta$={zeta:g}",
            )

        if reduced_match is not None and reduced_key is not None and reduced_key in reduced_match:
            y0 = float(reduced_match[reduced_key])
            plt.axhline(
                y0,
                linewidth=1.5,
                color=colors[lam_label],
                linestyle=(0, (6, 3)),
                alpha=0.9,
                label=fr"reduced $\lambda$={lam_label}",
            )

    plt.xlabel(r"$\omega_n$ [rad/s]")
    plt.ylabel(ylabel)
    plt.title(title_prefix + " (master plot)")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()

    png_path = output_dir / f"{filename_prefix}_master.png"
    pdf_path = output_dir / f"{filename_prefix}_master.pdf"
    plt.savefig(png_path, dpi=180)
    plt.savefig(pdf_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint files or run folders.")
    parser.add_argument("--lambdas", nargs="+", type=float, required=True, help="Lambda values corresponding to checkpoints.")
    parser.add_argument("--seed_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default="multi_lambda_gain_eval")

    parser.add_argument("--deploy_script", default="")
    parser.add_argument("--controller", default="pd", choices=["pd", "mrc", "adaptive_bias", "adaptive_gain"])
    parser.add_argument("--scenario", default="nominal", choices=["nominal", "wind", "bias", "wind_bias"])

    parser.add_argument("--theta_limit_deg", type=float, default=20.0)
    parser.add_argument("--step_penalty", type=float, default=0.02)
    parser.add_argument("--timeout_penalty", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num_episodes", type=int, default=333)

    parser.add_argument("--wn_values", nargs="+", type=float, default=[2, 4, 6, 8, 10, 12])
    parser.add_argument("--zeta_values", nargs="+", type=float, default=[0.3, 0.7, 1.0])

    parser.add_argument("--theta_dot_filter_alpha", type=float, default=0.85)
    parser.add_argument("--zero_side_on_contact", type=str2bool, default=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deterministic_eval", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=20260528)

    parser.add_argument("--side_gain", type=float, default=1.0)
    parser.add_argument("--side_bias", type=float, default=0.0)
    parser.add_argument("--wind_power", type=float, default=15.0)
    parser.add_argument("--turbulence_power", type=float, default=1.5)
    parser.add_argument("--ref_wn", type=float, default=6.0)
    parser.add_argument("--ref_zeta", type=float, default=1.0)
    parser.add_argument("--make_master_plots", type=str2bool, default=True)

    args = parser.parse_args()

    if len(args.checkpoints) != len(args.lambdas):
        raise ValueError("Number of checkpoints must equal number of lambdas.")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.deploy_script:
        args.deploy_script = str(Path(__file__).with_name("deploy_reduced_order_lunar_lander.py"))
    if not Path(args.deploy_script).exists():
        raise FileNotFoundError(f"Could not find deploy script: {args.deploy_script}")

    resolved_checkpoints = [resolve_checkpoint(x) for x in args.checkpoints]
    seeds = load_or_create_seeds(args.seed_file, args.num_episodes, args.seed)

    reduced_rows = []
    full_rows = []

    for checkpoint, lam_value in zip(resolved_checkpoints, args.lambdas):
        lam_label = ("%g" % lam_value)
        lam_dir = out_dir / f"lambda_{tag_float(lam_value)}"
        lam_dir.mkdir(parents=True, exist_ok=True)

        print(f"evaluating reduced model for lambda={lam_label}", flush=True)
        reduced_summary = evaluate_reduced(checkpoint, args, lam_value, seeds, lam_dir / "reduced_eval")
        reduced_summary["lambda_label"] = lam_label
        reduced_summary["lambda_value"] = float(lam_value)
        reduced_summary["checkpoint"] = str(checkpoint)
        reduced_rows.append(reduced_summary)

        for zeta in args.zeta_values:
            for wn in args.wn_values:
                print(f"evaluating full deployment for lambda={lam_label}, zeta={zeta:g}, wn={wn:g}", flush=True)
                row = run_full_deployment(checkpoint, args, lam_value, lam_label, wn, zeta, lam_dir / "full_eval")
                full_rows.append(row)

        write_json(lam_dir / "reduced_eval_summary.json", reduced_summary)

    write_json(out_dir / "reduced_summaries.json", reduced_rows)
    write_json(out_dir / "full_summaries.json", full_rows)
    write_csv(out_dir / "reduced_summaries.csv", reduced_rows)
    write_csv(out_dir / "full_summaries.csv", full_rows)

    metric_specs = [
        {
            "metric_key": "mean_discounted_common_task_return",
            "reduced_key": "mean_discounted_task_return",
            "ylabel": "discounted task return",
            "filename_prefix": "discounted_task_return_vs_wn",
            "title_prefix": "Discounted task return vs omega_n",
        },
        {
            "metric_key": "mean_final_distance_to_pad",
            "reduced_key": "mean_final_distance_to_pad",
            "ylabel": "final distance to pad",
            "filename_prefix": "final_distance_to_pad_vs_wn",
            "title_prefix": "Final distance to pad vs omega_n",
        },
        {
            "metric_key": "mean_discounted_theta_star_variation",
            "reduced_key": "mean_discounted_variation",
            "ylabel": "discounted theta-reference variation",
            "filename_prefix": "discounted_theta_reference_variation_vs_wn",
            "title_prefix": "Discounted theta-reference variation vs omega_n",
        },
        {
            "metric_key": "official_success_rate",
            "reduced_key": "success_rate",
            "ylabel": "success rate",
            "filename_prefix": "success_rate_vs_wn",
            "title_prefix": "Success rate vs omega_n",
        },
        {
            "metric_key": "mean_mean_abs_theta_tracking_error",
            "reduced_key": None,
            "ylabel": "mean absolute theta tracking error",
            "filename_prefix": "theta_tracking_error_vs_wn",
            "title_prefix": "Theta tracking error vs omega_n",
        },
    ]

    for spec in metric_specs:
        plot_metric_by_zeta(
            full_rows,
            reduced_rows,
            metric_key=spec["metric_key"],
            reduced_key=spec["reduced_key"],
            ylabel=spec["ylabel"],
            filename_prefix=spec["filename_prefix"],
            title_prefix=spec["title_prefix"],
            output_dir=out_dir,
        )
        if args.make_master_plots:
            plot_metric_master(
                full_rows,
                reduced_rows,
                metric_key=spec["metric_key"],
                reduced_key=spec["reduced_key"],
                ylabel=spec["ylabel"],
                filename_prefix=spec["filename_prefix"],
                title_prefix=spec["title_prefix"],
                output_dir=out_dir,
            )

    final_summary = {
        "run_name": args.run_name,
        "num_lambdas": len(args.lambdas),
        "num_full_rows": len(full_rows),
        "num_reduced_rows": len(reduced_rows),
        "output_dir": str(out_dir),
        "plots": [
            "discounted_task_return_vs_wn_zeta_*.png",
            "final_distance_to_pad_vs_wn_zeta_*.png",
            "discounted_theta_reference_variation_vs_wn_zeta_*.png",
            "success_rate_vs_wn_zeta_*.png",
            "theta_tracking_error_vs_wn_zeta_*.png",
            "discounted_task_return_vs_wn_master.png",
            "final_distance_to_pad_vs_wn_master.png",
            "discounted_theta_reference_variation_vs_wn_master.png",
            "success_rate_vs_wn_master.png",
            "theta_tracking_error_vs_wn_master.png",
        ],
    }
    write_json(out_dir / "final_summary.json", final_summary)
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
