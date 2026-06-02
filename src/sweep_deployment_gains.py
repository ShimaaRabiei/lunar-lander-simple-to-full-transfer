from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).lower()
    if x in ["1", "true", "yes", "y"]:
        return True
    if x in ["0", "false", "no", "n"]:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {x}")


def tag_float(x):
    return ("%g" % float(x)).replace("-", "m").replace(".", "p")


def parse_json_from_stdout(text):
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("Could not find JSON object in deployment output.")
    return json.loads(text[start:end + 1])


def run_one(args, zeta, wn):
    deploy_script = Path(args.deploy_script)
    run_name = f"{args.run_name}_zeta{tag_float(zeta)}_wn{tag_float(wn)}"
    cmd = [
        sys.executable,
        str(deploy_script),
        "--checkpoint", args.checkpoint,
        "--seed_file", args.seed_file,
        "--num_episodes", str(args.num_episodes),
        "--save_dir", args.save_dir,
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
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{run_name}.log"
    result = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(result.stdout + "\nSTDERR:\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Deployment failed for zeta={zeta}, wn={wn}. See {log_path}")
    data = parse_json_from_stdout(result.stdout)
    data["sweep_zeta"] = float(zeta)
    data["sweep_wn"] = float(wn)
    data["sweep_run_name"] = run_name
    data["sweep_log_path"] = str(log_path)
    data["sweep_command"] = " ".join(cmd)
    return data


def write_tables(rows, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gain_sweep_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    keys = sorted({k for row in rows for k in row.keys()})
    with (out_dir / "gain_sweep_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows, out_dir, metric, ylabel, baseline=None, baseline_key=None):
    out_dir = Path(out_dir)
    zetas = sorted({row["sweep_zeta"] for row in rows})
    plt.figure(figsize=(8, 5))
    for zeta in zetas:
        group = sorted([row for row in rows if row["sweep_zeta"] == zeta], key=lambda r: r["sweep_wn"])
        xs = [row["sweep_wn"] for row in group]
        ys = [row.get(metric, float("nan")) for row in group]
        plt.plot(xs, ys, marker="o", label=f"zeta={zeta:g}")
    if baseline is not None and baseline_key is not None and baseline_key in baseline:
        plt.axhline(float(baseline[baseline_key]), linestyle="--", label="reduced baseline")
    plt.xlabel("inner-loop natural frequency omega_n")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_vs_wn.png", dpi=180)
    plt.close()


def load_baseline(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed_file", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--deploy_script", default="")
    p.add_argument("--controller", default="pd", choices=["pd", "mrc", "adaptive_bias", "adaptive_gain"])
    p.add_argument("--scenario", default="nominal", choices=["nominal", "wind", "bias", "wind_bias"])
    p.add_argument("--variation_lambda", type=float, default=0.0)
    p.add_argument("--theta_limit_deg", type=float, default=20.0)
    p.add_argument("--step_penalty", type=float, default=0.02)
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
    p.add_argument("--reduced_summary_json", default="")
    args = p.parse_args()

    if not args.deploy_script:
        args.deploy_script = str(Path(__file__).with_name("deploy_reduced_order_lunar_lander.py"))

    rows = []
    for zeta in args.zeta_values:
        for wn in args.wn_values:
            print(f"running zeta={zeta:g}, wn={wn:g}", flush=True)
            row = run_one(args, zeta, wn)
            rows.append(row)
            write_tables(rows, args.output_dir)

    baseline = load_baseline(args.reduced_summary_json)
    plot_metric(rows, args.output_dir, "official_success_rate", "official success rate", baseline, "success_rate")
    plot_metric(rows, args.output_dir, "mean_common_task_return", "mean common task return", baseline, "mean_task_return")
    plot_metric(rows, args.output_dir, "mean_discounted_common_task_return", "mean discounted common task return", baseline, "mean_discounted_task_return")
    plot_metric(rows, args.output_dir, "mean_final_distance_to_pad", "mean final distance to pad", baseline, "mean_final_distance_to_pad")
    plot_metric(rows, args.output_dir, "mean_discounted_theta_star_variation", "mean discounted theta-star variation", baseline, "mean_discounted_variation")
    plot_metric(rows, args.output_dir, "mean_mean_abs_theta_tracking_error", "mean absolute theta tracking error")
    plot_metric(rows, args.output_dir, "mean_mean_side_power", "mean side power")
    print(json.dumps({"rows": len(rows), "output_dir": str(Path(args.output_dir).resolve())}, indent=2))


if __name__ == "__main__":
    main()
