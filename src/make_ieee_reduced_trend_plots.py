from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def to_float(x):
    try:
        if x is None:
            return np.nan
        s = str(x).strip()
        if s == "":
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def read_csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def pick(row, names, default=np.nan):
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return row[name]
    return default


def parse_lambda(row):
    direct = pick(row, ["lambda", "lambda_value", "variation_lambda"], default="")
    val = to_float(direct)
    if np.isfinite(val):
        return val

    for key in ["run_name", "checkpoint", "run_dir", "log_path"]:
        if key in row:
            text = str(row[key])
            matches = re.findall(r"lam(?:bda)?[_-]?(-?\d+(?:p\d+)?|-?\d+(?:\.\d+)?)", text)
            if matches:
                val = to_float(matches[-1].replace("p", "."))
                if np.isfinite(val):
                    return val
    return np.nan


def normalize_reduced(rows):
    out = []
    for row in rows:
        lam = parse_lambda(row)
        vr = to_float(pick(row, ["mean_discounted_task_return", "mean_discounted_common_task_return", "V_R"]))
        jr = to_float(pick(row, ["mean_discounted_variation", "mean_discounted_theta_star_variation", "J_R"]))
        task_return = to_float(pick(row, ["mean_task_return", "mean_common_task_return"]))
        success = to_float(pick(row, ["success_rate", "reduced_success_rule_rate", "official_success_rate"]))
        final_distance = to_float(pick(row, ["mean_final_distance_to_pad"]))
        if np.isfinite(lam):
            out.append({
                "lambda": lam,
                "V_R": vr,
                "J_R": jr,
                "mean_task_return": task_return,
                "success_rate": success,
                "mean_final_distance_to_pad": final_distance,
            })
    out.sort(key=lambda r: r["lambda"])
    return out


def set_ieee_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })


def format_samples(n: int) -> str:
    return f"{int(n):,}"


def annotate_points(ax, xs, ys, lambdas, dy=8):
    for x, y, lam in zip(xs, ys, lambdas):
        if np.isfinite(y):
            ax.annotate(f"{lam:g}", (x, y), textcoords="offset points", xytext=(0, dy), ha="center")


def plot_one(rows, key, ylabel, title, out_base: Path, ordered: bool):
    lambdas = [r["lambda"] for r in rows]
    ys = [r[key] for r in rows]

    if ordered:
        xs = list(range(len(rows)))
        xticks = xs
        xticklabels = [f"{lam:g}" for lam in lambdas]
    else:
        xs = lambdas
        xticks = None
        xticklabels = None

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(xs, ys, marker="o", linewidth=2.2)
    annotate_points(ax, xs, ys, lambdas)

    if ordered:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    suffix = "_ordered" if ordered else "_numeric"
    fig.savefig(out_base.with_name(out_base.name + suffix).with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_name(out_base.name + suffix).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_two_panel(rows, training_samples: int, out_base: Path, ordered: bool):
    lambdas = [r["lambda"] for r in rows]

    if ordered:
        xs = list(range(len(rows)))
        xticks = xs
        xticklabels = [f"{lam:g}" for lam in lambdas]
    else:
        xs = lambdas
        xticks = None
        xticklabels = None

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    specs = [
        ("V_R", r"$V_R$", r"Reduced discounted task return $V_R$"),
        ("J_R", r"$J_R$", r"Reduced budget $J_R$"),
    ]

    for ax, (key, ylabel, subtitle) in zip(axes, specs):
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, marker="o", linewidth=2.2)
        annotate_points(ax, xs, ys, lambdas, dy=7)
        if ordered:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        ax.grid(True, alpha=0.35)

    fig.suptitle(rf"Reduced-model trends (training samples = {format_samples(training_samples)})", y=1.03)
    fig.tight_layout()
    suffix = "_ordered" if ordered else "_numeric"
    fig.savefig(out_base.with_name(out_base.name + suffix).with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_name(out_base.name + suffix).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="", help="Folder containing reduced_summaries.csv.")
    parser.add_argument("--reduced_csv", default="", help="Direct path to reduced_summaries.csv.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--training_samples", type=int, default=1572864)
    args = parser.parse_args()

    set_ieee_style()

    if args.reduced_csv:
        reduced_path = Path(args.reduced_csv)
    else:
        reduced_path = Path(args.log_dir) / "reduced_summaries.csv"

    if not reduced_path.exists():
        raise FileNotFoundError(f"Could not find {reduced_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = normalize_reduced(read_csv_rows(reduced_path))
    if not rows:
        raise ValueError("No reduced rows with lambda values were found.")

    write_csv_rows(out_dir / "reduced_trends_ieee.csv", rows)

    sample_text = format_samples(args.training_samples)

    plot_one(
        rows,
        "V_R",
        r"$V_R$",
        rf"Reduced discounted task return $V_R$ (training samples = {sample_text})",
        out_dir / "reduced_VR_vs_lambda",
        ordered=True,
    )
    plot_one(
        rows,
        "V_R",
        r"$V_R$",
        rf"Reduced discounted task return $V_R$ (training samples = {sample_text})",
        out_dir / "reduced_VR_vs_lambda",
        ordered=False,
    )
    plot_one(
        rows,
        "J_R",
        r"$J_R$",
        rf"Reduced budget $J_R$ (training samples = {sample_text})",
        out_dir / "reduced_JR_vs_lambda",
        ordered=True,
    )
    plot_one(
        rows,
        "J_R",
        r"$J_R$",
        rf"Reduced budget $J_R$ (training samples = {sample_text})",
        out_dir / "reduced_JR_vs_lambda",
        ordered=False,
    )

    plot_two_panel(rows, args.training_samples, out_dir / "reduced_VR_JR_trends", ordered=True)
    plot_two_panel(rows, args.training_samples, out_dir / "reduced_VR_JR_trends", ordered=False)

    summary = {
        "reduced_csv": str(reduced_path),
        "out_dir": str(out_dir),
        "training_samples": int(args.training_samples),
        "lambdas": [r["lambda"] for r in rows],
        "outputs": [
            "reduced_VR_vs_lambda_ordered.png",
            "reduced_JR_vs_lambda_ordered.png",
            "reduced_VR_JR_trends_ordered.png",
            "reduced_VR_vs_lambda_numeric.png",
            "reduced_JR_vs_lambda_numeric.png",
            "reduced_VR_JR_trends_numeric.png",
            "reduced_trends_ieee.csv",
        ],
    }
    (out_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
