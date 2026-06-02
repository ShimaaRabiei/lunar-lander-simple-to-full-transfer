from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_lambda_from_name(name: str):
    matches = re.findall(r"_lam(-?\d+(?:p\d+)?|-?\d+(?:\.\d+)?)", name)
    if not matches:
        return None
    token = matches[-1].replace("p", ".")
    try:
        return float(token)
    except ValueError:
        return None


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_rows(runs_root: Path, name_contains: str, summary_name: str):
    rows = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        if name_contains and name_contains not in d.name:
            continue

        lam = parse_lambda_from_name(d.name)
        if lam is None:
            continue

        summary_path = d / summary_name
        if not summary_path.exists():
            continue

        s = read_json(summary_path)
        rows.append(
            {
                "lambda": lam,
                "run_dir": str(d),
                "success_rate": float(s.get("success_rate", np.nan)),
                "crash_rate": float(s.get("crash_rate", np.nan)),
                "out_of_bounds_rate": float(s.get("out_of_bounds_rate", np.nan)),
                "mean_task_return": float(s.get("mean_task_return", np.nan)),
                "mean_discounted_task_return": float(s.get("mean_discounted_task_return", np.nan)),
                "mean_variation": float(s.get("mean_variation", np.nan)),
                "mean_discounted_variation": float(s.get("mean_discounted_variation", np.nan)),
                "mean_final_distance_to_pad": float(s.get("mean_final_distance_to_pad", np.nan)),
                "mean_final_speed": float(s.get("mean_final_speed", np.nan)),
                "landing_candidate_rate": float(s.get("landing_candidate_rate", np.nan)),
            }
        )

    rows.sort(key=lambda r: r["lambda"])
    return rows


def line_plot_numeric(rows, col, ylabel, title, out_path: Path):
    x = [r["lambda"] for r in rows]
    y = [r[col] for r in rows]

    plt.figure(figsize=(9.5, 5.6))
    plt.plot(x, y, marker="o", linewidth=2)
    for xi, yi in zip(x, y):
        if np.isnan(yi):
            continue
        plt.annotate(f"{xi:g}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center")
    plt.xlabel("lambda")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def line_plot_ordered(rows, col, ylabel, title, out_path: Path):
    labels = [f"{r['lambda']:g}" for r in rows]
    x = list(range(len(rows)))
    y = [r[col] for r in rows]

    plt.figure(figsize=(9.5, 5.6))
    plt.plot(x, y, marker="o", linewidth=2)
    for xi, yi, lab in zip(x, y, labels):
        if np.isnan(yi):
            continue
        plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center")
    plt.xticks(x, labels)
    plt.xlabel("lambda")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def make_2x2(rows, specs, out_path: Path, title: str, ordered: bool):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    if ordered:
        x = list(range(len(rows)))
        labels = [f"{r['lambda']:g}" for r in rows]
        xlabel = "lambda"
    else:
        x = [r["lambda"] for r in rows]
        labels = None
        xlabel = "lambda"

    for ax, (col, ylabel, _) in zip(axs.ravel(), specs):
        y = [r[col] for r in rows]
        ax.plot(x, y, marker="o", linewidth=2)
        for xi, yi, r in zip(x, y, rows):
            if np.isnan(yi):
                continue
            ax.annotate(f"{r['lambda']:g}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center")
        if ordered:
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.35)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def make_plots(rows, out_dir: Path, training_samples: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_text = f"training samples = {training_samples}"

    specs = [
        ("mean_discounted_task_return", "discounted task return", "discounted_task_return_vs_lambda"),
        ("mean_discounted_variation", r"$J_R$", "J_R_vs_lambda"),
        ("mean_final_distance_to_pad", "final distance to pad", "final_distance_to_pad_vs_lambda"),
        ("success_rate", "success rate", "success_rate_vs_lambda"),
    ]

    for col, ylabel, stem in specs:
        title = f"{ylabel} vs lambda ({sample_text})"
        line_plot_numeric(rows, col, ylabel, title, out_dir / f"{stem}_numeric.png")
        line_plot_ordered(rows, col, ylabel, title, out_dir / f"{stem}_ordered.png")

    make_2x2(
        rows,
        specs,
        out_dir / "reduced_summary_2x2_ordered.png",
        f"Reduced model summary ({sample_text})",
        ordered=True,
    )
    make_2x2(
        rows,
        specs,
        out_dir / "reduced_summary_2x2_numeric.png",
        f"Reduced model summary ({sample_text})",
        ordered=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", required=True, help="Root folder containing run directories.")
    parser.add_argument("--out_dir", default="", help="Where to save plots and reduced_summaries.csv.")
    parser.add_argument(
        "--name_contains",
        default="reduced_order_lander_lateral_x08_vx08_from_latbase_lam",
        help="Only scan run directories whose names contain this string.",
    )
    parser.add_argument("--summary_name", default="final_fixed333_summary.json")
    parser.add_argument("--training_samples", type=int, default=393216)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir) if args.out_dir else runs_root / "reduced_lambda_trend_plots"

    rows = collect_rows(runs_root, args.name_contains, args.summary_name)
    if not rows:
        raise FileNotFoundError(
            "No matching run folders with summary JSON were found. "
            "Check --runs_root, --name_contains, and --summary_name."
        )

    write_csv(out_dir / "reduced_summaries.csv", rows)
    make_plots(rows, out_dir, args.training_samples)

    summary = {
        "runs_root": str(runs_root.resolve()),
        "out_dir": str(out_dir.resolve()),
        "num_runs": len(rows),
        "lambdas": [r["lambda"] for r in rows],
        "main_outputs": [
            "discounted_task_return_vs_lambda_ordered.png",
            "J_R_vs_lambda_ordered.png",
            "final_distance_to_pad_vs_lambda_ordered.png",
            "success_rate_vs_lambda_ordered.png",
            "reduced_summary_2x2_ordered.png",
            "reduced_summaries.csv",
        ],
    }

    (out_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
