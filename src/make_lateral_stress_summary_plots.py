from __future__ import annotations

import argparse
import csv
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def tag_float(x: float) -> str:
    return ("%g" % float(x)).replace("-", "m").replace(".", "p")


def find_input_dir(path: Path):
    if path.is_file() and path.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(tmp.name)
        roots = [p for p in Path(tmp.name).iterdir() if p.is_dir()]
        if len(roots) == 1:
            return roots[0], tmp
        return Path(tmp.name), tmp
    return path, None


def to_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def read_csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if v is None:
                continue
            vv = str(v).strip()
            if vv == "":
                continue
            try:
                r[k] = float(vv)
            except ValueError:
                r[k] = vv
    return rows


def write_csv_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_required_csv(log_dir: Path):
    reduced_path = log_dir / "reduced_summaries.csv"
    full_path = log_dir / "full_summaries.csv"

    if not reduced_path.exists():
        raise FileNotFoundError(f"Could not find {reduced_path}")
    if not full_path.exists():
        raise FileNotFoundError(f"Could not find {full_path}")

    reduced = read_csv_rows(reduced_path)
    full = read_csv_rows(full_path)

    needed_reduced = ["lambda", "mean_discounted_task_return", "mean_discounted_variation", "mean_final_distance_to_pad", "success_rate"]
    needed_full = ["lambda", "zeta", "wn", "mean_discounted_task_return", "mean_discounted_theta_star_variation", "mean_final_distance_to_pad", "success_rate", "mean_mean_abs_theta_tracking_error"]

    missing_r = [k for k in needed_reduced if reduced and k not in reduced[0]]
    missing_f = [k for k in needed_full if full and k not in full[0]]

    if missing_r:
        raise ValueError(f"Missing columns in reduced_summaries.csv: {missing_r}")
    if missing_f:
        raise ValueError(f"Missing columns in full_summaries.csv: {missing_f}")

    reduced = sorted(reduced, key=lambda r: to_float(r["lambda"]))
    full = sorted(full, key=lambda r: (to_float(r["zeta"]), to_float(r["wn"]), to_float(r["lambda"])))
    return reduced, full


def attach_budget(full, reduced):
    budget = {to_float(r["lambda"]): to_float(r["mean_discounted_variation"]) for r in reduced}
    out = []
    for r in full:
        rr = dict(r)
        rr["J_R"] = budget.get(to_float(r["lambda"]), np.nan)
        out.append(rr)
    return out


def group_by_lambda(full):
    groups = {}
    for r in full:
        lam = to_float(r["lambda"])
        groups.setdefault(lam, []).append(r)
    return groups


def aggregate_over_gains(full):
    rows = []
    groups = group_by_lambda(full)
    for lam in sorted(groups):
        g = groups[lam]
        rows.append({
            "lambda": lam,
            "J_R": to_float(g[0].get("J_R", np.nan)),
            "mean_deployed_discounted_task_return": float(np.mean([to_float(r["mean_discounted_task_return"]) for r in g])),
            "std_deployed_discounted_task_return": float(np.std([to_float(r["mean_discounted_task_return"]) for r in g])),
            "mean_deployed_discounted_theta_variation": float(np.mean([to_float(r["mean_discounted_theta_star_variation"]) for r in g])),
            "mean_deployed_final_distance": float(np.mean([to_float(r["mean_final_distance_to_pad"]) for r in g])),
            "mean_deployed_success_rate": float(np.mean([to_float(r["success_rate"]) for r in g])),
            "mean_deployed_tracking_error": float(np.mean([to_float(r["mean_mean_abs_theta_tracking_error"]) for r in g])),
        })
    return rows


def save_tables(out_dir: Path, reduced, full):
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(out_dir / "reduced_summary_with_budget.csv", reduced)
    write_csv_rows(out_dir / "full_summary_with_reduced_budget.csv", full)
    agg = aggregate_over_gains(full)
    write_csv_rows(out_dir / "lambda_aggregate_over_gains.csv", agg)
    return agg


def line_plot(x, y, xlabel, ylabel, title, out_path: Path, annotate=False):
    plt.figure(figsize=(8.5, 5.4))
    plt.plot(x, y, marker="o", linewidth=2)
    if annotate:
        for xi, yi in zip(x, y):
            plt.annotate(f"{xi:g}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def plot_reduced_vs_lambda(reduced, out_dir: Path, training_samples: int | None):
    lambdas = [to_float(r["lambda"]) for r in reduced]
    sample_text = f"training samples = {training_samples}" if training_samples else "training samples not specified"

    specs = [
        ("mean_discounted_task_return", "discounted task return", "reduced_discounted_task_return_vs_lambda"),
        ("mean_discounted_variation", r"$J_R$", "reduced_J_R_vs_lambda"),
        ("mean_final_distance_to_pad", "final distance to pad", "reduced_final_distance_vs_lambda"),
        ("success_rate", "success rate", "reduced_success_rate_vs_lambda"),
    ]

    for col, ylabel, name in specs:
        y = [to_float(r[col]) for r in reduced]
        line_plot(lambdas, y, "lambda", ylabel, f"{ylabel} vs lambda ({sample_text})", out_dir / f"{name}.png", annotate=True)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, ylabel, _) in zip(axs.ravel(), specs):
        ax.plot(lambdas, [to_float(r[col]) for r in reduced], marker="o", linewidth=2)
        ax.set_xlabel("lambda")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.35)
    fig.suptitle(f"Reduced model summary ({sample_text})")
    fig.tight_layout()
    fig.savefig(out_dir / "reduced_summary_2x2.png", dpi=180)
    fig.savefig(out_dir / "reduced_summary_2x2.pdf")
    plt.close(fig)


def plot_full_aggregate_vs_lambda(agg, out_dir: Path, training_samples: int | None):
    lambdas = [to_float(r["lambda"]) for r in agg]
    sample_text = f"training samples = {training_samples}" if training_samples else "training samples not specified"

    specs = [
        ("mean_deployed_discounted_task_return", "deployed discounted task return", "deployed_discounted_task_return_vs_lambda"),
        ("mean_deployed_discounted_theta_variation", "deployed discounted theta-reference variation", "deployed_J_K_vs_lambda"),
        ("mean_deployed_final_distance", "deployed final distance to pad", "deployed_final_distance_vs_lambda"),
        ("mean_deployed_success_rate", "deployed success rate", "deployed_success_rate_vs_lambda"),
        ("mean_deployed_tracking_error", "theta tracking error", "deployed_tracking_error_vs_lambda"),
    ]

    for col, ylabel, name in specs:
        y = [to_float(r[col]) for r in agg]
        line_plot(lambdas, y, "lambda", ylabel, f"{ylabel} vs lambda ({sample_text})", out_dir / f"{name}.png", annotate=True)

    plt.figure(figsize=(8.5, 5.4))
    x = [to_float(r["J_R"]) for r in agg]
    y = [to_float(r["mean_deployed_discounted_task_return"]) for r in agg]
    plt.plot(x, y, marker="o", linewidth=2)
    for r in agg:
        plt.annotate(f"λ={to_float(r['lambda']):g}", (to_float(r["J_R"]), to_float(r["mean_deployed_discounted_task_return"])), textcoords="offset points", xytext=(0, 8), ha="center")
    plt.xlabel(r"$J_R$ reduced discounted theta-reference variation")
    plt.ylabel("deployed discounted task return")
    plt.title(f"Deployed return vs reduced budget ({sample_text})")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_dir / "deployed_return_vs_J_R.png", dpi=180)
    plt.savefig(out_dir / "deployed_return_vs_J_R.pdf")
    plt.close()


def sorted_unique(rows, key):
    return sorted({to_float(r[key]) for r in rows})


def heatmap_best_lambda(full, out_dir: Path, value_col: str, name: str, title: str, annotate_mode: str):
    zetas = sorted_unique(full, "zeta")
    wns = sorted_unique(full, "wn")

    matrix = np.full((len(zetas), len(wns)), np.nan)
    labels = [["" for _ in wns] for _ in zetas]

    for i, zeta in enumerate(zetas):
        for j, wn in enumerate(wns):
            sub = [r for r in full if to_float(r["zeta"]) == zeta and to_float(r["wn"]) == wn]
            if not sub:
                continue
            best = max(sub, key=lambda r: to_float(r[value_col]))
            matrix[i, j] = to_float(best["lambda"])
            if annotate_mode == "lambda_return":
                labels[i][j] = f"λ={to_float(best['lambda']):g}\nR={to_float(best[value_col]):.1f}"
            elif annotate_mode == "lambda_JR":
                labels[i][j] = f"λ={to_float(best['lambda']):g}\nJ_R={to_float(best['J_R']):.3f}"
            else:
                labels[i][j] = f"{to_float(best['lambda']):g}"

    plt.figure(figsize=(11, 6))
    im = plt.imshow(matrix, aspect="auto")
    plt.colorbar(im, label="best lambda")
    plt.xticks(range(len(wns)), [f"{x:g}" for x in wns])
    plt.yticks(range(len(zetas)), [f"{z:g}" for z in zetas])
    plt.xlabel(r"$\omega_n$")
    plt.ylabel(r"$\zeta$")
    plt.title(title)
    for i in range(len(zetas)):
        for j in range(len(wns)):
            plt.text(j, i, labels[i][j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}.png", dpi=180)
    plt.savefig(out_dir / f"{name}.pdf")
    plt.close()


def heatmap_best_value(full, out_dir: Path, metric: str, name: str, title: str):
    zetas = sorted_unique(full, "zeta")
    wns = sorted_unique(full, "wn")

    matrix = np.full((len(zetas), len(wns)), np.nan)

    for i, zeta in enumerate(zetas):
        for j, wn in enumerate(wns):
            sub = [r for r in full if to_float(r["zeta"]) == zeta and to_float(r["wn"]) == wn]
            if not sub:
                continue
            matrix[i, j] = max(to_float(r[metric]) for r in sub)

    plt.figure(figsize=(11, 6))
    im = plt.imshow(matrix, aspect="auto")
    plt.colorbar(im, label=metric)
    plt.xticks(range(len(wns)), [f"{x:g}" for x in wns])
    plt.yticks(range(len(zetas)), [f"{z:g}" for z in zetas])
    plt.xlabel(r"$\omega_n$")
    plt.ylabel(r"$\zeta$")
    plt.title(title)
    for i in range(len(zetas)):
        for j in range(len(wns)):
            if not np.isnan(matrix[i, j]):
                plt.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}.png", dpi=180)
    plt.savefig(out_dir / f"{name}.pdf")
    plt.close()


def plot_full_curves_by_zeta(full, out_dir: Path, metric: str, ylabel: str, name_prefix: str, title_prefix: str):
    zetas = sorted_unique(full, "zeta")
    lambdas = sorted_unique(full, "lambda")

    for zeta in zetas:
        plt.figure(figsize=(9, 5.6))
        for lam in lambdas:
            sub = [r for r in full if to_float(r["zeta"]) == zeta and to_float(r["lambda"]) == lam]
            sub = sorted(sub, key=lambda r: to_float(r["wn"]))
            if not sub:
                continue
            plt.plot([to_float(r["wn"]) for r in sub], [to_float(r[metric]) for r in sub], marker="o", linewidth=2, label=f"λ={lam:g}")
        plt.xlabel(r"$\omega_n$")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix}, zeta={zeta:g}")
        plt.grid(True, alpha=0.35)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / f"{name_prefix}_zeta_{tag_float(zeta)}.png", dpi=180)
        plt.savefig(out_dir / f"{name_prefix}_zeta_{tag_float(zeta)}.pdf")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--training_samples", type=int, default=393216)
    args = parser.parse_args()

    input_path = Path(args.log_dir)
    log_dir, tmp = find_input_dir(input_path)

    try:
        out_dir = Path(args.out_dir) if args.out_dir else (log_dir / "summary_plots")
        out_dir.mkdir(parents=True, exist_ok=True)

        reduced, full = read_required_csv(log_dir)
        full = attach_budget(full, reduced)
        agg = save_tables(out_dir, reduced, full)

        plot_reduced_vs_lambda(reduced, out_dir, args.training_samples)
        plot_full_aggregate_vs_lambda(agg, out_dir, args.training_samples)

        heatmap_best_lambda(
            full,
            out_dir,
            "mean_discounted_task_return",
            "best_lambda_heatmap_discounted_task_return",
            "Best lambda by deployed discounted task return",
            "lambda_return",
        )
        heatmap_best_lambda(
            full,
            out_dir,
            "mean_discounted_task_return",
            "best_J_R_heatmap_discounted_task_return",
            r"Reduced budget $J_R$ of best deployed return",
            "lambda_JR",
        )
        heatmap_best_value(
            full,
            out_dir,
            "mean_discounted_task_return",
            "best_value_heatmap_discounted_task_return",
            "Best deployed discounted task return",
        )

        plot_full_curves_by_zeta(full, out_dir, "mean_discounted_task_return", "discounted task return", "curves_discounted_task_return", "Deployed discounted task return")
        plot_full_curves_by_zeta(full, out_dir, "mean_discounted_theta_star_variation", "discounted theta-reference variation", "curves_discounted_theta_variation", "Deployed discounted theta-reference variation")
        plot_full_curves_by_zeta(full, out_dir, "mean_final_distance_to_pad", "final distance to pad", "curves_final_distance", "Deployed final distance")
        plot_full_curves_by_zeta(full, out_dir, "success_rate", "success rate", "curves_success_rate", "Deployed success rate")

        summary = {
            "log_dir": str(log_dir),
            "out_dir": str(out_dir.resolve()),
            "num_lambdas": len({to_float(r["lambda"]) for r in reduced}),
            "num_full_rows": len(full),
            "training_samples": int(args.training_samples),
            "main_outputs": [
                "reduced_summary_2x2.png",
                "deployed_return_vs_J_R.png",
                "best_lambda_heatmap_discounted_task_return.png",
                "best_J_R_heatmap_discounted_task_return.png",
                "best_value_heatmap_discounted_task_return.png",
            ],
        }
        (out_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
