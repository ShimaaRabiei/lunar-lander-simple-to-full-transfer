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
        rows = list(csv.DictReader(f))
    return [{k: v for k, v in row.items()} for row in rows]


def write_csv_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def pick(row, names, default=np.nan):
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return row[name]
    return default


def parse_lambda_from_any(row):
    direct = pick(row, ["lambda", "lambda_value", "variation_lambda"], default="")
    val = to_float(direct)
    if np.isfinite(val):
        return val

    for key in ["lambda_label", "run_name", "checkpoint", "log_path"]:
        if key in row:
            text = str(row[key])
            matches = re.findall(r"lam(?:bda)?[_-]?(-?\d+(?:p\d+)?|-?\d+(?:\.\d+)?)", text)
            if matches:
                token = matches[-1].replace("p", ".")
                val = to_float(token)
                if np.isfinite(val):
                    return val
    return np.nan


def normalize_reduced(rows):
    out = []
    for row in rows:
        lam = parse_lambda_from_any(row)
        out.append({
            "lambda": lam,
            "mean_discounted_task_return": to_float(pick(row, ["mean_discounted_task_return", "mean_discounted_common_task_return"])),
            "mean_discounted_variation": to_float(pick(row, ["mean_discounted_variation", "mean_discounted_theta_star_variation"])),
            "mean_final_distance_to_pad": to_float(pick(row, ["mean_final_distance_to_pad"])),
            "success_rate": to_float(pick(row, ["success_rate", "official_success_rate", "reduced_success_rule_rate"])),
            "mean_task_return": to_float(pick(row, ["mean_task_return", "mean_common_task_return"])),
            "mean_variation": to_float(pick(row, ["mean_variation", "mean_theta_star_variation"])),
        })
    out = [r for r in out if np.isfinite(r["lambda"])]
    out.sort(key=lambda r: r["lambda"])
    return out


def normalize_full(rows):
    out = []
    for row in rows:
        lam = parse_lambda_from_any(row)
        zeta = to_float(pick(row, ["zeta", "sweep_zeta", "inner_zeta"]))
        wn = to_float(pick(row, ["wn", "sweep_wn", "inner_wn"]))
        out.append({
            "lambda": lam,
            "zeta": zeta,
            "wn": wn,
            "mean_discounted_task_return": to_float(pick(row, ["mean_discounted_task_return", "mean_discounted_common_task_return"])),
            "mean_discounted_theta_star_variation": to_float(pick(row, ["mean_discounted_theta_star_variation", "mean_discounted_variation"])),
            "mean_final_distance_to_pad": to_float(pick(row, ["mean_final_distance_to_pad"])),
            "success_rate": to_float(pick(row, ["success_rate", "official_success_rate", "reduced_success_rule_rate"])),
            "mean_mean_abs_theta_tracking_error": to_float(pick(row, ["mean_mean_abs_theta_tracking_error", "mean_abs_theta_tracking_error"])),
            "mean_task_return": to_float(pick(row, ["mean_task_return", "mean_common_task_return"])),
        })
    out = [r for r in out if np.isfinite(r["lambda"]) and np.isfinite(r["zeta"]) and np.isfinite(r["wn"])]
    out.sort(key=lambda r: (r["zeta"], r["wn"], r["lambda"]))
    return out


def add_reduced_budget(full, reduced):
    budget = {r["lambda"]: r["mean_discounted_variation"] for r in reduced}
    out = []
    for r in full:
        rr = dict(r)
        rr["J_R"] = budget.get(r["lambda"], np.nan)
        out.append(rr)
    return out


def unique_sorted(rows, key):
    return sorted({r[key] for r in rows if np.isfinite(r[key])})


def line_plot(rows, x_key, y_key, xlabel, ylabel, title, out_path, annotate_key=None):
    xs = [r[x_key] for r in rows]
    ys = [r[y_key] for r in rows]
    plt.figure(figsize=(8.8, 5.5))
    plt.plot(xs, ys, marker="o", linewidth=2)
    if annotate_key:
        for x, y, r in zip(xs, ys, rows):
            if np.isfinite(x) and np.isfinite(y):
                plt.annotate(f"{r[annotate_key]:g}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def plot_reduced_summary(reduced, out_dir, training_samples):
    specs = [
        ("mean_discounted_task_return", "discounted task return", "reduced_discounted_task_return_vs_lambda"),
        ("mean_discounted_variation", r"$J_R$", "reduced_J_R_vs_lambda"),
        ("mean_final_distance_to_pad", "final distance to pad", "reduced_final_distance_vs_lambda"),
        ("success_rate", "success rate", "reduced_success_rate_vs_lambda"),
    ]
    title_suffix = f"training samples = {training_samples}"

    for key, ylabel, name in specs:
        line_plot(reduced, "lambda", key, "lambda", ylabel, f"{ylabel} vs lambda ({title_suffix})", out_dir / f"{name}.png", annotate_key="lambda")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, ylabel, _) in zip(axs.ravel(), specs):
        xs = [r["lambda"] for r in reduced]
        ys = [r[key] for r in reduced]
        ax.plot(xs, ys, marker="o", linewidth=2)
        for x, y, r in zip(xs, ys, reduced):
            if np.isfinite(x) and np.isfinite(y):
                ax.annotate(f"{r['lambda']:g}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center")
        ax.set_xlabel("lambda")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.35)
    fig.suptitle(f"Reduced model summary ({title_suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / "reduced_summary_2x2.png", dpi=200)
    fig.savefig(out_dir / "reduced_summary_2x2.pdf")
    plt.close(fig)


def aggregate_full_by_lambda(full):
    rows = []
    for lam in unique_sorted(full, "lambda"):
        group = [r for r in full if r["lambda"] == lam]
        if not group:
            continue
        rows.append({
            "lambda": lam,
            "J_R": group[0].get("J_R", np.nan),
            "mean_deployed_discounted_task_return": float(np.nanmean([r["mean_discounted_task_return"] for r in group])),
            "mean_deployed_discounted_theta_variation": float(np.nanmean([r["mean_discounted_theta_star_variation"] for r in group])),
            "mean_deployed_final_distance": float(np.nanmean([r["mean_final_distance_to_pad"] for r in group])),
            "mean_deployed_success_rate": float(np.nanmean([r["success_rate"] for r in group])),
            "mean_deployed_tracking_error": float(np.nanmean([r["mean_mean_abs_theta_tracking_error"] for r in group])),
        })
    rows.sort(key=lambda r: r["lambda"])
    return rows


def plot_deployed_aggregate(agg, out_dir, training_samples):
    specs = [
        ("mean_deployed_discounted_task_return", "deployed discounted task return", "deployed_discounted_task_return_vs_lambda"),
        ("mean_deployed_discounted_theta_variation", "deployed discounted theta-reference variation", "deployed_theta_variation_vs_lambda"),
        ("mean_deployed_final_distance", "deployed final distance to pad", "deployed_final_distance_vs_lambda"),
        ("mean_deployed_success_rate", "deployed success rate", "deployed_success_rate_vs_lambda"),
        ("mean_deployed_tracking_error", "theta tracking error", "deployed_tracking_error_vs_lambda"),
    ]
    title_suffix = f"training samples = {training_samples}"

    for key, ylabel, name in specs:
        line_plot(agg, "lambda", key, "lambda", ylabel, f"{ylabel} vs lambda ({title_suffix})", out_dir / f"{name}.png", annotate_key="lambda")

    line_plot(
        agg,
        "J_R",
        "mean_deployed_discounted_task_return",
        r"$J_R$ reduced discounted theta-reference variation",
        "deployed discounted task return",
        f"Deployed return vs reduced budget ({title_suffix})",
        out_dir / "deployed_return_vs_J_R.png",
        annotate_key="lambda",
    )


def plot_curves_by_zeta(full, out_dir):
    specs = [
        ("mean_discounted_task_return", "discounted task return", "curves_discounted_task_return", "Deployed discounted task return"),
        ("mean_discounted_theta_star_variation", "discounted theta-reference variation", "curves_discounted_theta_variation", "Deployed discounted theta-reference variation"),
        ("mean_final_distance_to_pad", "final distance to pad", "curves_final_distance", "Deployed final distance"),
        ("success_rate", "success rate", "curves_success_rate", "Deployed success rate"),
        ("mean_mean_abs_theta_tracking_error", "theta tracking error", "curves_theta_tracking_error", "Theta tracking error"),
    ]

    zetas = unique_sorted(full, "zeta")
    lambdas = unique_sorted(full, "lambda")

    for metric, ylabel, prefix, title in specs:
        for zeta in zetas:
            plt.figure(figsize=(9.5, 5.8))
            for lam in lambdas:
                group = [r for r in full if r["zeta"] == zeta and r["lambda"] == lam]
                group.sort(key=lambda r: r["wn"])
                if not group:
                    continue
                plt.plot([r["wn"] for r in group], [r[metric] for r in group], marker="o", linewidth=2, label=f"λ={lam:g}")
            plt.xlabel(r"$\omega_n$")
            plt.ylabel(ylabel)
            plt.title(f"{title}, zeta={zeta:g}")
            plt.grid(True, alpha=0.35)
            plt.legend(ncol=2, fontsize=9)
            plt.tight_layout()
            plt.savefig(out_dir / f"{prefix}_zeta_{str(zeta).replace('.', 'p')}.png", dpi=200)
            plt.savefig(out_dir / f"{prefix}_zeta_{str(zeta).replace('.', 'p')}.pdf")
            plt.close()


def text_color_for_value(value, norm, cmap):
    r, g, b, _ = cmap(norm(value))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.55 else "white"


def make_heatmap(matrix, labels, zetas, wns, title, colorbar_label, out_path, cmap_name="YlGnBu"):
    cmap = plt.get_cmap(cmap_name)
    finite_vals = matrix[np.isfinite(matrix)]
    if len(finite_vals) == 0:
        return
    vmin = float(np.min(finite_vals))
    vmax = float(np.max(finite_vals))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    plt.figure(figsize=(12, 6.2))
    im = plt.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    plt.colorbar(im, label=colorbar_label)
    plt.xticks(range(len(wns)), [f"{x:g}" for x in wns])
    plt.yticks(range(len(zetas)), [f"{z:g}" for z in zetas])
    plt.xlabel(r"$\omega_n$")
    plt.ylabel(r"$\zeta$")
    plt.title(title)

    for i in range(len(zetas)):
        for j in range(len(wns)):
            val = matrix[i, j]
            if not np.isfinite(val):
                continue
            plt.text(j, i, labels[i][j], ha="center", va="center", color=text_color_for_value(val, norm, cmap), fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def best_rows_by_gain(full):
    best_rows = []
    zetas = unique_sorted(full, "zeta")
    wns = unique_sorted(full, "wn")
    for zeta in zetas:
        for wn in wns:
            group = [r for r in full if r["zeta"] == zeta and r["wn"] == wn]
            if not group:
                continue
            best = max(group, key=lambda r: r["mean_discounted_task_return"])
            best_rows.append({
                "zeta": zeta,
                "wn": wn,
                "lambda": best["lambda"],
                "J_R": best.get("J_R", np.nan),
                "discounted_task_return": best["mean_discounted_task_return"],
                "success_rate": best["success_rate"],
                "final_distance": best["mean_final_distance_to_pad"],
            })
    return best_rows


def heatmaps(full, out_dir):
    best_rows = best_rows_by_gain(full)
    write_csv_rows(out_dir / "best_by_gain_summary.csv", best_rows)

    zetas = unique_sorted(best_rows, "zeta")
    wns = unique_sorted(best_rows, "wn")

    lambda_mat = np.full((len(zetas), len(wns)), np.nan)
    jr_mat = np.full((len(zetas), len(wns)), np.nan)
    ret_mat = np.full((len(zetas), len(wns)), np.nan)
    lambda_labels = [["" for _ in wns] for _ in zetas]
    jr_labels = [["" for _ in wns] for _ in zetas]
    ret_labels = [["" for _ in wns] for _ in zetas]

    for r in best_rows:
        i = zetas.index(r["zeta"])
        j = wns.index(r["wn"])
        lambda_mat[i, j] = r["lambda"]
        jr_mat[i, j] = r["J_R"]
        ret_mat[i, j] = r["discounted_task_return"]
        lambda_labels[i][j] = f"λ={r['lambda']:g}\nR={r['discounted_task_return']:.1f}"
        jr_labels[i][j] = f"λ={r['lambda']:g}\nJ_R={r['J_R']:.3f}"
        ret_labels[i][j] = f"R={r['discounted_task_return']:.1f}\nλ={r['lambda']:g}"

    make_heatmap(lambda_mat, lambda_labels, zetas, wns, "Best lambda by deployed discounted task return", "best lambda", out_dir / "best_lambda_heatmap_discounted_task_return_readable.png")
    make_heatmap(jr_mat, jr_labels, zetas, wns, r"Reduced budget $J_R$ of best deployed return", r"$J_R$ of best lambda", out_dir / "best_J_R_heatmap_discounted_task_return_readable.png")
    make_heatmap(ret_mat, ret_labels, zetas, wns, "Best deployed discounted task return", "discounted task return", out_dir / "best_value_heatmap_discounted_task_return_readable.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--training_samples", type=int, default=393216)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir else log_dir / "summary_plots_robust"
    out_dir.mkdir(parents=True, exist_ok=True)

    reduced_raw = read_csv_rows(log_dir / "reduced_summaries.csv")
    full_raw = read_csv_rows(log_dir / "full_summaries.csv")

    reduced = normalize_reduced(reduced_raw)
    full = add_reduced_budget(normalize_full(full_raw), reduced)
    agg = aggregate_full_by_lambda(full)

    write_csv_rows(out_dir / "reduced_summaries_normalized.csv", reduced)
    write_csv_rows(out_dir / "full_summaries_normalized.csv", full)
    write_csv_rows(out_dir / "lambda_aggregate_over_gains.csv", agg)

    plot_reduced_summary(reduced, out_dir, args.training_samples)
    plot_deployed_aggregate(agg, out_dir, args.training_samples)
    plot_curves_by_zeta(full, out_dir)
    heatmaps(full, out_dir)

    summary = {
        "log_dir": str(log_dir),
        "out_dir": str(out_dir),
        "num_reduced_rows": len(reduced),
        "num_full_rows": len(full),
        "lambdas": unique_sorted(reduced, "lambda"),
        "outputs": [
            "reduced_summary_2x2.png",
            "deployed_return_vs_J_R.png",
            "curves_discounted_task_return_zeta_*.png",
            "best_lambda_heatmap_discounted_task_return_readable.png",
            "best_J_R_heatmap_discounted_task_return_readable.png",
            "best_value_heatmap_discounted_task_return_readable.png",
        ],
    }
    (out_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
