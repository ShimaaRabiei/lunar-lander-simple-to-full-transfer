from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def read_csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        clean = {}
        for k, v in row.items():
            if v is None:
                clean[k] = v
                continue
            vv = str(v).strip()
            try:
                clean[k] = float(vv)
            except ValueError:
                clean[k] = vv
        out.append(clean)
    return out


def sorted_unique(rows, key):
    return sorted({to_float(r[key]) for r in rows})


def text_color_for_value(value, norm, cmap):
    if value is None or np.isnan(value):
        return "black"
    r, g, b, _ = cmap(norm(value))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.55 else "white"


def build_best_rows(full_rows, reduced_rows):
    jr_by_lambda = {
        to_float(r["lambda"]): to_float(r["mean_discounted_variation"])
        for r in reduced_rows
    }

    zetas = sorted_unique(full_rows, "zeta")
    wns = sorted_unique(full_rows, "wn")
    best_rows = []

    for zeta in zetas:
        for wn in wns:
            candidates = [
                r for r in full_rows
                if to_float(r["zeta"]) == zeta and to_float(r["wn"]) == wn
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda r: to_float(r["mean_discounted_task_return"]))
            lam = to_float(best["lambda"])
            best_rows.append({
                "zeta": zeta,
                "wn": wn,
                "lambda": lam,
                "J_R": jr_by_lambda.get(lam, np.nan),
                "discounted_task_return": to_float(best["mean_discounted_task_return"]),
                "success_rate": to_float(best.get("success_rate", np.nan)),
                "final_distance": to_float(best.get("mean_final_distance_to_pad", np.nan)),
            })

    return best_rows


def matrix_from_best(best_rows, zetas, wns, key):
    mat = np.full((len(zetas), len(wns)), np.nan)
    for r in best_rows:
        i = zetas.index(to_float(r["zeta"]))
        j = wns.index(to_float(r["wn"]))
        mat[i, j] = to_float(r[key])
    return mat


def make_heatmap(matrix, labels, zetas, wns, title, colorbar_label, out_path, cmap_name="YlGnBu"):
    cmap = plt.get_cmap(cmap_name)
    finite_vals = matrix[np.isfinite(matrix)]
    if len(finite_vals) == 0:
        raise ValueError("No finite values to plot.")

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
            color = text_color_for_value(val, norm, cmap)
            plt.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def save_best_csv(best_rows, out_path):
    fieldnames = ["zeta", "wn", "lambda", "J_R", "discounted_task_return", "success_rate", "final_distance"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in best_rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--cmap", default="YlGnBu")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir else log_dir / "summary_plots_readable"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = log_dir / "full_summaries.csv"
    reduced_path = log_dir / "reduced_summaries.csv"

    if not full_path.exists():
        raise FileNotFoundError(f"Could not find {full_path}")
    if not reduced_path.exists():
        raise FileNotFoundError(f"Could not find {reduced_path}")

    full_rows = read_csv_rows(full_path)
    reduced_rows = read_csv_rows(reduced_path)

    best_rows = build_best_rows(full_rows, reduced_rows)
    save_best_csv(best_rows, out_dir / "best_by_gain_summary.csv")

    zetas = sorted_unique(best_rows, "zeta")
    wns = sorted_unique(best_rows, "wn")

    lambda_mat = matrix_from_best(best_rows, zetas, wns, "lambda")
    jr_mat = matrix_from_best(best_rows, zetas, wns, "J_R")
    return_mat = matrix_from_best(best_rows, zetas, wns, "discounted_task_return")

    lambda_labels = [["" for _ in wns] for _ in zetas]
    jr_labels = [["" for _ in wns] for _ in zetas]
    return_labels = [["" for _ in wns] for _ in zetas]

    for r in best_rows:
        i = zetas.index(to_float(r["zeta"]))
        j = wns.index(to_float(r["wn"]))
        lam = to_float(r["lambda"])
        jr = to_float(r["J_R"])
        ret = to_float(r["discounted_task_return"])
        lambda_labels[i][j] = f"λ={lam:g}\nR={ret:.1f}"
        jr_labels[i][j] = f"λ={lam:g}\nJ_R={jr:.3f}"
        return_labels[i][j] = f"R={ret:.1f}\nλ={lam:g}"

    make_heatmap(
        lambda_mat,
        lambda_labels,
        zetas,
        wns,
        "Best lambda by deployed discounted task return",
        "best lambda",
        out_dir / "best_lambda_heatmap_discounted_task_return_readable.png",
        cmap_name=args.cmap,
    )

    make_heatmap(
        jr_mat,
        jr_labels,
        zetas,
        wns,
        r"Reduced budget $J_R$ of best deployed return",
        r"$J_R$ of best lambda",
        out_dir / "best_J_R_heatmap_discounted_task_return_readable.png",
        cmap_name=args.cmap,
    )

    make_heatmap(
        return_mat,
        return_labels,
        zetas,
        wns,
        "Best deployed discounted task return",
        "discounted task return",
        out_dir / "best_value_heatmap_discounted_task_return_readable.png",
        cmap_name=args.cmap,
    )

    print({
        "out_dir": str(out_dir),
        "outputs": [
            "best_lambda_heatmap_discounted_task_return_readable.png",
            "best_J_R_heatmap_discounted_task_return_readable.png",
            "best_value_heatmap_discounted_task_return_readable.png",
            "best_by_gain_summary.csv",
        ],
    })


if __name__ == "__main__":
    main()
