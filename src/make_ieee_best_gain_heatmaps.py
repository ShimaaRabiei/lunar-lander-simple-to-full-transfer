from __future__ import annotations

import argparse
import csv
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
        with path.open("w", newline="", encoding="utf-8") as f:
            pass
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
            "J_R": to_float(pick(row, ["mean_discounted_variation", "mean_discounted_theta_star_variation"])),
            "V_R": to_float(pick(row, ["mean_discounted_task_return", "mean_discounted_common_task_return"])),
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
            "V_K": to_float(pick(row, ["mean_discounted_task_return", "mean_discounted_common_task_return"])),
            "J_K": to_float(pick(row, ["mean_discounted_theta_star_variation", "mean_discounted_variation"])),
            "tracking_error": to_float(pick(row, ["mean_mean_abs_theta_tracking_error", "mean_abs_theta_tracking_error"])),
        })
    out = [r for r in out if np.isfinite(r["lambda"]) and np.isfinite(r["zeta"]) and np.isfinite(r["wn"])]
    out.sort(key=lambda r: (r["zeta"], r["wn"], r["lambda"]))
    return out


def unique_sorted(rows, key):
    return sorted({r[key] for r in rows if np.isfinite(r[key])})


def set_ieee_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })


def text_color_for_value(value, norm, cmap):
    r, g, b, _ = cmap(norm(value))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.60 else "white"


def format_samples(x):
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)


def build_best_rows(full_rows, reduced_rows):
    jr_by_lambda = {}
    for r in reduced_rows:
        lam = r["lambda"]
        if np.isfinite(lam) and np.isfinite(r["J_R"]):
            jr_by_lambda[lam] = r["J_R"]

    best_rows = []
    for zeta in unique_sorted(full_rows, "zeta"):
        for wn in unique_sorted(full_rows, "wn"):
            candidates = [r for r in full_rows if r["zeta"] == zeta and r["wn"] == wn]
            if not candidates:
                continue
            best = max(candidates, key=lambda r: (-1e18 if not np.isfinite(r["V_K"]) else r["V_K"]))
            lam = best["lambda"]
            best_rows.append({
                "zeta": zeta,
                "wn": wn,
                "lambda": lam,
                "J_R": jr_by_lambda.get(lam, np.nan),
                "V_K": best["V_K"],
                "J_K": best["J_K"],
                "tracking_error": best["tracking_error"],
            })
    best_rows.sort(key=lambda r: (r["zeta"], r["wn"]))
    return best_rows


def matrix_from_best(best_rows, zetas, wns, key):
    mat = np.full((len(zetas), len(wns)), np.nan)
    for r in best_rows:
        i = zetas.index(r["zeta"])
        j = wns.index(r["wn"])
        mat[i, j] = r[key]
    return mat


def label_matrices(best_rows, zetas, wns):
    v_labels = [["" for _ in wns] for _ in zetas]
    trk_labels = [["" for _ in wns] for _ in zetas]
    jk_labels = [["" for _ in wns] for _ in zetas]

    for r in best_rows:
        i = zetas.index(r["zeta"])
        j = wns.index(r["wn"])
        lam = r["lambda"]
        jr = r["J_R"]
        vk = r["V_K"]
        jk = r["J_K"]
        trk = r["tracking_error"]

        v_labels[i][j] = rf"$V_K$={vk:.1f}" + "\n" + rf"$\lambda$={lam:g}, $J_R$={jr:.3f}"
        trk_labels[i][j] = rf"tracking={trk:.3f}" + "\n" + rf"$\lambda$={lam:g}, $J_R$={jr:.3f}"
        jk_labels[i][j] = rf"$J_K$={jk:.3f}" + "\n" + rf"$\lambda$={lam:g}, $J_R$={jr:.3f}"

    return v_labels, trk_labels, jk_labels


def make_heatmap(matrix, labels, zetas, wns, title, colorbar_label, out_base, cmap_name="YlGnBu"):
    finite_vals = matrix[np.isfinite(matrix)]
    if len(finite_vals) == 0:
        raise ValueError(f"No finite values found for {out_base.name}")

    vmin = float(np.min(finite_vals))
    vmax = float(np.max(finite_vals))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6

    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12.8, 6.6))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xticks(range(len(wns)))
    ax.set_xticklabels([f"{x:g}" for x in wns])
    ax.set_yticks(range(len(zetas)))
    ax.set_yticklabels([f"{z:g}" for z in zetas])
    ax.set_xlabel(r"$\omega_n$")
    ax.set_ylabel(r"$\zeta$")
    ax.set_title(title)

    for i in range(len(zetas)):
        for j in range(len(wns)):
            val = matrix[i, j]
            if not np.isfinite(val):
                continue
            color = text_color_for_value(val, norm, cmap)
            ax.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=11)

    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--training_samples", type=int, required=True)
    parser.add_argument("--cmap_vk", default="YlGnBu")
    parser.add_argument("--cmap_tracking", default="YlOrRd")
    parser.add_argument("--cmap_jk", default="PuBuGn")
    args = parser.parse_args()

    set_ieee_style()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir else log_dir / "summary_plots_ieee"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = log_dir / "full_summaries.csv"
    reduced_path = log_dir / "reduced_summaries.csv"

    if not full_path.exists():
        raise FileNotFoundError(f"Could not find {full_path}")
    if not reduced_path.exists():
        raise FileNotFoundError(f"Could not find {reduced_path}")

    full_rows = normalize_full(read_csv_rows(full_path))
    reduced_rows = normalize_reduced(read_csv_rows(reduced_path))
    best_rows = build_best_rows(full_rows, reduced_rows)

    out_csv = out_dir / "best_by_gain_summary_ieee.csv"
    write_csv_rows(out_csv, best_rows)

    zetas = unique_sorted(best_rows, "zeta")
    wns = unique_sorted(best_rows, "wn")

    vk_mat = matrix_from_best(best_rows, zetas, wns, "V_K")
    trk_mat = matrix_from_best(best_rows, zetas, wns, "tracking_error")
    jk_mat = matrix_from_best(best_rows, zetas, wns, "J_K")

    v_labels, trk_labels, jk_labels = label_matrices(best_rows, zetas, wns)

    sample_txt = format_samples(args.training_samples)

    make_heatmap(
        vk_mat,
        v_labels,
        zetas,
        wns,
        rf"Best transfer discounted task return $V_K$ (training samples = {sample_txt})",
        r"discounted task return $V_K$",
        out_dir / "heatmap_best_VK",
        cmap_name=args.cmap_vk,
    )

    make_heatmap(
        trk_mat,
        trk_labels,
        zetas,
        wns,
        rf"Tracking error of the policy achieving the best $V_K$ (training samples = {sample_txt})",
        "mean absolute theta tracking error",
        out_dir / "heatmap_best_tracking_error",
        cmap_name=args.cmap_tracking,
    )

    make_heatmap(
        jk_mat,
        jk_labels,
        zetas,
        wns,
        rf"Transfer variation $J_K$ of the policy achieving the best $V_K$ (training samples = {sample_txt})",
        r"discounted transfer variation $J_K$",
        out_dir / "heatmap_best_JK",
        cmap_name=args.cmap_jk,
    )

    print({
        "out_dir": str(out_dir),
        "outputs": [
            "heatmap_best_VK.png",
            "heatmap_best_tracking_error.png",
            "heatmap_best_JK.png",
            "heatmap_best_VK.pdf",
            "heatmap_best_tracking_error.pdf",
            "heatmap_best_JK.pdf",
            "best_by_gain_summary_ieee.csv",
        ],
    })


if __name__ == "__main__":
    main()
