# -*- coding: utf-8 -*-
"""
Controllers-only common success rate plot
No std/error bars
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to common_metrics_summary.csv")
    parser.add_argument("--output_png", type=str, required=True,
                        help="Path to output PNG")
    parser.add_argument("--output_pdf", type=str, default=None,
                        help="Optional output PDF")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Keep only transfer controllers
    source_to_label = {
        "full_eval_pd_common": "Transfer-PD",
        "full_eval_adaptive_gain_common": "Transfer-AdaptiveGain",
        "full_eval_adaptive_bias_common": "Transfer-AdaptiveBias",
    }

    df = df[df["source"].isin(source_to_label.keys())].copy()
    df["controller"] = df["source"].map(source_to_label)

    # Regime order
    regime_order = ["nominal", "biased", "delay1", "windy", "hard"]
    df["regime"] = pd.Categorical(df["regime"], categories=regime_order, ordered=True)
    df = df.sort_values(["regime", "controller"])

    # Same colors as before
    color_map = {
        "Transfer-PD": "green",
        "Transfer-AdaptiveGain": "red",
        "Transfer-AdaptiveBias": "purple",
    }

    controllers = ["Transfer-PD", "Transfer-AdaptiveGain", "Transfer-AdaptiveBias"]
    regimes = [r for r in regime_order if r in df["regime"].dropna().unique()]

    # Plot
    x = np.arange(len(regimes))
    width = 0.22

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, ctrl in enumerate(controllers):
        sub = df[df["controller"] == ctrl].set_index("regime").reindex(regimes)
        means = sub["common_success_mean"].values

        ax.bar(
            x + (i - 1) * width,
            means,
            width=width,
            label=ctrl,
            color=color_map[ctrl]
        )

    ax.set_title("Common success rate | controllers only")
    ax.set_ylabel("Common success rate")
    ax.set_xlabel("Regime")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=20)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()

    outdir = os.path.dirname(args.output_png)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    plt.savefig(args.output_png, dpi=200, bbox_inches="tight")
    print(f"Saved PNG: {args.output_png}")

    if args.output_pdf:
        plt.savefig(args.output_pdf, bbox_inches="tight")
        print(f"Saved PDF: {args.output_pdf}")

    plt.close()


if __name__ == "__main__":
    main()