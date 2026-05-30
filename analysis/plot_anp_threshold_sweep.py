"""
Plot ANP threshold sweep — quantitative ACC/ASR/RA trade-off as the
adversarial-pruning mask threshold increases.

Run from project root:
    python analysis/plot_anp_threshold_sweep.py --result_file wanet_cifar10_fast
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_file", type=str, required=True)
    args = p.parse_args()

    record_dir = Path("./record") / args.result_file
    csv = record_dir / "defense" / "anp" / "threshold_df.csv"
    df = pd.read_csv(csv).sort_values("threshold")

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax[0].plot(df["threshold"], df["test_acc"], marker="o", color="#3b82f6", label="clean ACC")
    ax[0].plot(df["threshold"], df["test_asr"], marker="s", color="#ef4444", label="ASR")
    ax[0].plot(df["threshold"], df["test_ra"],  marker="^", color="#10b981", label="RA")
    ax[0].set_xlabel("ANP mask threshold")
    ax[0].set_ylabel("metric")
    ax[0].set_ylim(0, 1.02)
    ax[0].set_title("Threshold sweep — ACC / ASR / RA")
    ax[0].grid(linestyle=":", alpha=0.5)
    ax[0].legend()

    ax[1].plot(df["test_asr"], df["test_acc"], marker="o", linestyle="-", color="#6366f1")
    for _, r in df.iterrows():
        ax[1].annotate(f"t={r['threshold']:.2f}",
                       (r["test_asr"], r["test_acc"]),
                       textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.7)
    ax[1].set_xlabel("ASR (lower = better defense)")
    ax[1].set_ylabel("clean ACC (higher = better)")
    ax[1].set_title("ACC vs ASR — Pareto frontier")
    ax[1].invert_xaxis()
    ax[1].grid(linestyle=":", alpha=0.5)

    fig.suptitle(f"ANP threshold sweep — {args.result_file}", fontsize=11)
    fig.tight_layout()
    out = record_dir / "anp_threshold_sweep.png"
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")

    knee = df.iloc[(df["test_asr"] + (1 - df["test_acc"])).argmin()]
    print(f"\nknee (minimizing ASR + (1-ACC)): t={knee['threshold']:.3f}, "
          f"ACC={knee['test_acc']:.4f}, ASR={knee['test_asr']:.4f}, RA={knee['test_ra']:.4f}")


if __name__ == "__main__":
    main()
