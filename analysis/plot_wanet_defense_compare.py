"""
Compare WaNet against multiple defenses. Reads each defense's *_df_summary.csv,
plots a grouped bar chart of test_acc / test_asr / test_ra.
Run from project root:
    python analysis/plot_wanet_defense_compare.py --result_file wanet_cifar10_fast
Output: record/<result>/defense_compare.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_file", type=str, required=True)
    p.add_argument("--defenses", nargs="+", default=["ft", "nad", "anp", "abl", "nc"])
    args = p.parse_args()

    record_dir = Path("./record") / args.result_file
    rows = []

    attack_csv = record_dir / "attack_df_summary.csv"
    df = pd.read_csv(attack_csv)
    last = df[df.iloc[:, 0] == "last"].iloc[0]
    rows.append(("no defense", float(last["test_acc"]), float(last["test_asr"]), float(last["test_ra"])))

    for d in args.defenses:
        ddir = record_dir / "defense" / d
        cands = list(ddir.glob("*_df_summary.csv"))
        if not cands:
            print(f"[skip] {d}: no summary csv at {ddir}")
            continue
        df = pd.read_csv(cands[0])
        last_rows = df[df.iloc[:, 0] == "last"]
        target = last_rows.iloc[0] if len(last_rows) else df.iloc[-1]
        rows.append((d.upper(), float(target["test_acc"]), float(target["test_asr"]), float(target["test_ra"])))

    labels = [r[0] for r in rows]
    acc = np.array([r[1] for r in rows])
    asr = np.array([r[2] for r in rows])
    ra = np.array([r[3] for r in rows])

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars_acc = ax.bar(x - w, acc, w, label="clean ACC", color="#3b82f6")
    bars_asr = ax.bar(x, asr, w, label="ASR (lower=better)", color="#ef4444")
    bars_ra = ax.bar(x + w, ra, w, label="RA (recovery)", color="#10b981")

    for bars in (bars_acc, bars_asr, bars_ra):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("metric")
    ax.set_title(f"WaNet on CIFAR-10 — defense comparison ({args.result_file})")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right")

    out = record_dir / "defense_compare.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")

    print("\n| defense | acc    | asr    | ra     |")
    print("|---------|--------|--------|--------|")
    for lbl, a, s, r in rows:
        print(f"| {lbl:<7} | {a:.4f} | {s:.4f} | {r:.4f} |")


if __name__ == "__main__":
    main()
