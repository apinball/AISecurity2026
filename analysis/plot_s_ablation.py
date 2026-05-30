"""
WaNet warping-strength `s` ablation analysis.
For each s in {0.1, 0.3, 0.5, 0.7, 1.0}:
  - read test_acc / test_asr / test_ra from the run's summary csv
  - compute mean L2 perturbation between bd_test PNG and the original CIFAR-10 test image

Output:
  - 3-panel summary plot
  - aggregated csv

Run from project root:
    python analysis/plot_s_ablation.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision
from PIL import Image


S_VALUES = [0.1, 0.3, 0.5, 0.7, 1.0]


def mean_l2_per_pixel(record_dir: Path, cifar, max_per_class: int = 50) -> float:
    bd_dir = record_dir / "bd_test_dataset"
    if not bd_dir.exists():
        return float("nan")
    diffs = []
    for cls_dir in sorted(bd_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        png_files = sorted(cls_dir.glob("*.png"))[:max_per_class]
        for bd_p in png_files:
            try:
                idx = int(bd_p.stem)
                orig_img, _ = cifar[idx]
                bd_img = Image.open(bd_p).convert("RGB")
                diff = np.asarray(bd_img, dtype=np.float32) - np.asarray(orig_img, dtype=np.float32)
                diffs.append(float(np.sqrt((diff * diff).mean())))
            except Exception:
                continue
    return float(np.mean(diffs)) if diffs else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="./record")
    args = p.parse_args()

    cifar = torchvision.datasets.CIFAR10(root="./data/cifar10", train=False, download=False)
    out_dir = Path(args.out_dir)

    rows = []
    for s in S_VALUES:
        run_dir = out_dir / f"wanet_s{s}_e30"
        csv = run_dir / "attack_df_summary.csv"
        if not csv.exists():
            print(f"[skip] missing {csv}")
            continue
        df = pd.read_csv(csv)
        last = df[df.iloc[:, 0] == "last"].iloc[0]
        l2 = mean_l2_per_pixel(run_dir, cifar)
        rows.append({
            "s": s,
            "test_acc": float(last["test_acc"]),
            "test_asr": float(last["test_asr"]),
            "test_ra":  float(last["test_ra"]),
            "test_cross_acc": float(last["test_cross_acc"]),
            "bd_l2_per_pixel": l2,
        })
        print(f"s={s:.1f} | acc={rows[-1]['test_acc']:.4f}  asr={rows[-1]['test_asr']:.4f}  "
              f"ra={rows[-1]['test_ra']:.4f}  L2/pix={l2:.3f}")

    df = pd.DataFrame(rows)
    agg_csv = out_dir / "s_ablation_summary.csv"
    df.to_csv(agg_csv, index=False)
    print(f"saved: {agg_csv}")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    axes[0].plot(df["s"], df["test_acc"], marker="o", color="#3b82f6", label="clean ACC")
    axes[0].plot(df["s"], df["test_asr"], marker="s", color="#ef4444", label="ASR")
    axes[0].plot(df["s"], df["test_ra"],  marker="^", color="#10b981", label="RA")
    axes[0].plot(df["s"], df["test_cross_acc"], marker="d", color="#a855f7",
                 label="cross ACC", linestyle="--", alpha=0.7)
    axes[0].set_xlabel("warping strength  s")
    axes[0].set_ylabel("metric")
    axes[0].set_title("Effectiveness vs s")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(linestyle=":", alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].plot(df["s"], df["bd_l2_per_pixel"], marker="o", color="#0ea5e9")
    for _, r in df.iterrows():
        axes[1].annotate(f"{r['bd_l2_per_pixel']:.2f}",
                         (r["s"], r["bd_l2_per_pixel"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=8)
    axes[1].set_xlabel("warping strength  s")
    axes[1].set_ylabel("mean L2 per pixel  (bd vs clean)")
    axes[1].set_title("Visual perturbation vs s")
    axes[1].grid(linestyle=":", alpha=0.5)

    axes[2].plot(df["bd_l2_per_pixel"], df["test_asr"], marker="o",
                 color="#dc2626", linestyle="-")
    for _, r in df.iterrows():
        axes[2].annotate(f"s={r['s']}",
                         (r["bd_l2_per_pixel"], r["test_asr"]),
                         textcoords="offset points", xytext=(6, -4), fontsize=8)
    axes[2].set_xlabel("mean L2 per pixel  (bd vs clean)")
    axes[2].set_ylabel("ASR")
    axes[2].set_title("Stealth vs Effectiveness (Pareto)")
    axes[2].set_ylim(0, 1.02)
    axes[2].grid(linestyle=":", alpha=0.5)

    fig.suptitle("WaNet  s  ablation — 30 epochs / CIFAR-10 / PreActResNet18", fontsize=11)
    fig.tight_layout()
    out = out_dir / "s_ablation.png"
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
