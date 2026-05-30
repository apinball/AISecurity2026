"""
Side-by-side comparison: clean CIFAR-10 test image vs WaNet bd_test image (same idx).
Run from project root:
    python analysis/compare_wanet_clean_vs_bd.py --result_file wanet_cifar10_fast --n 8
Output: record/<result>/clean_vs_bd_compare.png
"""
import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_file", type=str, required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    record_dir = Path("./record") / args.result_file
    bd_dir = record_dir / "bd_test_dataset"
    cross_dir = record_dir / "cross_test_dataset"
    out_path = record_dir / "clean_vs_bd_compare.png"

    cifar = torchvision.datasets.CIFAR10(root="./data/cifar10", train=False, download=False)

    bd_paths = []
    for cls_dir in sorted(bd_dir.iterdir()):
        if cls_dir.is_dir():
            bd_paths.extend(list(cls_dir.glob("*.png")))
    rng = random.Random(args.seed)
    sample = rng.sample(bd_paths, args.n)

    fig, axes = plt.subplots(3, args.n, figsize=(args.n * 2, 6.2))
    classes = cifar.classes

    for i, bd_p in enumerate(sample):
        idx = int(bd_p.stem)
        orig_img, orig_label = cifar[idx]
        bd_img = Image.open(bd_p).convert("RGB")

        cross_p = cross_dir / bd_p.parent.name / bd_p.name
        cross_img = Image.open(cross_p).convert("RGB") if cross_p.exists() else None

        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f"clean\nlabel={classes[orig_label]}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(bd_img)
        axes[1, i].set_title(f"WaNet bd\n→ target=0", fontsize=9)
        axes[1, i].axis("off")

        if cross_img is not None:
            axes[2, i].imshow(cross_img)
            axes[2, i].set_title("cross (noise)", fontsize=9)
        axes[2, i].axis("off")

        diff = (np.asarray(bd_img, dtype=np.int16) - np.asarray(orig_img, dtype=np.int16)).astype(np.int16)
        l_inf = int(np.max(np.abs(diff)))
        l_2 = float(np.linalg.norm(diff.astype(np.float32)) / np.sqrt(diff.size))
        axes[1, i].set_xlabel(f"L∞={l_inf} L2={l_2:.2f}", fontsize=8)
        axes[1, i].xaxis.set_visible(True)
        axes[1, i].set_xticks([])

    fig.suptitle(f"WaNet imperceptibility — {args.result_file}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
