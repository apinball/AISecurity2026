"""
Per-layer relative weight change between attack model and NAD-purified model.
Shows where NAD actually modifies weights, which is indirect evidence for
where the backdoor lives in the network.

delta_l = ||W_l^attack - W_l^NAD||_F / ||W_l^attack||_F

Run from project root:
    python analysis/plot_layer_weight_delta.py --result_file wanet_cifar10_fast
"""
import argparse
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


LAYER_GROUPS = [
    ("conv1",   r"^conv1\."),
    ("layer1",  r"^layer1\."),
    ("layer2",  r"^layer2\."),
    ("layer3",  r"^layer3\."),
    ("layer4",  r"^layer4\."),
    ("linear",  r"^linear\."),
]


def group_of(name):
    for g, pat in LAYER_GROUPS:
        if re.match(pat, name):
            return g
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_file", type=str, required=True)
    p.add_argument("--defense", type=str, default="nad")
    args = p.parse_args()

    record_dir = Path("./record") / args.result_file
    a = torch.load(record_dir / "attack_result.pt",
                   map_location="cpu", weights_only=False)["model"]
    n = torch.load(record_dir / "defense" / args.defense / "defense_result.pt",
                   map_location="cpu", weights_only=False)["model"]

    per_param = []
    for name, w_a in a.items():
        if name not in n:
            continue
        w_n = n[name]
        if not torch.is_floating_point(w_a):
            continue
        if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
            continue
        diff = (w_a - w_n).flatten().float()
        denom = w_a.flatten().float().norm().item() + 1e-12
        rel = diff.norm().item() / denom
        per_param.append((name, rel, w_a.numel()))

    groups = OrderedDict((g, {"num": 0.0, "den": 0.0, "n_params": 0}) for g, _ in LAYER_GROUPS)
    for name, rel, n_w in per_param:
        g = group_of(name)
        if g is None:
            continue
        w_a = a[name].flatten().float()
        w_n = n[name].flatten().float()
        num = ((w_a - w_n).norm() ** 2).item()
        den = (w_a.norm() ** 2).item()
        groups[g]["num"] += num
        groups[g]["den"] += den
        groups[g]["n_params"] += n_w

    group_names, group_rels, group_counts = [], [], []
    for g, d in groups.items():
        if d["den"] < 1e-12:
            continue
        group_names.append(g)
        group_rels.append(np.sqrt(d["num"] / d["den"]))
        group_counts.append(d["n_params"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(group_names)))
    bars = axes[0].bar(group_names, group_rels, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, group_rels):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("relative L2 change  ||ΔW|| / ||W_attack||")
    axes[0].set_title(f"Per-stage weight delta — attack vs {args.defense.upper()}")
    axes[0].grid(axis="y", linestyle=":", alpha=0.4)

    per_param.sort(key=lambda x: -x[1])
    top = per_param[:20]
    names = [x[0] for x in top]
    vals  = [x[1] for x in top]
    axes[1].barh(range(len(names)), vals[::-1], color="#dc2626", edgecolor="black", linewidth=0.3)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names[::-1], fontsize=7)
    axes[1].set_xlabel("relative L2 change")
    axes[1].set_title("Top-20 most-changed parameters")
    axes[1].grid(axis="x", linestyle=":", alpha=0.4)

    fig.suptitle(f"Where does {args.defense.upper()} modify the network? — {args.result_file}", fontsize=11)
    fig.tight_layout()
    out = record_dir / f"layer_weight_delta_{args.defense}.png"
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")

    print("\nPer-stage relative L2 change:")
    for g, v, c in zip(group_names, group_rels, group_counts):
        print(f"  {g:<8} | rel ΔW = {v:.4f} | params = {c:>9,}")


if __name__ == "__main__":
    main()
