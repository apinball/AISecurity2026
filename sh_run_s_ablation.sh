#!/usr/bin/env bash
# WaNet warping-strength `s` ablation: 5 trainings at s in {0.1, 0.3, 0.5, 0.7, 1.0}
# 30 epochs each, otherwise identical hparams to the main run.

set -e
cd "$(dirname "$0")"

for S in 0.1 0.3 0.5 0.7 1.0; do
    NAME="wanet_s${S}_e30"
    OUT="record/${NAME}"
    if [ -d "$OUT" ]; then
        echo "[skip] $OUT already exists"
        continue
    fi
    echo "[run]  s=$S → $OUT"
    python attack/wanet.py \
      --yaml_path ./config/attack/prototype/cifar10.yaml \
      --bd_yaml_path ./config/attack/wanet/default.yaml \
      --dataset cifar10 \
      --amp True --batch_size 256 \
      --num_workers 0 --pin_memory True \
      --epochs 30 \
      --s "$S" \
      --save_folder_name "$NAME"
done
echo "[done] all s ablation runs finished"
