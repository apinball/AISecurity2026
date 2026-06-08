"""
WaNet/IA: 기존 bd_test PNG(load_attack_result로 로드)로 백도어 모델 ASR 검증.
"""
import os, sys, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import torch
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.save_load_attack import load_attack_result

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

ATTACKS = {
    "WaNet":      "record/wanet_cifar10_fast/attack_result.pt",
    "InputAware": "record/inputaware_cifar10_fast/attack_result.pt",
}

def evaluate(name, rel, n=128):
    raw = torch.load(rel, map_location="cpu", weights_only=False)
    res = load_attack_result(rel)
    model = generate_cls_model(res["model_name"], raw["num_classes"])
    model.load_state_dict(res["model"]); model.to(DEVICE).eval()

    bd_ds = res["bd_test"]; cl_ds = res["clean_test"]
    def batch(ds, n):
        xs, ys = [], []
        for i in range(min(n, len(ds))):
            x, y = ds[i][0], ds[i][1]
            xs.append(x); ys.append(int(y))
        return torch.stack(xs).to(DEVICE), np.array(ys)
    xc, yc = batch(cl_ds, n); xb, yb = batch(bd_ds, n)
    with torch.no_grad():
        pc = model(xc).argmax(1).cpu().numpy(); pb = model(xb).argmax(1).cpu().numpy()
    acc = (pc == yc).mean()
    target = np.bincount(pb, minlength=10).argmax()
    asr = (pb == target).mean()
    print(f"[{name}] clean_ACC={acc:5.1%}  target={target}({CLASSES[target]})  ASR={asr:5.1%}  (n={n})")

if __name__ == "__main__":
    print(f"DEVICE={DEVICE}\n" + "=" * 60)
    for k, v in ATTACKS.items():
        evaluate(k, v)
    print("=" * 60)
