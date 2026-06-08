"""
데모 사전 검증 스크립트.
6개 공격 각각: attack_result.pt 로드 -> 모델 빌드 -> clean/bd 샘플 추론 -> ASR/ACC 확인.
이게 통과하면 gradio 데모가 안전하게 동작한다는 뜻.

실행: backdoor310 환경에서
    python demo/_verify.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.save_load_attack import load_attack_result

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 공격 이름 -> attack_result.pt 경로 (record/ 기준)
ATTACKS = {
    "WaNet":      "record/wanet_cifar10_fast/attack_result.pt",
    "InputAware": "record/inputaware_cifar10_fast/attack_result.pt",
    "BadNets":    "record/badnet_team/checkpoints/BADNET_attack_result.pt",
    "Blended":    "record/blended_team/checkpoints/BLENDED_attack_result.pt",
    "LF":         "record/lf_team/checkpoints/LF_attack_result.pt",
    "SIG":        "record/sig_team/checkpoints/SIG_attack_result.pt",
}

CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]

def main():
    print(f"DEVICE = {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    = {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    for name, rel in ATTACKS.items():
        t0 = time.time()
        try:
            raw = torch.load(rel, map_location="cpu", weights_only=False)
            num_classes = raw["num_classes"]
            res = load_attack_result(rel)
            model = generate_cls_model(res["model_name"], num_classes)
            model.load_state_dict(res["model"])
            model.to(DEVICE).eval()

            bd_ds = res["bd_test"]      # with_transform dataset
            cl_ds = res["clean_test"]   # with_transform dataset

            # 앞쪽 64개 샘플로 ACC / ASR 빠르게 측정
            N = 64
            def batch(ds, n):
                xs, ys = [], []
                for i in range(min(n, len(ds))):
                    item = ds[i]
                    x, y = item[0], item[1]
                    xs.append(x); ys.append(int(y))
                return torch.stack(xs).to(DEVICE), np.array(ys)

            xc, yc = batch(cl_ds, N)
            xb, yb = batch(bd_ds, N)
            with torch.no_grad():
                pc = model(xc).argmax(1).cpu().numpy()
                pb = model(xb).argmax(1).cpu().numpy()

            acc = (pc == yc).mean()
            # bd 예측 분포에서 최빈 클래스 = 타깃 추정
            target = np.bincount(pb, minlength=10).argmax()
            asr = (pb == target).mean()
            dt = time.time() - t0
            print(f"[OK] {name:11s} ACC(clean)={acc:5.1%}  "
                  f"target≈{target}({CIFAR10_CLASSES[target]})  "
                  f"ASR(bd→target)={asr:5.1%}  [{dt:4.1f}s]")
        except Exception as e:
            print(f"[FAIL] {name:11s} {type(e).__name__}: {e}")

    print("=" * 70)
    print("전부 [OK]면 gradio 데모 진행 가능.")

if __name__ == "__main__":
    main()
