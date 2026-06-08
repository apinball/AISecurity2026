"""
LF/SIG 트리거를 재생성해서 실제 백도어 모델에 먹였을 때 ASR이 나오는지 검증.
이게 통과해야 데모에서 LF/SIG를 '실시간 트리거 생성'으로 보여줄 수 있다.

실행: python demo/_verify_lfsig.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_img_transform.sig import sigTriggerAttack
from utils.bd_img_transform.patch import SimpleAdditiveTrigger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# CIFAR-10 정규화 (dataset_and_transform_generate.py 확인값)
NORM_MEAN = [0.4914, 0.4822, 0.4465]
NORM_STD  = [0.247, 0.243, 0.261]
normalize = T.Normalize(NORM_MEAN, NORM_STD)
to_tensor = T.ToTensor()

ATTACK_TARGET = 0  # SIG/LF 둘 다 attack_target=0 (config 확인)

def build_model(pt_path):
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    model = generate_cls_model(raw["model_name"], raw["num_classes"])
    # state_dict 키 정리 (module. 제거)
    sd = raw["model"]
    new = {}
    for k, v in sd.items():
        new[k[7:] if k.startswith("module.") else k] = v
    model.load_state_dict(new)
    model.to(DEVICE).eval()
    return model

def sig_trigger(pil_img):
    """bd_attack_generate.py의 sig 파이프라인 재현: Resize→np.array→sigTrigger→clip uint8→PIL"""
    trans = sigTriggerAttack(delta=40, f=6)
    arr = np.array(pil_img.resize((32, 32)))
    out = trans(arr)  # 내부에서 clip+uint8
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def lf_trigger(pil_img):
    """lowFrequency 파이프라인 재현: SimpleAdditiveTrigger(패턴 npy)"""
    patt = np.load("./resource/lowFrequency/cifar10_preactresnet18_0_255.npy")
    if patt.ndim == 4:
        patt = patt[0]
    elif patt.ndim == 2:
        patt = np.stack((patt,) * 3, axis=-1)
    adder = SimpleAdditiveTrigger(trigger_array=patt)
    arr = np.array(pil_img.resize((32, 32)))
    out = adder(arr)  # np.clip 내장
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def evaluate(name, pt_path, trigger_fn, n=128):
    model = build_model(pt_path)
    # CIFAR-10 test set (clean 원본)
    ds = torchvision.datasets.CIFAR10(root="./data/cifar10", train=False, download=False)
    clean_correct = 0
    hit_target = 0
    total = 0
    # 타깃 클래스(0=airplane)가 아닌 샘플만 ASR 대상
    for i in range(len(ds)):
        if total >= n:
            break
        img, label = ds[i]  # PIL, int
        if label == ATTACK_TARGET:
            continue
        # clean 추론
        xc = normalize(to_tensor(img.resize((32, 32)))).unsqueeze(0).to(DEVICE)
        # bd 추론
        bd = trigger_fn(img)
        xb = normalize(to_tensor(bd)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pc = model(xc).argmax(1).item()
            pb = model(xb).argmax(1).item()
        clean_correct += int(pc == label)
        hit_target += int(pb == ATTACK_TARGET)
        total += 1
    acc = clean_correct / total
    asr = hit_target / total
    print(f"[{name}] clean_ACC={acc:5.1%}  ASR(bd->{CLASSES[ATTACK_TARGET]})={asr:5.1%}  (n={total})")
    return asr

if __name__ == "__main__":
    print(f"DEVICE={DEVICE}")
    print("=" * 60)
    evaluate("SIG", "record/sig_team/checkpoints/SIG_attack_result.pt", sig_trigger)
    evaluate("LF",  "record/lf_team/checkpoints/LF_attack_result.pt",   lf_trigger)
    print("=" * 60)
    print("ASR이 높으면(>80%) 재생성 트리거가 정확. 데모 진행 가능.")
