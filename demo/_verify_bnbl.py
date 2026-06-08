"""
BadNets/Blended 트리거를 재생성해서 백도어 모델에 ASR이 나오는지 검증.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import imageio.v2 as imageio
from PIL import Image

from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.bd_img_transform.patch import AddMaskPatchTrigger
from utils.bd_img_transform.blended import blendedImageAttack

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
NORM_MEAN = [0.4914, 0.4822, 0.4465]; NORM_STD = [0.247, 0.243, 0.261]
normalize = T.Normalize(NORM_MEAN, NORM_STD); to_tensor = T.ToTensor()
ATTACK_TARGET = 0

def build_model(pt_path):
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    model = generate_cls_model(raw["model_name"], raw["num_classes"])
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in raw["model"].items()}
    model.load_state_dict(sd); model.to(DEVICE).eval()
    return model

# --- BadNets: trigger_image.png 마스크 패치 ---
_mask = np.array(Image.open("./resource/badnet/trigger_image.png").resize((32, 32)))
_badnet = AddMaskPatchTrigger(_mask)
def badnet_trigger(pil_img):
    arr = np.array(pil_img.resize((32, 32)))
    out = _badnet(arr)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

# --- Blended: hello_kitty 알파 0.2 ---
_trans = T.Compose([T.ToPILImage(), T.Resize((32, 32)), T.ToTensor()])
_kitty = _trans(imageio.imread("./resource/blended/hello_kitty.jpeg")).cpu().numpy().transpose(1, 2, 0) * 255
_blended = blendedImageAttack(_kitty, 0.2)
def blended_trigger(pil_img):
    arr = np.array(pil_img.resize((32, 32))).astype(np.float32)
    out = _blended(arr)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

def evaluate(name, pt_path, trigger_fn, n=128):
    model = build_model(pt_path)
    ds = torchvision.datasets.CIFAR10(root="./data/cifar10", train=False, download=False)
    cc = ht = total = 0
    for i in range(len(ds)):
        if total >= n: break
        img, label = ds[i]
        if label == ATTACK_TARGET: continue
        xc = normalize(to_tensor(img.resize((32, 32)))).unsqueeze(0).to(DEVICE)
        xb = normalize(to_tensor(trigger_fn(img))).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pc = model(xc).argmax(1).item(); pb = model(xb).argmax(1).item()
        cc += int(pc == label); ht += int(pb == ATTACK_TARGET); total += 1
    print(f"[{name}] clean_ACC={cc/total:5.1%}  ASR(bd->{CLASSES[ATTACK_TARGET]})={ht/total:5.1%}  (n={total})")

if __name__ == "__main__":
    print(f"DEVICE={DEVICE}\n" + "=" * 60)
    evaluate("BadNets", "record/badnet_team/checkpoints/BADNET_attack_result.pt", badnet_trigger)
    evaluate("Blended", "record/blended_team/checkpoints/BLENDED_attack_result.pt", blended_trigger)
    print("=" * 60)
