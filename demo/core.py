"""
데모 추론 코어. 6개 공격 + 5개 방어 모델 로딩, 트리거 생성, 추론, Grad-CAM.
검증 스크립트(_verify_*.py)에서 ASR 통과가 확인된 로직만 사용.

모델은 lazy-load + 캐시. 첫 호출 때만 디스크에서 읽고 이후 메모리 재사용.
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
from utils.save_load_attack import load_attack_result
from utils.bd_img_transform.sig import sigTriggerAttack
from utils.bd_img_transform.patch import SimpleAdditiveTrigger, AddMaskPatchTrigger
from utils.bd_img_transform.blended import blendedImageAttack

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
ATTACK_TARGET = 0  # 6개 공격 모두 all2one target=0 (config 확인)

NORM_MEAN = [0.4914, 0.4822, 0.4465]
NORM_STD = [0.247, 0.243, 0.261]
_normalize = T.Normalize(NORM_MEAN, NORM_STD)
_to_tensor = T.ToTensor()

# ---------------------------------------------------------------------------
# 공격/방어 가중치 경로 맵 (record/ 기준 상대경로)
# ---------------------------------------------------------------------------
ATTACK_PT = {
    "WaNet":      "record/wanet_cifar10_fast/attack_result.pt",
    "InputAware": "record/inputaware_cifar10_fast/attack_result.pt",
    "BadNets":    "record/badnet_team/checkpoints/BADNET_attack_result.pt",
    "Blended":    "record/blended_team/checkpoints/BLENDED_attack_result.pt",
    "LF":         "record/lf_team/checkpoints/LF_attack_result.pt",
    "SIG":        "record/sig_team/checkpoints/SIG_attack_result.pt",
}

# 방어 모델 경로 (공격 -> 방어 -> defense_result.pt). 일부 네이밍이 팀마다 달라 개별 지정.
DEFENSE_PT = {
    "WaNet": {d: f"record/wanet_cifar10_fast/defense/{d.lower()}/defense_result.pt"
              for d in ["FT", "NAD", "ANP", "ABL", "NC"]},
    "InputAware": {d: f"record/inputaware_cifar10_fast/defense/{d.lower()}/defense_result.pt"
                   for d in ["FT", "NAD", "ANP", "ABL", "NC"]},
    "BadNets": {d: f"record/badnet_team/checkpoints/BADNET_{d}_defense_result.pt"
                for d in ["FT", "NAD", "ANP", "ABL", "NC"]},
    "Blended": {d: f"record/blended_team/checkpoints/BLENDED_{d}_defense_result.pt"
                for d in ["FT", "NAD", "ANP", "ABL", "NC"]},
    "SIG": {d: f"record/sig_team/checkpoints/SIG_{d}_defense_result.pt"
            for d in ["FT", "NAD", "ANP", "ABL", "NC"]},
    "LF": {
        "FT":  "record/lf_team/checkpoints/LF_defense_FT(Fine-tuning)_result.pt",
        "NAD": "record/lf_team/checkpoints/LF_defense_NAD(Neural_Attention_Distillation)_result.pt",
        "ANP": "record/lf_team/checkpoints/LF_defense_ANP(Adversarial_Neuron_Pruning)_result.pt",
        "ABL": "record/lf_team/checkpoints/LF_defense_ABL(Anti-Backdoor_Learning)_result.pt",
        "NC":  "record/lf_team/checkpoints/LF_defense_NC(Neural_Cleanse)_result.pt",
    },
}

DEFENSES = ["FT", "NAD", "ANP", "ABL", "NC"]
# 실시간 '복구' 재현이 가능한 방어 (정화 모델 가중치 저장됨)
REPAIR_DEFENSES = ["FT", "NAD", "ANP", "ABL"]
# NC는 '탐지' 방어 — 정화 모델 미저장, 역공학 트리거(mask/pattern)만 남음

# NC 역공학 트리거 PNG 경로 (있는 공격만). log/0/pattern.png = 타깃 클래스(0) 복원 트리거
NC_TRIGGER_PNG = {
    "WaNet":      "record/wanet_cifar10_fast/defense/nc/log/0/pattern.png",
    "InputAware": "record/inputaware_cifar10_fast/defense/nc/log/0/pattern.png",
}
# 팀 기록 CSV 기준 NC test_asr (실시간 재현 불가, 기록값)
NC_RECORDED_ASR = {
    "BadNets": 0.007, "Blended": 0.999,
    # LF/SIG는 record summary에서 채움 (없으면 None)
}

# 공격별 트리거 방식: 재생성(regen) or 기존 PNG(dataset)
REGEN_ATTACKS = ["BadNets", "Blended", "LF", "SIG"]
DATASET_ATTACKS = ["WaNet", "InputAware"]

# ---------------------------------------------------------------------------
# 캐시
# ---------------------------------------------------------------------------
_model_cache = {}      # key -> nn.Module
_bd_sample_cache = {}  # attack -> list[(clean_pil, bd_pil, label)] (WaNet/IA 용)
_cifar_test = None


def _abs(rel):
    return os.path.join(ROOT, rel)


def get_cifar_test():
    global _cifar_test
    if _cifar_test is None:
        _cifar_test = torchvision.datasets.CIFAR10(
            root=_abs("data/cifar10"), train=False, download=False)
    return _cifar_test


def _load_state_dict_clean(sd):
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def get_attack_model(attack):
    """공격(백도어) 모델 로드 + 캐시."""
    key = f"attack::{attack}"
    if key not in _model_cache:
        raw = torch.load(_abs(ATTACK_PT[attack]), map_location="cpu", weights_only=False)
        model = generate_cls_model(raw["model_name"], raw["num_classes"])
        model.load_state_dict(_load_state_dict_clean(raw["model"]))
        model.to(DEVICE).eval()
        _model_cache[key] = model
    return _model_cache[key]


def get_defense_model(attack, defense):
    """방어 모델 로드 + 캐시. defense_result.pt는 {model_name, num_classes, model}."""
    key = f"defense::{attack}::{defense}"
    if key not in _model_cache:
        raw = torch.load(_abs(DEFENSE_PT[attack][defense]), map_location="cpu", weights_only=False)
        nc = raw.get("num_classes", 10)
        mn = raw.get("model_name", "preactresnet18")
        model = generate_cls_model(mn, nc)
        model.load_state_dict(_load_state_dict_clean(raw["model"]))
        model.to(DEVICE).eval()
        _model_cache[key] = model
    return _model_cache[key]


# ---------------------------------------------------------------------------
# 트리거 생성기 (재생성 공격 4종) — 검증 통과 로직
# ---------------------------------------------------------------------------
_sig = sigTriggerAttack(delta=40, f=6)

_lf_patt = None
def _get_lf_adder():
    global _lf_patt
    if _lf_patt is None:
        patt = np.load(_abs("resource/lowFrequency/cifar10_preactresnet18_0_255.npy"))
        if patt.ndim == 4:
            patt = patt[0]
        elif patt.ndim == 2:
            patt = np.stack((patt,) * 3, axis=-1)
        _lf_patt = SimpleAdditiveTrigger(trigger_array=patt)
    return _lf_patt

_badnet_mask = None
def _get_badnet():
    global _badnet_mask
    if _badnet_mask is None:
        mask = np.array(Image.open(_abs("resource/badnet/trigger_image.png")).resize((32, 32)))
        _badnet_mask = AddMaskPatchTrigger(mask)
    return _badnet_mask

_blended_attacker = None
def _get_blended():
    global _blended_attacker
    if _blended_attacker is None:
        trans = T.Compose([T.ToPILImage(), T.Resize((32, 32)), T.ToTensor()])
        kitty = trans(imageio.imread(_abs("resource/blended/hello_kitty.jpeg"))
                      ).cpu().numpy().transpose(1, 2, 0) * 255
        _blended_attacker = blendedImageAttack(kitty, 0.2)
    return _blended_attacker


def apply_trigger(attack, pil_img):
    """clean PIL(32x32 RGB) -> triggered PIL. 재생성 공격 4종만 지원."""
    arr = np.array(pil_img.resize((32, 32)))
    if attack == "SIG":
        out = _sig(arr)
    elif attack == "LF":
        out = _get_lf_adder()(arr)
    elif attack == "BadNets":
        out = _get_badnet()(arr)
    elif attack == "Blended":
        out = _get_blended()(arr.astype(np.float32))
    else:
        raise ValueError(f"{attack}는 재생성 트리거 미지원 (dataset 방식)")
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# 강도 슬라이더 지원 공격과 (파라미터명, 기본값, 최소, 최대, step)
STRENGTH_PARAM = {
    "SIG":     ("진폭(delta)", 40, 5, 80, 5),     # sigTriggerAttack delta
    "Blended": ("블렌딩(alpha)", 0.20, 0.05, 0.6, 0.05),  # blendedImageAttack rate
}

_kitty_cache = None
def _kitty_image():
    global _kitty_cache
    if _kitty_cache is None:
        trans = T.Compose([T.ToPILImage(), T.Resize((32, 32)), T.ToTensor()])
        _kitty_cache = trans(imageio.imread(_abs("resource/blended/hello_kitty.jpeg"))
                             ).cpu().numpy().transpose(1, 2, 0) * 255
    return _kitty_cache


def apply_trigger_strength(attack, pil_img, strength):
    """강도 파라미터를 받아 트리거 적용. SIG=delta, Blended=alpha.
    그 외 공격은 기본 apply_trigger로 폴백."""
    arr = np.array(pil_img.resize((32, 32)))
    if attack == "SIG":
        out = sigTriggerAttack(delta=float(strength), f=6)(arr)
    elif attack == "Blended":
        out = blendedImageAttack(_kitty_image(), float(strength))(arr.astype(np.float32))
    else:
        return apply_trigger(attack, pil_img)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def random_clean_sample(seed_offset=0):
    """CIFAR test에서 non-target 라벨의 임의 원본 1개 반환 (clean_pil, label).
    Math.random 불가 환경 대비: 캐시 카운터 + offset으로 의사난수 인덱스."""
    ds = get_cifar_test()
    n = len(ds)
    # 호출마다 달라지도록 내부 카운터 사용
    idx = (random_clean_sample._ctr * 2654435761 + seed_offset * 40503) % n
    random_clean_sample._ctr += 1
    # non-target 보장
    for _ in range(n):
        img, label = ds[idx]
        if label != ATTACK_TARGET:
            return img.resize((32, 32)), label
        idx = (idx + 1) % n
    img, label = ds[0]
    return img.resize((32, 32)), label
random_clean_sample._ctr = 1


# dataset 공격(WaNet/IA)의 bd_test 전체 풀 캐시
_full_bd_cache = {}
def _get_full_bd(attack):
    if attack not in _full_bd_cache:
        res = load_attack_result(ATTACK_PT[attack])
        bd = res["bd_test"]
        oia = np.array(bd.wrapped_dataset.original_index_array)
        _full_bd_cache[attack] = (bd, oia, res["model_name"])
    return _full_bd_cache[attack]


def random_dataset_sample(attack):
    """WaNet/IA: bd_test 전체 9000장 풀에서 진짜 랜덤으로 1장.
    동일 원본의 clean과 짝지어 (clean_pil, bd_pil, label, clean_in, bd_in) 반환."""
    bd, oia, _ = _get_full_bd(attack)
    raw = get_cifar_test()
    n = len(bd)
    denorm = T.Normalize(
        mean=[-m / s for m, s in zip(NORM_MEAN, NORM_STD)],
        std=[1 / s for s in NORM_STD],
    )
    def to_pil(x):
        return T.ToPILImage()(denorm(x).clamp(0, 1))
    # non-target 원본이 나올 때까지 의사난수 인덱스
    for _ in range(n):
        i = (random_clean_sample._ctr * 2654435761) % n
        random_clean_sample._ctr += 1
        raw_idx = int(oia[i])
        _, raw_label = raw[raw_idx]
        if raw_label == ATTACK_TARGET:
            continue
        clean_pil = raw[raw_idx][0].resize((32, 32))
        xc = _normalize(_to_tensor(clean_pil)).to(DEVICE)
        xb = bd[i][0].to(DEVICE)
        return clean_pil, to_pil(xb), raw_label, xc, xb
    # fallback
    i = 0
    raw_idx = int(oia[i])
    clean_pil = raw[raw_idx][0].resize((32, 32))
    xc = _normalize(_to_tensor(clean_pil)).to(DEVICE)
    xb = bd[i][0].to(DEVICE)
    return clean_pil, to_pil(xb), int(raw[raw_idx][1]), xc, xb


def prewarm(progress=None):
    """시연 전 전체 예열: 6공격 모델 + 24방어 모델 + 갤러리 샘플 + ASR표 미리 로딩.
    시연 중 첫 클릭 멈춤 제거. 반환: 로그 문자열 리스트."""
    logs = []
    attacks = list(ATTACK_PT.keys())
    for ai, atk in enumerate(attacks):
        get_attack_model(atk)
        # 갤러리 샘플 미리 선별
        if atk in REGEN_ATTACKS:
            get_regen_samples(atk, k=8)
        else:
            get_dataset_samples(atk, k=8)
        # 방어 모델 + ASR 미리 계산
        measure_asr(atk, None, n=40)
        for d in REPAIR_DEFENSES:
            get_defense_model(atk, d)
            measure_asr(atk, d, n=40)
        msg = f"예열 {ai+1}/{len(attacks)}: {atk} 완료"
        logs.append(msg)
        if progress is not None:
            progress((ai + 1) / len(attacks), desc=msg)
    return logs


def get_regen_samples(attack, k=12):
    """재생성 공격(BadNets/Blended/LF/SIG): CIFAR test 원본에서
    'non-target 라벨 + 트리거 시 타깃으로 바뀌는' clean PIL 샘플 선별 + 캐시."""
    ck = f"regen::{attack}"
    if ck in _bd_sample_cache:
        return _bd_sample_cache[ck]
    ds = get_cifar_test()
    model = get_attack_model(attack)
    samples = []
    for i in range(len(ds)):
        if len(samples) >= k:
            break
        img, label = ds[i]
        if label == ATTACK_TARGET:
            continue
        clean = img.resize((32, 32))
        bd = apply_trigger(attack, img)
        pc, _ = predict(model, clean)
        pb, _ = predict(model, bd)
        if pc == label and pb == ATTACK_TARGET:
            samples.append((clean, bd, label))
    if not samples:  # fallback
        for i in range(len(ds)):
            img, label = ds[i]
            if label == ATTACK_TARGET:
                continue
            samples.append((img.resize((32, 32)), apply_trigger(attack, img), label))
            if len(samples) >= k:
                break
    _bd_sample_cache[ck] = samples
    return samples


def get_dataset_samples(attack, k=12):
    """WaNet/IA: bd_test의 각 샘플과 '동일 원본'의 clean을 짝지어 추출 + 캐시.

    핵심: bd_test(9000개)는 타깃 클래스를 제외해 clean_test(10000개)와 인덱스가
    어긋난다. 그래서 clean을 clean_test[i]로 가져오면 다른 이미지가 된다.
    bd_test.wrapped_dataset.original_index_array[i] = 원본 CIFAR test 인덱스이므로,
    이걸로 raw CIFAR에서 같은 원본을 가져와 clean으로 쓴다(트리거 유무만 차이)."""
    if attack in _bd_sample_cache:
        return _bd_sample_cache[attack]
    res = load_attack_result(ATTACK_PT[attack])
    bd_ds = res["bd_test"]
    oia = np.array(bd_ds.wrapped_dataset.original_index_array)  # bd[i] -> raw idx
    raw = get_cifar_test()
    model = get_attack_model(attack)
    denorm = T.Normalize(
        mean=[-m / s for m, s in zip(NORM_MEAN, NORM_STD)],
        std=[1 / s for s in NORM_STD],
    )
    def to_pil(x):
        return T.ToPILImage()(denorm(x).clamp(0, 1))

    samples = []
    fallback = []
    for i in range(len(bd_ds)):
        if len(samples) >= k:
            break
        raw_idx = int(oia[i])
        raw_img, raw_label = raw[raw_idx]      # 원본 PIL, 라벨
        if raw_label == ATTACK_TARGET:          # 타깃 클래스 원본은 데모 효과 없음
            continue
        # clean = 동일 원본을 정규화한 텐서 / bd = bd_test의 트리거 적용 텐서
        clean_pil = raw_img.resize((32, 32))
        xc = _normalize(_to_tensor(clean_pil)).to(DEVICE)
        xb = bd_ds[i][0].to(DEVICE)
        pc, _ = predict(model, xc)
        pb, _ = predict(model, xb)
        entry = (clean_pil, to_pil(xb), raw_label, xc, xb)
        if pc == raw_label and pb == ATTACK_TARGET:
            samples.append(entry)
        elif len(fallback) < k:
            fallback.append(entry)
    if not samples:
        samples = fallback
    _bd_sample_cache[attack] = samples
    return samples


# ---------------------------------------------------------------------------
# 추론
# ---------------------------------------------------------------------------
_asr_cache = {}
def measure_asr(attack, defense, n=40):
    """공격 또는 방어 모델에서 n개 샘플 ASR(타깃 적중률) 측정 + 캐시.
    defense=None이면 공격(백도어) 모델."""
    ck = f"{attack}::{defense}::{n}"
    if ck in _asr_cache:
        return _asr_cache[ck]
    model = get_attack_model(attack) if defense in (None, "없음") \
        else get_defense_model(attack, defense)
    if attack in REGEN_ATTACKS:
        samples = get_regen_samples(attack, k=n)
        pairs = [(bd, lab) for (_, bd, lab) in samples]
    else:
        s = get_dataset_samples(attack, k=n)
        pairs = [(xb, lab) for (_, _, lab, _, xb) in s]
    hit = 0
    for bd_in, _ in pairs:
        pb, _ = predict(model, bd_in)
        hit += int(pb == ATTACK_TARGET)
    asr = hit / max(len(pairs), 1)
    _asr_cache[ck] = asr
    return asr


def _to_input(pil_or_tensor):
    if isinstance(pil_or_tensor, torch.Tensor):
        return pil_or_tensor.unsqueeze(0).to(DEVICE)
    return _normalize(_to_tensor(pil_or_tensor.resize((32, 32)))).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def predict(model, pil_or_tensor):
    """returns (pred_idx, prob_dict{class:prob})"""
    x = _to_input(pil_or_tensor)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return int(probs.argmax()), {CLASSES[i]: float(probs[i]) for i in range(10)}


def nc_detection(attack):
    """NC(탐지 방어) 결과. (역공학 트리거 PIL or None, 설명 markdown)."""
    img = None
    rel = NC_TRIGGER_PNG.get(attack)
    if rel and os.path.exists(_abs(rel)):
        img = Image.open(_abs(rel)).convert("RGB").resize((128, 128), Image.NEAREST)
    # NC 정화모델이 공격모델과 동일한지 실측(가중치 합 비교)
    same = None
    try:
        am = get_attack_model(attack)
        dm = get_defense_model(attack, "NC")
        a = am.state_dict()["linear.weight"].sum().item()
        d = dm.state_dict()["linear.weight"].sum().item()
        same = abs(a - d) < 1e-6
    except Exception:
        pass
    rec = NC_RECORDED_ASR.get(attack)
    md = "### 🔍 NC (Neural Cleanse) — 탐지 방어\n"
    md += ("NC는 가중치를 고치는 '복구'가 아니라, 각 클래스로 오분류시키는 "
           "**최소 트리거를 역공학**해 백도어 존재를 탐지하는 방어입니다.\n")
    if img is not None:
        md += "- 좌측: NC가 역공학으로 복원한 **타깃 클래스 트리거 패턴**\n"
    if same:
        md += "- ⚠️ 저장된 정화모델 = 공격모델 (BackdoorBench NC는 정화 가중치를 저장하지 않음 → 복구 미재현)\n"
    if rec is not None:
        md += f"- 📄 팀 기록상 NC 적용 후 ASR: **{rec:.1%}** {'(방어 성공)' if rec < 0.2 else '(방어 실패)'}\n"
    return img, md


def gradcam(model, pil_or_tensor):
    """PreActResNet18 마지막 stage 기준 Grad-CAM heatmap(PIL) 생성."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    x = _to_input(pil_or_tensor)
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale = cam(input_tensor=x)[0]  # (32,32)
    # 입력 역정규화해서 배경 이미지로
    denorm = T.Normalize(
        mean=[-m / s for m, s in zip(NORM_MEAN, NORM_STD)],
        std=[1 / s for s in NORM_STD],
    )
    rgb = denorm(x[0].cpu()).clamp(0, 1).numpy().transpose(1, 2, 0)
    vis = show_cam_on_image(rgb, grayscale, use_rgb=True)
    return Image.fromarray(vis)
