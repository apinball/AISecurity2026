# Input-aware 백도어 공격 / 방어 실험 보고서

**대상 공격**: Input-aware Dynamic Backdoor Attack (Nguyen & Tran, NeurIPS 2020)
**데이터셋 / 모델**: CIFAR-10 / PreAct-ResNet18
**평가 방어**: FT, NAD, ANP, ABL, NC (5종)
**비교 대상**: WaNet (별도 보고서) — 동일 저자의 후속 공격, 동일 실험 셋팅

---

## 0. 한눈에 보기

| 단계 | clean ACC | ASR | RA | WaNet과 비교 |
|---|---|---|---|---|
| 공격 (Input-aware) | 0.9106 | **0.9449** | 0.0524 | 거의 동일 (WaNet 0.914/0.951) |
| + FT 방어 | 0.9265 | 0.2127 | 0.744 | 약간 더 견딤 (WaNet 0.17) |
| + NAD 방어 | 0.9260 | **0.5473** | 0.432 | **훨씬 강하게 견딤** (WaNet 0.087) |
| + ANP 방어 (기본) | 0.8634 | **0.0050** | 0.844 | 거의 동일 (WaNet 0.010) |
| + ANP (Pareto knee t=0.45) | 0.9102 | 0.0109 | 0.885 | knee도 거의 동일 |
| + ABL 방어 (소규모) | **0.5964** | 0.9899 | 0.007 | **완전 실패** (WaNet 0.84/0.59) |
| + NC 방어 | 0.9120 | 0.9844 | 0.015 | 둘 다 실패 |

핵심:
- Input-aware는 WaNet과 헤드라인 ACC/ASR이 거의 동일하나 **방어 견딤 정도가 다름**
- NAD에 훨씬 강하게 견딤 (ASR 6배), ABL에 더 치명적으로 깨짐 (model collapse)
- ANP / NC에 대한 반응은 WaNet과 거의 동일 — 두 attack의 공통 특성

---

## 1. 환경

WaNet 보고서와 동일 환경 (별도 보고서 §1 참조). Windows + RTX 5070 Ti + Python 3.10 + PyTorch 2.11+cu128.

---

## 2. Input-aware 공격 원리

Input-aware는 **입력별로 다른 트리거**를 생성기 네트워크가 생성한다. WaNet의 고정 워핑과 정반대 설계.

| | Input-aware | WaNet |
|---|---|---|
| 트리거 | **per-sample, generator가 합성** | **per-network, 고정 워핑** |
| 트리거 결정 | 입력 이미지에 의존 | 학습 시작 시 고정 |
| Cross sample | 있음 (디코이) | 있음 |
| 저자 | Nguyen & Tran | Nguyen & Tran |
| 발표 | NeurIPS 2020 | ICLR 2021 |

### 학습 구조
- **stage1 (clean training)**: 25 epochs 동안 generator/mask 없이 backbone classifier C만 학습 → 좋은 초기 표현 확보
- **stage2 (backdoor training)**: 100 epochs 동안 세 네트워크를 *동시* 학습
  - Generator G: trigger 이미지 합성 (입력 의존)
  - Mask M: trigger의 spatial location 결정
  - Classifier C: clean + bd + cross 샘플 학습

### 핵심 손실
- `loss_div`: G가 입력에 따라 *다양한* 트리거를 생성하도록 강제 (per-image diversity)
- `loss_norm`: G가 만든 trigger의 spatial norm 제한 (imperceptibility)
- bd_loss: classifier가 trigger를 target으로 분류
- cross_loss: classifier가 cross sample (다른 입력의 trigger를 갖다 붙임)을 *원래 라벨*로 분류 → "특정 입력-트리거 쌍"만 백도어로 작동하게 만드는 lock-in

### 기본 하이퍼파라미터

| 파라미터 | 값 | 의미 |
|---|---|---|
| `pratio` | 0.1 | poison ratio |
| `attack_target` | 0 (airplane) | target class |
| `mask_density` | 0.032 | mask 영역 비율 (전체 픽셀의 3.2%) |
| `lambda_div` | 1 | diversity loss 가중치 |
| `lambda_norm` | 100 | trigger norm penalty 가중치 |
| `clean_train_epochs` | 25 | stage1 길이 |

---

## 3. 실험 1: Input-aware 공격 학습

### 3.1 학습 셋팅
- 100 epochs (WaNet 메인 run과 동일), batch_size 256, AMP, SGD
- stage1 25 epoch (clean) + stage2 100 epoch (bd 학습)
- 학습 시간: 약 100분 (RTX 5070 Ti)

### 3.2 결과 (epoch 99 / 마지막)

| 지표 | 값 |
|---|---|
| **test_acc** (clean) | 0.9106 |
| **test_asr** | 0.9449 |
| **test_ra** | 0.0524 |
| **test_cross_acc** | 0.8440 |
| max test_asr (학습 도중 피크) | 0.9953 |

### 3.3 비교 시각화 — clean vs Input-aware 백도어

3행 × 8열: (1) clean / (2) Input-aware 백도어(bd, label은 모두 target=0으로 변경됨) / (3) cross-noise. 저장된 변형 이미지를 인덱스로 원본 CIFAR-10 테스트셋과 매칭해 나란히 비교한다.

![clean vs bd vs cross](record/inputaware_cifar10_fast/clean_vs_bd_compare.png)

**관찰 포인트** (WaNet 보고서 §3.5의 동일 시각화와 결정적 차이):
- 2행(bd)에 **눈에 보이는 작은 colored line/dot 패턴** 존재 — 빨강/민트(cyan)/검정/노랑 색이 좁은 영역(가장자리 또는 객체 위)에 집중. WaNet의 bd는 clean과 시각적 구별 불가였던 것과 대조
- 이 colored 패턴이 **Input-aware의 실제 트리거**다. Generator G가 입력마다 다른 패턴을 합성하고 Mask M이 spatial 위치를 결정한 결과
- 채도가 극단적인 이유: `mask_density=0.032` (전체 픽셀의 3.2%만 사용) + `lambda_norm=100` (트리거 norm 강한 페널티) → 작은 영역에 강한 신호를 담아야 하므로 고채도 RGB 극단값이 최적
- 3행(cross)은 *다른 입력*의 trigger를 갖다 붙인 샘플 — 트리거 위치/패턴이 bd와 같은 입력에 어울리지 않게 misaligned되어 있음. 이 cross 샘플은 학습 시 *원래 라벨로 분류되도록* 강제됨 → 모델이 "특정 입력-트리거 쌍"만 백도어로 작동하게 lock-in

**Stealth 개념의 차이**:
- WaNet stealth: *픽셀 차이 거의 0* (육안 무관)
- Input-aware stealth: *트리거가 작고 클래스/위치가 가변* (눈에 보이지만 무시할 만한 크기)

같은 0.94+ ASR을 달성하는 두 attack이지만, **imperceptibility 기준에서는 WaNet이 명확히 우위**. Input-aware는 imperceptibility 일부를 동적 트리거 메커니즘에 양보한 셈.

→ WaNet (0.914 / 0.951 / 0.048)과 헤드라인 수치 거의 동일. 두 attack 모두 같은 stealth 목표 도달.

---

## 4. 실험 2: 방어 평가 (5종)

각 방어를 동일 셋팅(num_workers 0, pin_memory False)으로 실행.

| 방어 | 카테고리 | 핵심 가정 / 동작 |
|---|---|---|
| **FT** | Fine-tuning | 깨끗한 데이터로 재학습 |
| **NAD** | Distillation | FT teacher → attention map 증류 |
| **ANP** | Pruning | adversarial perturbation으로 백도어 뉴런 찾아 제거 |
| **ABL** | Isolation + unlearning | low-loss outlier 격리 → finetuning → gradient ascent unlearning |
| **NC** | Trigger reverse-eng. | minimum-L1 reverse trigger → MAD anomaly → 미세조정 |

### 4.1 방어별 결과

#### FT
| | 공격 후 | **FT 후** |
|---|---|---|
| ACC | 0.9106 | 0.9265 |
| ASR | 0.9449 | **0.2127** |
| RA | 0.0524 | 0.7437 |

→ asr 95% → 21%로 감소. WaNet (FT 후 asr 0.17)보다 4%p 더 견딤. per-sample trigger가 단순 재학습에 약간 더 강함.

#### NAD
| | 공격 후 | **NAD 후** |
|---|---|---|
| ACC | 0.9106 | 0.9260 |
| ASR | 0.9449 | **0.5473** |
| RA | 0.0524 | 0.4321 |

→ asr 95% → 55% (절반밖에 감소 안 됨). **WaNet (NAD 후 asr 0.087) 대비 6배 강하게 견딤**. NAD에 대한 Input-aware의 robustness는 이 보고서의 가장 큰 발견.

#### ANP (기본 threshold)
| | 공격 후 | **ANP 후** |
|---|---|---|
| ACC | 0.9106 | 0.8634 |
| ASR | 0.9449 | **0.0050** |
| RA | 0.0524 | 0.8436 |

→ ANP는 매우 효과적. 단 ACC 5%p 손해. (Pareto knee 결과는 §5.1 참조)

#### ABL (소규모: tuning 10 / finetuning 20 / unlearning 10)
| | 공격 후 | **ABL 후** |
|---|---|---|
| ACC | 0.9106 | **0.5964** |
| ASR | 0.9449 | 0.9899 |
| RA | 0.0524 | 0.0066 |

→ **완전 실패**. ABL이 unlearning 단계에서 모델을 collapse 시키면서 ASR을 떨어뜨리지도 못함. WaNet ABL 소규모 결과(ACC 0.84 / ASR 0.59)와 비교해도 훨씬 나쁨. Input-aware의 per-sample 트리거가 ABL의 "low-loss outlier = 백도어 샘플" heuristic을 깨뜨려 잘못된 샘플을 격리한 것으로 추정.

#### NC — **실패**
| | 공격 후 | **NC 후** |
|---|---|---|
| ACC | 0.9106 | 0.9120 |
| ASR | 0.9449 | **0.9844** |
| RA | 0.0524 | 0.0147 |

NC가 reverse-engineering한 클래스별 트리거 L1 norm:
```
117.72, 111.47, 102.16, 103.10, 108.00, 114.39, 137.00, 128.66, 101.49, 121.63
```
모든 클래스가 101~137 범위로 평탄 → target=0 (airplane, L1=117.72) anomaly 검출 불가 → unlearning 미수행.
WaNet의 동일 결과(91~139)와 같은 패턴으로 **NC가 가정하는 "작은 고정 패턴" 가설이 두 stealthy attack 모두에 부적합**함을 재확인.

### 4.2 종합 비교 (6 그룹 막대)

![defense compare](record/inputaware_cifar10_fast/defense_compare.png)

---

## 5. 정량 분석

### 5.1 ANP threshold sweep — Pareto frontier

ANP의 mask threshold `t`를 [0.0, 0.9] 구간에서 sweep.

![anp threshold sweep](record/inputaware_cifar10_fast/anp_threshold_sweep.png)

| t | ACC | ASR | RA |
|---|---|---|---|
| 0.00 | 0.910 | 0.858 | (init) |
| 0.05 | 0.912 | 0.044 | — |
| 0.20 | 0.914 | 0.039 | — |
| 0.35 | 0.916 | **0.016** | — |
| **0.45 (knee)** | **0.9102** | **0.0109** | **0.8848** |
| 0.55 | 0.895 | 0.015 | — |
| 0.65 | 0.863 | 0.005 | — |
| 0.70 | 0.811 | **0.0003** | — |
| 0.90 | 0.540 | 0.000 | — |

**관찰**:
- **t ∈ [0.05, 0.65]** 구간이 ACC 거의 무손실(≥0.86)이면서 ASR < 0.05인 wide flat region. WaNet과 거의 동일한 형태의 ⌐ 모양 Pareto frontier
- Knee t=0.45: **ACC 0.9102 / ASR 0.0109 / RA 0.8848** — 본 보고서 §4.1의 ANP 단일 실행값(ACC 0.8634)보다 더 좋은 trade-off
- **WaNet의 ANP knee와 거의 동일** (WaNet: ACC 0.9064 / ASR 0.0114). 두 attack 모두 ANP에 비슷한 수준으로 깨진다는 정량 증거

### 5.2 Layer-wise weight delta — attack vs NAD

NAD 정화가 네트워크의 어느 부분을 수정했는지 stage 단위로 측정.

![layer weight delta](record/inputaware_cifar10_fast/layer_weight_delta_nad.png)

| stage | rel ΔW (Input-aware) | rel ΔW (WaNet, 참고) |
|---|---|---|
| conv1  | 0.038 | 0.044 |
| layer1 | 0.031 | 0.039 |
| layer2 | 0.035 | 0.047 |
| layer3 | **0.038** | **0.049** |
| layer4 | 0.024 | 0.036 |
| **linear** | **0.020** | **0.029** |

**Top-20 most-changed parameters** (Input-aware):
1. `layer4.1.bn2.bias` — 0.144 (1위, WaNet에서도 1위였음. 두 attack 공통)
2. `layer3.0.conv2.weight` — 0.058
3. `layer3.1.conv1.weight` — 0.053
4. ... 이어서 conv weight, BN bias 다수

**관찰**:
- **모든 stage가 WaNet 대비 변화량이 작다** (예: layer4 0.024 vs 0.036). NAD가 Input-aware에서 가중치를 덜 수정했음 → §4.1 NAD 결과가 ASR 0.55로 부분 정화에 그친 것과 일관
- **Top-1 파라미터가 두 attack 모두 `layer4.1.bn2.bias`로 동일**. 분류 헤드 직전의 BN bias가 백도어 결정 경로의 공통 핵심 노드
- `linear`(분류 헤드) 변화가 가장 작다는 패턴도 두 attack 공통 (0.020 vs 0.029). 백도어는 분류 헤드가 아닌 중-후반 conv stage + BN bias에 분산 형성

### 5.3 ABL 실패 메커니즘 분석

ABL은 가정: *백도어 샘플은 학습 도중 비정상적으로 낮은 loss를 보임* → 그것만 격리해서 unlearn.

WaNet에서는 고정 워핑이라 backbone이 "워핑된 = 백도어" 매핑을 빠르게 학습 → 백도어 샘플 loss가 정말로 낮음 → ABL의 low-loss 격리가 어느 정도 맞음.

Input-aware에서는 generator가 매번 다른 trigger를 만들기 때문에 backbone이 "이 특정 패턴 = 백도어"라고 외울 수 없음 → 백도어 샘플 loss가 clean과 별로 다르지 않음 → ABL이 *잘못된* 샘플 500개를 격리 → unlearning 단계에서 정상 representation을 unlearn 시키며 model collapse.

이는 본 실험에서 ABL이 unlearning 직후 train_acc 0.10 (10클래스 random), test_acc 0.60, test_asr 0.99를 보인 패턴과 일관.

---

## 6. 시각화 분석

PreAct-ResNet18의 동일 layer (`layer4.1.conv2`)에서 공격 모델과 NAD 정화 모델을 비교.

### 6.1 t-SNE — feature 공간 분포

각 색은 클래스(0~9), **검정색 ★ = Poisoned 샘플** (bd_test).

**공격 모델**
![tsne attack](record/inputaware_cifar10_fast/visual/tsne_mixed.png)

**NAD 방어 후**
![tsne nad](record/inputaware_cifar10_fast/defense/nad/visual/tsne_mixed.png)

**관찰** (WaNet과 명확한 차이):
- **공격 모델에서 poisoned 점들이 중앙에 매우 cohesive한 cluster를 형성** — 모든 입력에 대해 generator가 만든 트리거가 같은 feature-space 위치로 수렴. WaNet 공격 모델에서는 poisoned 점이 클러스터 사이에 흩어져 있었음 (§WaNet 보고서 §5.1)
- **NAD 후엔 poisoned 점이 여러 클래스 cluster로 분산** — NAD가 cohesive backdoor representation을 일부 깨뜨렸음. 단 완전 정화는 아님 (NAD ASR 0.547과 일관)
- Input-aware의 *동적 per-sample* 트리거가 feature-space에서는 *정적 cluster*를 만든다는 발견 — trigger 다양성과 feature 다양성이 분리되어 있음

### 6.2 Grad-CAM — 모델이 보는 영역

위쪽 행 2개: clean 샘플 (Dog, Horse). 아래쪽 행 2개: Input-aware 백도어 샘플 (Cat → 트리거 적용).

**공격 모델 Grad-CAM**
![gradcam attack](record/inputaware_cifar10_fast/visual/gradcam_bd_test.png)

**NAD 방어 후 Grad-CAM**
![gradcam nad](record/inputaware_cifar10_fast/defense/nad/visual/gradcam_bd_test.png)

| | clean Dog | clean Horse | bd Cat #1 | bd Cat #2 |
|---|---|---|---|---|
| 공격 모델 | Cat 88% (오) | Horse 99% ✓ | **Airplane** 98% ✗ | **Airplane** 99% ✗ |
| NAD 후 | Cat 89% (오) | Horse 99% ✓ | **Airplane** 83% ✗ | **Airplane** 96% ✗ |

**관찰**:
- bd 이미지에 **눈에 보이는 작은 colored line/dot 패턴**(cyan/magenta/yellow 줄무늬)이 가장자리나 객체 위에 존재. **WaNet의 invisible 워핑과 가장 큰 시각적 차이** — Input-aware는 imperceptibility 일부를 동적 트리거에 양보
- Grad-CAM heatmap은 두 모델 모두 객체 중심 + 패턴이 있는 영역에 분산. 공격 모델은 같은 영역을 보고도 Airplane으로 분류
- **NAD 후에도 이 특정 샘플들은 여전히 Airplane으로 오분류** (이 sample들은 NAD가 정화하지 못한 ASR 0.55의 영역에 속함) — WaNet NAD에서 동일 샘플들이 Cat으로 회복된 것과 대조

### 6.3 Frequency saliency — 주파수 영역

위쪽: clean (Deer, Truck). 아래쪽: 백도어 (Bird → Airplane 99.56%, Cat → Airplane 99.77%).

![frequency saliency](record/inputaware_cifar10_fast/visual/frequency_mixed.png)

**관찰**:
- 백도어 입력의 colored line 트리거가 이미지에서 명확히 보이지만, 주파수 saliency map은 clean과 비슷한 분포 (red dominant + 산발적 blue)
- WaNet의 frequency 분석에서는 가장자리/모서리에 saliency가 약하게 응집했으나, Input-aware는 그런 패턴 없음 → 동적 per-sample 트리거가 spatial 위치를 자유롭게 옮기기 때문에 *일관된 frequency 시그니처*가 존재하지 않음
- 결론: 주파수 영역만으로 Input-aware 백도어를 식별하기는 어려움. WaNet보다 더 frequency-agnostic

### 6.4 Confusion Matrix — bd_test에서 분류 결과

bd_test의 True label은 모두 target=0(airplane)으로 변경되어 있어, airplane 행만 의미 있는 데이터 보유.

**공격 모델 CM**
![cm attack](record/inputaware_cifar10_fast/visual/cm_bd_test.png)

→ airplane 행: **0.98이 airplane으로 예측** = ASR 98%. 백도어가 매우 강하고 일관됨.

**NAD 방어 후 CM**
![cm nad](record/inputaware_cifar10_fast/defense/nad/visual/cm_bd_test.png)

→ airplane 행: **0.55 airplane** (= NAD 후 ASR 0.547), 나머지 45%는 9개 클래스로 흩어짐 (horse 0.09, truck 0.08, automobile 0.06 등).

**WaNet과의 차이**:
- WaNet NAD CM은 airplane 행이 9~14% **거의 균등 분포** — 백도어 거의 완전 무력화
- Input-aware NAD CM은 airplane이 여전히 0.55로 **최대값** — 백도어 정렬이 절반은 남아있음
- 시각적으로 두 stealthy attack의 *NAD 견딤 차이* 가 가장 명확하게 드러나는 그림

---

## 7. WaNet vs Input-aware 종합 비교

같은 저자의 두 attack을 동일 셋팅(100 epoch, CIFAR-10, PreActResNet18, 동일 hparam 골격)에서 평가한 결과.

### 6.1 공격 효과 (방어 없이)

| 지표 | WaNet | Input-aware |
|---|---|---|
| clean ACC | 0.914 | 0.911 |
| ASR | 0.951 | 0.945 |
| RA | 0.048 | 0.052 |

→ **거의 동일**. 두 attack 모두 stealth 목표 도달.

### 6.2 방어 견딤 (ASR after defense, 낮을수록 방어 효과 큼)

| 방어 | WaNet ASR | Input-aware ASR | 차이 |
|---|---|---|---|
| FT | 0.168 | 0.213 | +4.5%p |
| **NAD** | **0.087** | **0.547** | **+46.0%p** ← 가장 큰 차이 |
| ANP (default) | 0.010 | 0.005 | −0.5%p |
| ANP knee (t=0.45) | 0.011 | 0.011 | 동일 |
| **ABL (소규모)** | 0.586 | 0.990 | +40.4%p (model collapse) |
| NC | 0.951 (실패) | 0.984 (실패) | 둘 다 실패 |

### 6.3 핵심 발견 4가지

1. **NC는 두 attack 모두 무력화** — reverse trigger L1 분포가 모든 클래스에서 평탄(WaNet 91~139, Input-aware 101~137) → MAD anomaly 미검출. NC가 가정하는 "작은 고정 패턴" 가설은 **stealthy attack 일반에 부적합**함을 두 사례로 확인

2. **NAD는 Input-aware에 6배 더 약함** (ASR 0.087 → 0.547). 가설: WaNet의 고정 워핑은 teacher의 attention map에서 단일 패턴으로 학습 가능하나, Input-aware의 per-sample 동적 트리거는 teacher가 백도어 attention을 *하나의 일관된 패턴*으로 포착하지 못해 student 증류 효과 약화. §5.2의 layer-wise weight delta가 이를 지지 — NAD가 Input-aware에서 가중치를 더 적게 수정했음

3. **ANP는 두 attack 모두 동등하게 깸** (knee 모두 ACC 0.91 / ASR 0.011). 백도어 형성 위치(중-후반 conv stage + BN bias)와 마스크 가능한 뉴런 수가 두 attack에서 비슷하다는 정량 증거 (Top-1 파라미터도 둘 다 `layer4.1.bn2.bias`)

4. **ABL은 Input-aware에 완전 실패** (model collapse). 가정(*"백도어 샘플 = low-loss outlier"*)이 per-sample dynamic trigger에서 무너짐 → 잘못된 샘플을 격리해 정상 표현까지 unlearn. WaNet에서는 적어도 부분 효과(asr 0.95 → 0.59)가 있었던 반면 Input-aware에선 model이 무너지며 asr 변화 없음

---

## 8. 결론

1. **두 attack은 헤드라인이 거의 동일하나 방어 견딤은 완전히 다름** — 같은 ACC/ASR을 보이는 백도어가 본질적으로 다른 강건성을 가질 수 있음
2. **트리거의 동적 여부가 NAD / ABL의 효과를 가른다** — 고정 트리거(WaNet)는 attention 패턴을 한 가지로 학습시키고 백도어 샘플의 loss를 일관되게 낮춤 → NAD가 attention에서 식별 가능, ABL이 격리 가능. 동적 트리거(Input-aware)는 둘 다 깨뜨림
3. **ANP는 트리거 형태와 무관하게 효과적** — 두 attack 모두 mask threshold sweep의 ⌐ 모양 frontier가 거의 동일. 백도어 학습된 뉴런의 *수와 위치*는 트리거 동적 여부에 의존하지 않음
4. **NC의 한계는 stealthy attack 일반에 적용** — 워핑이든 generator든 "작은 고정 패턴"이 아니면 reverse-engineering이 정상 클래스 트리거와 구별 안 됨
5. **공통 백도어 노드 `layer4.1.bn2.bias`** — 두 stealthy attack이 동일한 단일 파라미터를 백도어 결정의 핵심으로 사용. BN affine 파라미터의 백도어 거점 역할은 attack-agnostic 현상으로 보임

---

## 부록: 본 실험 산출물

- 학습된 모델 ckpt + 메타 (5종 방어 결과 포함)
- 정량 분석 PNG 3종: ANP threshold sweep, layer-wise weight delta, defense_compare bar chart
- epoch별 메트릭 csv 일체
