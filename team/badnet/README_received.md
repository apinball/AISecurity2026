# BackdoorBench 결과 (BadNets) — 시각화용 데이터셋

[BackdoorBench](https://github.com/SCLBD/BackdoorBench) 로 **BadNets** 백도어 공격 1건과
방어 기법 6종을 실행한 결과물입니다. 이 리포는 아래 `.pt` 파일들을 직접 로딩해서 시각화하는 용도입니다.

- 모델: `preactresnet18` / 데이터셋: CIFAR-10 (`num_classes=10`, `img_size=(32, 32, 3)`)
- 공격: BadNets, poison ratio 0.1 (`badnet_0_1`)
- 방어: ABL, FP(Fine-Pruning), FT(Fine-Tuning), NAD, ANP, NC(Neural Cleanse)

---

## ⚠️ 먼저 읽어주세요 — `.pt` 로딩 시 필수 옵션

이 `.pt` 파일들은 **구버전 PyTorch로 저장**되었습니다.
**PyTorch 2.6+ 에서는 `torch.load` 의 기본값이 `weights_only=True` 로 바뀌어서, 그냥 열면 에러가 납니다.**
시각화 코드에서 `.pt` 를 열 때 반드시 `weights_only=False` 를 명시하세요.

```python
import torch

# ✅ 올바른 로딩 (PyTorch 2.6+ 포함 모든 버전에서 동작)
result = torch.load("record/badnet_0_1/attack_result.pt",
                    map_location="cpu", weights_only=False)

# ❌ 이렇게 열면 PyTorch 2.6+ 에서 UnpicklingError / weights_only 에러
# result = torch.load("record/badnet_0_1/attack_result.pt")
```

> 이 한 줄이 아래 "원본 코드 변경 사항 1번"과 정확히 같은 이유입니다.
> 즉, 데이터를 만든 쪽 코드뿐 아니라 **이 시각화 리포의 로딩 코드에도 동일한 패치가 필요**합니다.

---

## 데이터 파일 구조

```
record/badnet_0_1/
├── attack_result.pt              # 공격 결과 (아래 키 구조)
├── attack_df.csv                 # 학습 과정 지표 (epoch별)
├── attack_df_summary.csv         # 공격 결과 요약 지표
├── *.png                         # acc / loss 그래프
├── bd_train_dataset/  bd_test_dataset/   # 백도어 트리거가 적용된 샘플 이미지
└── defense/
    ├── abl/ fp/ ft/ nad/ anp/ nc/
    │   ├── defense_result.pt     # 방어 후 모델 (아래 키 구조)
    │   ├── *_df.csv / *_summary.csv  # 방어별 지표
    │   └── log/*.log             # 실행 로그
```

### `attack_result.pt` 키
| 키 | 타입 | 설명 |
|----|------|------|
| `model_name` | str | `preactresnet18` |
| `num_classes` | int | 10 |
| `img_size` | tuple | `(32, 32, 3)` |
| `model` | OrderedDict | 백도어가 심어진 모델 state_dict |
| `data_path` / `clean_data` | str | 데이터 경로 정보 |
| `bd_train` | dict | 백도어 학습 데이터 |
| `bd_test` | dict | 백도어 테스트 데이터 |

### `defense_result.pt` 키
| 키 | 타입 | 설명 |
|----|------|------|
| `model_name` | str | `preactresnet18` |
| `num_classes` | int | 10 |
| `model` | OrderedDict | 방어가 적용된 모델 state_dict |

---

## 원본 BackdoorBench 코드 변경 사항

원본 깃 코드가 작성된 시점보다 실행 환경(PyTorch / NumPy)이 최신 버전으로 올라가서,
**버전 호환성 패치 2가지**만 적용했습니다. **실험 알고리즘/로직 변경은 없습니다.**
(아래 결과 `.pt` 파일들은 이 패치가 적용된 코드로 생성된 것입니다.)

| # | 파일 | 변경 | 이유 |
|---|------|------|------|
| 1 | `utils/save_load_attack.py` | `torch.load(...)` → `torch.load(..., weights_only=False)` | PyTorch 2.6+ 에서 `torch.load` 기본값이 `weights_only=True` 로 변경됨 |
| 2 | `utils/trainer_cls.py` 및 전체 `*.py` | `np.infty` → `np.inf` | NumPy 2.0+ 에서 `np.infty` 제거됨 (ANP 방어가 죽던 원인) |

### 패치 1 — PyTorch 2.6+ (`utils/save_load_attack.py`)
```diff
- torch.load(save_path)
+ torch.load(save_path, weights_only=False)
- torch.load(load_path)
+ torch.load(load_path, weights_only=False)
```
```bash
sed -i 's/torch.load(save_path)/torch.load(save_path, weights_only=False)/g' ./utils/save_load_attack.py
sed -i 's/torch.load(load_path)/torch.load(load_path, weights_only=False)/g' ./utils/save_load_attack.py
```

### 패치 2 — NumPy 2.0+ (전체 `*.py`)
```diff
- np.infty
+ np.inf
```
```bash
find . -type f -name "*.py" -exec sed -i 's/np.infty/np.inf/g' {} +
```

---

## 재현 환경

- 실행: Google Colab (GPU)
- 원본 코드: `git clone https://github.com/SCLBD/BackdoorBench.git` 후 위 패치 2건 적용
- 주의: 최신 PyTorch 2.6+ / NumPy 2.0+ 환경 기준
