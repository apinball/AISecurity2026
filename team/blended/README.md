# Blended (팀원 작업물 통합)

Blended 백도어 공격 1건 + 방어 5종(ABL / ANP / FT / NAD / NC) 실험 결과를 통합한 것이다.
원본 전달 자료는 `code/blended_s0/`(zip 포함, git 미포함)이며, 경량 자료는 이 폴더에,
무거운 체크포인트(.pt)는 `record/blended_team/checkpoints/`(`.gitignore` 처리됨)에 배치했다.

- 모델: `preactresnet18` / 데이터셋: CIFAR-10 (`num_classes=10`, `img_size=(32,32,3)`)
- 공격: Blended (전역 이미지 블렌딩 트리거), 결과 디렉터리명 `blended_s0`
- 방어: ABL, ANP, FT, NAD, NC (BadNets와 달리 **FP는 없음**)

## 이 폴더 내용 (git 포함)

| 파일 | 설명 |
|---|---|
| `blended_정리.docx` | 작업자가 전달한 정리 문서 (파일 목록 + 코드 변경사항) |
| `attack_df.csv`, `attack_df_summary.csv` | 공격 학습 epoch별 지표 / 요약 |
| `metrics/` | 방어별 요약 지표 CSV (ABL/ANP/FT/NAD/NC) |
| `report_figures/` | 공격 곡선·방어 비교·ABL 단계·trade-off 그래프 4종 + NAD 시각화 zip |

## 체크포인트 (git 미포함, `record/blended_team/checkpoints/`)

| 파일 | 내용 | 크기 |
|---|---|---|
| `BLENDED_attack_result.pt` | 백도어 심긴 모델 + bd_train/bd_test | 45.6 MB |
| `BLENDED_ABL_defense_result.pt` | ABL 방어 후 모델 | 42.7 MB |
| `BLENDED_ANP_defense_result.pt` | ANP 방어 후 모델 | 42.7 MB |
| `BLENDED_FT_defense_result.pt` | Fine-Tuning 방어 후 모델 | 42.7 MB |
| `BLENDED_NAD_defense_result.pt` | NAD 방어 후 모델 | 42.7 MB |
| `BLENDED_NC_defense_result.pt` | Neural Cleanse 방어 후 모델 | 42.7 MB |

(원본 6개 모두 MD5 일치 확인. lf_team / sig_team / badnet_team 과 동일한 네이밍 규칙.)

## 결과 요약 (test_acc / test_asr / test_ra)

| 단계 | ACC | ASR | RA |
|---|---|---|---|
| 공격(Blended) | 0.9340 | **0.9988** | 0.0011 |
| FT  | 0.9254 | **0.8341** | 0.1492 |
| NAD | 0.9225 | **0.7117** | 0.2504 |
| ANP | 0.8835 | 0.0198 | 0.6059 |
| NC  | 0.9340 | **0.9988** | 0.0011 |
| ABL | 0.7285 | 0.0004 | 0.5677 |

(값 출처: 각 `*_df_summary.csv`의 `last`/`0` 행.)

### ⚠️ BadNets와의 핵심 차이 — Blended는 방어가 더 어렵다

- **BadNets**: 6방어 모두 ASR을 ~0%로 무력화.
- **Blended**: **FT(83%)·NAD(71%)는 방어 거의 실패**, **NC는 ASR 99.9%로 완전 실패**(공격값과 동일).
  유효한 방어는 **ANP(2%)·ABL(0%)** 뿐이며, 이마저 clean ACC 손실(ANP 88%, ABL 73%)이 큼.
- 전역 블렌딩 트리거가 patch형보다 뉴런에 분산되어 박혀, fine-tuning·neuron-attribution 계열
  방어로 지우기 어렵다는 점을 보여주는 결과.

## 코드 변경 / 로딩 주의

작업자가 적용한 BackdoorBench 코드 변경은 2건뿐이며, 본 저장소 통합 패치 #1·#3과 **동일**하다
(`team/README.md` 호환성 패치 표 참조). **새로 적용할 패치 없음.**

| 파일 | 변경 | 이유 |
|---|---|---|
| `utils/save_load_attack.py` | `torch.load(...)` → `torch.load(..., weights_only=False)` | PyTorch 2.6+ 기본값 변경 |
| `utils/trainer_cls.py` 등 | `np.infty` → `np.inf` | NumPy 2.0+ alias 삭제 |

`.pt`는 반드시 `weights_only=False`로 로딩:

```python
result = torch.load("record/blended_team/checkpoints/BLENDED_attack_result.pt",
                    map_location="cpu", weights_only=False)
```
