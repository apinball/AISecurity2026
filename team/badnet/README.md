# BadNets (팀원 작업물 통합)

BadNets 백도어 공격 1건 + 방어 6종(ABL / FP / FT / NAD / ANP / NC) 실험 결과를 통합한 것이다.
원본 전달 자료는 `code/BackdoorBench_results 2/` (zip 포함, git 미포함)이며, 그중 git에 올릴 만한
경량 자료(노트북·요약 CSV·README)는 이 폴더에, 무거운 체크포인트(.pt)는
`record/badnet_team/checkpoints/`(`.gitignore` 처리됨)에 배치했다.

- 모델: `preactresnet18` / 데이터셋: CIFAR-10 (`num_classes=10`, `img_size=(32,32,3)`)
- 공격: BadNets, poison ratio 0.1 (`badnet_0_1`)
- 실행 환경: Google Colab (GPU), 최신 PyTorch 2.6+ / NumPy 2.0+

## 이 폴더 내용 (git 포함)

| 파일 | 설명 |
|---|---|
| `badnets_attack.ipynb` | 공격 학습 + 6방어 + 시각화 Colab 노트북 (원본명 `badnets_attack_ (1).ipynb`) |
| `README_received.md` | 작업자가 함께 전달한 원본 README (로딩 주의사항·패치 설명 포함) |
| `attack_df.csv`, `attack_df_summary.csv` | 공격 학습 epoch별 지표 / 요약 |
| `metrics/` | 방어별 요약 지표 CSV (FT/NAD/ANP/NC/ABL/FP) |

## 체크포인트 (git 미포함, `record/badnet_team/checkpoints/`)

| 파일 | 내용 | 크기 |
|---|---|---|
| `BADNET_attack_result.pt` | 백도어 심긴 모델 + bd_train/bd_test | 45.6 MB |
| `BADNET_ABL_defense_result.pt` | ABL 방어 후 모델 | 42.7 MB |
| `BADNET_FP_defense_result.pt` | Fine-Pruning 방어 후 모델 | 42.7 MB |
| `BADNET_FT_defense_result.pt` | Fine-Tuning 방어 후 모델 | 42.7 MB |
| `BADNET_NAD_defense_result.pt` | NAD 방어 후 모델 | 42.7 MB |
| `BADNET_ANP_defense_result.pt` | ANP 방어 후 모델 | 42.7 MB |
| `BADNET_NC_defense_result.pt` | Neural Cleanse 방어 후 모델 | 42.7 MB |

(원본 6개 모두 MD5 일치 확인. lf_team / sig_team 과 동일한 `<공격>_<방어>_defense_result.pt` 네이밍 규칙.)

## 결과 요약 (test_acc / test_asr / test_ra)

| 단계 | ACC | ASR | RA |
|---|---|---|---|
| 공격(BadNets) | 0.9173 | **0.9440** | 0.0537 |
| FT  | 0.9034 | 0.0131 | 0.9002 |
| NAD | 0.8911 | 0.0096 | 0.8940 |
| ANP | 0.8690 | 0.0001 | 0.8987 |
| NC  | 0.9015 | 0.0073 | 0.9042 |
| FP  | 0.9183 | 0.0071 | 0.9154 |
| ABL | 0.8293 | 0.0000 | 0.9010 |

(공격 ASR 94.4% → 6방어 모두 ASR ~0%로 거의 무력화. 값 출처: 각 `*_df_summary.csv`의 `last` 행.)

## `.pt` 로딩 주의 (중요)

전달 자료의 `.pt`는 구버전 PyTorch로 저장돼, **PyTorch 2.6+에서는 반드시 `weights_only=False`** 로 열어야 한다.

```python
result = torch.load("record/badnet_team/checkpoints/BADNET_attack_result.pt",
                    map_location="cpu", weights_only=False)
```

이는 본 저장소 통합 패치 #1(`utils/save_load_attack.py`)과 동일한 이유다. 작업자가 적용한 원본
코드 변경 2건(`torch.load weights_only=False`, `np.infty → np.inf`)은 이미 본 저장소에 반영되어 있다
(`team/README.md` 호환성 패치 표 #1, #3 참조). 즉 **새로 적용할 코드 패치는 없다.**
