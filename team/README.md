# 팀 작업 분담 및 통합 정리

본 디렉토리는 AISecurity2026 (BackdoorBench fork) 프로젝트에서 각 팀원이 수행한 작업과 그 결과를 정리한다. 각자 다른 백도어 공격을 담당했고, 방어(FT / NAD / ANP / ABL / NC, BadNets는 FP 추가)는 공통으로 평가했다.

## 담당 공격

| 팀원 | 공격 | 보고서 위치 |
|---|---|---|
| (본인) | **WaNet** (Imperceptible Warping-based) | `/WANET_REPORT.{md,pdf}` |
| (본인) | **Input-aware** (Dynamic per-sample trigger) | `/INPUTAWARE_REPORT.{md,pdf}` |
| 조수빈 | **LF** (Low Frequency, additive frequency-domain trigger) | `team/lf/LF_코드_변경사항_보고서.{docx,pdf}` |
| (SIG 담당자) | **SIG** (Sinusoidal Signal) | `team/sig/SIG 변경코드목록.docx` |
| (BadNets 담당자) | **BadNets** (Patch-based, poison ratio 0.1) | `team/badnet/README.md` (+ `badnets_attack.ipynb`) |
| (Blended 담당자) | **Blended** (전역 이미지 블렌딩 트리거) | `team/blended/README.md` (+ `blended_정리.docx`) |

## 환경 호환성 패치 (통합)

저장소를 clone한 누구든 4종 공격을 모두 돌릴 수 있도록, 팀원 각자가 발견한 호환성 이슈를 모두 반영한 코드로 통합되어 있다.

| # | 파일 | 변경 | 발견자 | 이유 |
|---|---|---|---|---|
| 1 | `utils/save_load_attack.py:198` | `torch.load(..., weights_only=False)` | 전원 공통 | PyTorch 2.6+ default 변경 |
| 2 | `utils/trainer_cls.py:201` | `torch.load(..., weights_only=False)` | WaNet/IA | 동상 |
| 3 | `utils/trainer_cls.py:754-755` | `np.infty → np.inf` | 전원 공통 | NumPy 2.0 alias 삭제 |
| 4 | `utils/bd_img_transform/patch.py:56,62` | `np.float → float` | LF (조수빈) | NumPy 2.0 alias 삭제, additive trigger 적용 시 필요 |
| 5 | `models/preact_resnet.py:38-41, 70` | `out += shortcut → out = out + shortcut` (3곳) | SIG | backward hook + inplace 연산 충돌. Grad-CAM 안정성 |
| 6 | `analysis/visual_gradcam.py:188` | `FullGrad(...)`에서 `use_cuda` 인자 제거 | WaNet/IA | grad-cam 1.5+ 시그니처 변경 |
| 7 | `defense/*.py` (21개 파일) | `--num_workers type=float → type=int` | WaNet/IA | Windows DataLoader 워커 spawn 오류 (Linux/Colab은 안 만남) |

## 신규 분석/시각화 스크립트

| 파일 | 목적 | 작성자 |
|---|---|---|
| `analysis/compare_wanet_clean_vs_bd.py` | clean / bd / cross 3행 비교 PNG (WaNet, Input-aware에 적용) | WaNet/IA |
| `analysis/plot_wanet_defense_compare.py` | ACC/ASR/RA 6 그룹 막대 그래프 | WaNet/IA |
| `analysis/plot_anp_threshold_sweep.py` | ANP mask threshold Pareto frontier | WaNet/IA |
| `analysis/plot_layer_weight_delta.py` | per-stage ‖W_attack − W_defense‖ + top-20 파라미터 | WaNet/IA |
| `analysis/plot_s_ablation.py` | WaNet `s` ablation (3-panel) | WaNet/IA |
| `analysis/md_to_html.py` | Markdown → HTML (한글 폰트 CSS) | WaNet/IA |
| `sh_run_s_ablation.sh` | s ∈ {0.1, 0.3, 0.5, 0.7, 1.0} 자동 실행 | WaNet/IA |

LF 조수빈은 Colab 노트북 형식 (`team/lf/Low Frequency_test.ipynb`)으로 공격 학습 + 5방어 + 시각화 일괄 자동화 파이프라인을 구현했다. SIG는 별도 파이프라인을 사용했다 (자세한 코드 변경은 `team/sig/SIG 변경코드목록.docx` 참조). BadNets 담당자도 Colab 노트북 (`team/badnet/badnets_attack.ipynb`)으로 공격 + 6방어(FP 포함) + 시각화를 수행했고, 적용한 코드 변경은 위 호환성 패치 #1·#3과 동일하다 (추가 패치 없음). 결과 체크포인트는 `record/badnet_team/`에 통합되어 있다 (`team/badnet/README.md` 참조). Blended 담당자도 동일한 코드 변경(#1·#3, 추가 패치 없음)으로 공격 + 5방어(ABL/ANP/FT/NAD/NC) + 시각화를 수행했고, 결과 체크포인트는 `record/blended_team/`에 있다 (`team/blended/README.md` 참조). 단, Blended는 FT·NAD·NC 방어가 거의 실패해 ASR이 높게 남는 점이 BadNets와 다르다.

## 재현 가이드

1. conda env 생성 (Python 3.10+):
   ```
   conda create -n backdoor310 python=3.10 -y
   conda activate backdoor310
   pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
   pip install opencv-python pandas Pillow scikit-learn scikit-image tqdm pyyaml \
               tensorboard kornia imageio matplotlib scipy pytorch-wavelets seaborn \
               PyWavelets umap-learn grad-cam markdown
   ```
2. 데이터셋 폴더 생성: `sh ./sh/init_folders.sh`
3. 공격 실행: `python attack/<name>.py --yaml_path ./config/attack/prototype/cifar10.yaml --bd_yaml_path ./config/attack/<name>/default.yaml --dataset cifar10 --num_workers 0 --pin_memory False --save_folder_name <result_dir>`
4. 방어 실행: `python defense/<name>.py --result_file <attack_result_dir> --yaml_path ./config/defense/<name>/cifar10.yaml --dataset cifar10 --num_workers 0 --pin_memory False`
5. 시각화: `python analysis/plot_wanet_defense_compare.py --result_file <attack_result_dir>` 등

Windows 환경에서는 위 호환 패치 7건이 모두 필요하다. Colab/Linux는 #5-7이 환경 조합에 따라 불필요할 수 있으나, 미리 적용해 두면 어디서 돌려도 안전하다.
