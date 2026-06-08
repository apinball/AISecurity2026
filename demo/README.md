# 백도어 공격/방어 실시간 시연 데모

CIFAR-10 / PreActResNet18 기반. 6개 공격 × 4개 복구형 방어 + NC 탐지 방어를
Gradio 웹 UI로 실시간 시연한다.

- 공격: WaNet · InputAware · BadNets · Blended · LF · SIG (모두 all-to-one, 타깃=airplane)
- 복구형 방어: FT · NAD · ANP · ABL (정화 모델로 ASR 떨어뜨림)
- 탐지형 방어: NC (역공학 트리거로 백도어 탐지 — 정화 모델 미저장)

## 환경

- conda env: **backdoor310** (Python 3.10, PyTorch cu128). 호스트가 아니라 이 env에 격리됨.
- 가중치: `record/` 아래 (`.gitignore` 처리, GitHub 미포함). 로컬에만 존재.
- 실행 PC = 서버. 강의실 PC는 공개 URL로 접속만 함.

## 실행

```powershell
conda activate backdoor310
cd C:\Users\admin\Desktop\document\class\4_1\인공지능보안\code\AISecurity2026

# 로그인 ID/PW 설정 (하드코딩 금지 — 환경변수로). 시연 전 본인 값으로 변경.
$env:DEMO_USER = "team2026"
$env:DEMO_PASS = "원하는비밀번호!"

# 로컬만 (인터넷 노출 0)
python demo/app.py

# 공개 URL (시연용, frpc 터널) — xxxx.gradio.live 생성
python demo/app.py --share
```

- 로컬 주소: http://127.0.0.1:7860  (외부 노출 안 됨 — 127.0.0.1 바인딩)
- `--share`: 72시간 유효한 공개 URL. **강의실 PC 브라우저에서 이 URL 접속.**

## 보안 기능 (구현됨)

1. **로그인 인증** — `demo.launch(auth=(user, pass))`. ID/PW는 환경변수
   `DEMO_USER`/`DEMO_PASS`에서 읽음(GitHub에 비밀번호 노출 방지). 미설정 시 기본값
   사용하며 경고 출력. 로그인 없이 데모 접근 시 **401 차단** (검증 완료).
2. **입력값 화이트리스트 검증** — `_validate_inputs()`. 공격/방어 이름이 허용 목록에
   정확히 일치할 때만 처리, 샘플 인덱스는 정수로 강제, 강도 슬라이더는 범위 clamp.
   `WaNet; rm -rf /`, `NC && ls`, `__import__("os")` 같은 주입 시도는 전부 ⛔ 거부.
   ※ 자유 텍스트 입력창 없음(클릭·라디오·슬라이더·이미지 업로드만). 화이트리스트는
   과제 요건 충족 + 방어적 다중화 목적.
3. **접속 감사 로그** — `demo/access.log`(`.gitignore`, IP 기록이라 커밋 금지).
   Starlette 미들웨어로 페이지 접속·로그인 성공/실패를 IP·시각과 함께 기록.
   share 터널 경유 시 `fwd=`에 실제 접속자 IP가 찍힘. 예:
   `[2026-06-03 14:41:42] ACCESS page ip=172.31.x fwd=175.125.x status=200`

## 시연 흐름

1. **공격 선택** → 트리거 샘플 갤러리 자동 로드
2. 샘플 클릭 → 왼쪽 Clean(정상 예측) / 오른쪽 Triggered(타깃으로 오분류) 비교
3. **방어 선택**(FT/NAD/ANP/ABL) → Triggered 이미지가 정상 복구되는지 확인
4. **ASR 비교 막대**(40샘플 평균): 공격 100% → 방어 후 얼마나 떨어지나
5. **Grad-CAM 체크박스**: 트리거가 모델 attention을 어떻게 탈취하는지 시각화

### 🎬 라이브 입력 (실시간성)

- **🎲 랜덤 이미지 버튼**: 누를 때마다 CIFAR에서 무작위 원본 → 즉석 트리거 → 예측.
- **이미지 업로드**: 아무 이미지나 올리면 실시간 트리거 적용 → 오분류 시연.
- **트리거 강도 슬라이더** (SIG·Blended만): 진폭/알파를 드래그하면 트리거 세기와
  예측이 실시간으로 바뀜. "약하면 안 속고 강하면 속는다"를 눈앞에서 시연.
- ⚠️ 랜덤/업로드/강도는 **재생성 공격(BadNets·Blended·LF·SIG)에서만** 동작.
  WaNet·InputAware는 warping grid 의존이라 갤러리 샘플만 사용.

### 예열 (pre-warm)

서버 시작 시 6공격 + 24방어 모델 + ASR표를 미리 로딩(약 7초). 시연 중 모든
클릭이 즉시 반응. 빠른 시작이 필요하면 `--no-prewarm`.

### 검증된 결과 (40샘플 ASR)

| 공격 | 방어없음 | FT | NAD | ANP | ABL |
|---|---|---|---|---|---|
| WaNet | 100% | 25% | 12% | 0% | 50% |
| InputAware | 100% | 12% | 50% | 0% | 100% |
| BadNets | 100% | 0% | 0% | 0% | 0% |
| Blended | 100% | **100%** | 75% | 0% | 0% |
| LF | 100% | 62% | 38% | 0% | 0% |
| SIG | 100% | 0% | 12% | 0% | 0% |

> 이야깃거리: **ANP는 만능**(전부 0%), **BadNets는 모든 방어에 취약**,
> **Blended는 FT를 100% 뚫음**(가장 강한 공격) — BadNets와 정반대.

## frpc (gradio share 터널) 주의사항 — 시연용 임시 도구

`--share`는 `frpc`라는 reverse-proxy 클라이언트로 터널을 만든다.

- 출처: HuggingFace 공식 CDN, gradio 소스에 SHA256 해시 박힘(변조 시 실행 거부). 악성 아님.
- 위치: `C:\Users\admin\.cache\huggingface\gradio\frpc\` (conda env가 아니라 **홈 캐시 전역**)
- Windows Defender가 터널링 도구를 PUA로 분류해 **차단할 수 있음**.

### 차단 시 (Defender 예외) — 관리자 PowerShell

```powershell
$dir = "C:\Users\admin\.cache\huggingface\gradio\frpc"
Add-MpPreference -ExclusionPath $dir
Add-MpPreference -ExclusionProcess "$dir\frpc_windows_amd64_v0.3"
```

### 시연 끝나고 정리 (호스트 깨끗하게)

```powershell
# 1) 데모 서버 종료: 실행 창에서 Ctrl+C
# 2) frpc 삭제
Remove-Item "C:\Users\admin\.cache\huggingface\gradio\frpc" -Recurse -Force
# 3) Defender 예외 제거
Remove-MpPreference -ExclusionPath "C:\Users\admin\.cache\huggingface\gradio\frpc"
Remove-MpPreference -ExclusionProcess "C:\Users\admin\.cache\huggingface\gradio\frpc\frpc_windows_amd64_v0.3"
```

## 보안 메모 (발표 때 언급 포인트)

- 공개 URL은 **인증 없음** — URL 아는 사람은 내 GPU로 추론 가능. **시연 중에만 켜고 끝나면 종료.**
- 서버는 `127.0.0.1`에만 바인딩 — LAN 노출 없음. 외부 노출은 오직 frpc 터널로만.
- 시연 데이터/가중치는 GitHub에 안 올라감(`.gitignore`). 코드만 공개.

## 백업 플랜 (share 실패 / 강의실 네트워크 차단 대비)

라이브 시연이 죽는 1순위는 **네트워크**다. 순서대로 대비:

1. **폰 핫스팟**: 강의실 PC를 폰 LTE 핫스팟에 연결 → 학교 방화벽 우회. gradio.live가
   학교망에서 막혀도 살림. (집 PC는 그대로 share 서버)
2. **localhost 직결**: 집 PC를 강의실에 가져가 HDMI 연결 → `python demo/app.py`(로컬).
   인터넷 0 의존. 가장 안전.
3. **녹화 영상 + HTML 리포트**: URL이 끝내 죽어도 발표는 진행. `WANET_REPORT.html`,
   `INPUTAWARE_REPORT.html` + 미리 녹화한 데모 클립. **무조건 챙길 최후 백업.**

## 리허설 체크리스트 (시연 전날 필수)

- [ ] `conda activate backdoor310` → `python demo/app.py --share` 로 공개 URL 뜨는지
- [ ] **강의실과 같은 네트워크**(또는 같은 PC)에서 그 URL이 열리는지 ← 제일 중요
- [ ] 폰 핫스팟으로도 URL 열리는지 (백업 검증)
- [ ] 6개 공격 전부 클릭 → clean 정상 / triggered 오분류 뜨는지
- [ ] 방어 4종 → ASR 떨어지는지, Grad-CAM 뜨는지
- [ ] 집 PC 절전/슬립 끄기 (시연 내내 깨어 있어야 함)
- [ ] 데모 클립 녹화해두기 (최후 백업)

## 데모 파일 구조

```
demo/
  app.py          # Gradio UI (실행 진입점)
  core.py         # 모델 로딩·트리거 생성·추론·Grad-CAM·NC 탐지
  README.md       # 이 문서
  _verify*.py     # 사전 검증 스크립트 (6공격 ASR 확인용, 시연엔 불필요)
```
