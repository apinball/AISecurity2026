"""
백도어 공격/방어 실시간 시연 데모 (Gradio).

실행:
    conda activate backdoor310
    cd AISecurity2026
    $env:DEMO_USER="myid"; $env:DEMO_PASS="mypw!"   # 로그인 (환경변수)
    python demo/app.py            # 로컬만 (http://127.0.0.1:7860)
    python demo/app.py --share    # 공개 URL (xxxx.gradio.live)
    python demo/app.py --share --no-prewarm   # 예열 건너뛰기(빠른 시작)

시연 흐름:
  1. 공격 선택 -> 샘플 자동 로드 (왼: clean, 오: triggered)
  2. clean은 정상 예측, triggered는 타깃(airplane)으로 오분류 확인
  3. 방어 선택 -> triggered 이미지가 다시 정상 복구되는지 확인 (+ASR 표)
  4. [라이브] 랜덤 샘플 / 이미지 업로드 / 트리거 강도 슬라이더
  5. Grad-CAM 토글로 트리거가 모델 attention을 어떻게 탈취하는지 시각화
"""
import os, sys, argparse, time, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from PIL import Image

import demo.core as core
from demo.core import (CLASSES, ATTACK_TARGET, ATTACK_PT, REGEN_ATTACKS,
                       DATASET_ATTACKS, STRENGTH_PARAM)

ATTACKS = list(ATTACK_PT.keys())
TARGET_NAME = CLASSES[ATTACK_TARGET]
ACCESS_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "access.log")

ATTACK_INFO = {
    "WaNet":      "Imperceptible warping 트리거. 학습된 warping grid 사용.",
    "InputAware": "샘플마다 다른 동적 트리거(generator).",
    "BadNets":    "고정 패치(우하단 사각형) 덧씌움. 가장 고전적.",
    "Blended":    "Hello-Kitty 이미지를 alpha=0.2로 전역 블렌딩. 눈에 잘 안 띔.",
    "LF":         "저주파(Low-Frequency) 패턴을 더함. 주파수 영역 트리거.",
    "SIG":        "가로 sin파(진폭40, 주파수6) 오버레이. 라벨 변조 없는 공격.",
}


# ---------------------------------------------------------------------------
# 입력값 검증 (화이트리스트)
# ---------------------------------------------------------------------------
ALLOWED_ATTACKS = set(ATTACKS)
ALLOWED_DEFENSES = {"없음"} | set(core.REPAIR_DEFENSES)


def _validate_inputs(attack, defense, sample_idx):
    """사용자 입력값 화이트리스트 검증. 정해진 값만 통과시켜 주입을 원천 차단."""
    if attack not in ALLOWED_ATTACKS:
        return False, None, None, None, f"허용되지 않은 공격: {attack!r}"
    if defense not in ALLOWED_DEFENSES:
        return False, None, None, None, f"허용되지 않은 방어: {defense!r}"
    try:
        idx = int(sample_idx)
    except (TypeError, ValueError):
        idx = 0
    return True, attack, defense, max(0, idx), ""


def _clamp_strength(attack, strength):
    """강도 슬라이더 값 범위 강제(검증). 미지원 공격은 None."""
    if attack not in STRENGTH_PARAM:
        return None
    _, default, lo, hi, _ = STRENGTH_PARAM[attack]
    try:
        v = float(strength)
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, v))


# ---------------------------------------------------------------------------
# 샘플 로딩
# ---------------------------------------------------------------------------
def load_samples(attack):
    """공격 선택 시 갤러리용 (clean_pil, bd_pil, label) 리스트."""
    if attack in REGEN_ATTACKS:
        return list(core.get_regen_samples(attack, k=8))
    s = core.get_dataset_samples(attack, k=8)
    return [(c, b, lab) for (c, b, lab, _, _) in s]


# ---------------------------------------------------------------------------
# 핵심 추론
# ---------------------------------------------------------------------------
def _infer_pair(attack, defense, show_cam, clean_pil, bd_pil, label,
                clean_in=None, bd_in=None):
    """clean/bd 한 쌍에 대해 공격(+방어) 추론 + 결과 8-튜플 반환."""
    atk_model = core.get_attack_model(attack)
    if clean_in is None:
        clean_in = clean_pil.resize((32, 32))
    if bd_in is None:
        bd_in = bd_pil

    pc, probs_c = core.predict(atk_model, clean_in)
    pb, probs_b = core.predict(atk_model, bd_in)

    defense_md = ""
    bd_after = {}
    if defense in core.REPAIR_DEFENSES:
        dmodel = core.get_defense_model(attack, defense)
        pd, probs_d = core.predict(dmodel, bd_in)
        bd_after = probs_d
        recovered = "✅ 복구 성공" if pd == label else (
            "❌ 여전히 타깃" if pd == ATTACK_TARGET else f"⚠️ 다른 클래스({CLASSES[pd]})")
        defense_md = (f"\n### 🛡️ 방어: {defense}\n"
                      f"- 트리거 이미지 예측: **{CLASSES[pd]}** ({recovered})\n")

    lab_txt = CLASSES[label] if label is not None else "(업로드)"
    attack_ok = "✅ 공격 성공" if pb == ATTACK_TARGET else "⚠️ 공격 실패"
    summary = (
        f"### 🎯 공격: {attack}\n"
        f"- 정답 라벨: **{lab_txt}**\n"
        f"- 🟢 Clean 예측: **{CLASSES[pc]}**"
        f"{' ✅' if label is not None and pc == label else ''}\n"
        f"- 🔴 Triggered 예측: **{CLASSES[pb]}** ({attack_ok}, 타깃=`{TARGET_NAME}`)\n"
        + defense_md
    )

    cam_clean = cam_bd = None
    if show_cam:
        cam_clean = core.gradcam(atk_model, clean_in).resize((320, 320), Image.NEAREST)
        cam_bd = core.gradcam(atk_model, bd_in).resize((320, 320), Image.NEAREST)

    asr_bars = {"공격(방어없음)": core.measure_asr(attack, None, n=40)}
    for d in core.REPAIR_DEFENSES:
        asr_bars[f"{d} 방어후"] = core.measure_asr(attack, d, n=40)

    bd_probs_show = bd_after if bd_after else probs_b
    # 현재 보고 있는 이미지 쌍을 저장(방어/Grad-CAM 변경 시 재사용 → 이미지 유지)
    cur = {"attack": attack, "clean_pil": clean_pil, "bd_pil": bd_pil,
           "label": label, "clean_in": clean_in, "bd_in": bd_in}
    return (clean_pil.resize((320, 320), Image.NEAREST),
            bd_pil.resize((320, 320), Image.NEAREST),
            probs_c, bd_probs_show, summary, cam_clean, cam_bd, asr_bars, cur)


# 빈/에러 반환 (9개: 출력8 + cur state)
def _empty(msg):
    return (None, None, {}, {}, msg, None, None, {}, None)


def run_inference(attack, sample_idx, defense, show_cam, state):
    """갤러리에서 선택한 샘플로 추론."""
    ok, attack, defense, sample_idx, err = _validate_inputs(attack, defense, sample_idx)
    if not ok:
        return _empty(f"⛔ {err}")
    show_cam = bool(show_cam)
    if state is None or state.get("attack") != attack:
        return _empty("샘플을 먼저 선택하세요.")

    triples = state["triples"]
    if sample_idx >= len(triples):
        sample_idx = 0
    clean_pil, bd_pil, label = triples[sample_idx]

    # dataset 공격은 정규화 텐서 캐시 사용(정확도 보장)
    clean_in = bd_in = None
    if attack in DATASET_ATTACKS:
        s = core.get_dataset_samples(attack, k=8)
        _, _, _, clean_in, bd_in = s[sample_idx]
    return _infer_pair(attack, defense, show_cam, clean_pil, bd_pil, label,
                       clean_in, bd_in)


def run_custom(attack, defense, show_cam, strength, clean_pil):
    """랜덤/업로드 이미지에 강도 적용해 즉석 트리거 추론 (재생성 공격만)."""
    ok, attack, defense, _, err = _validate_inputs(attack, defense, 0)
    if not ok:
        return _empty(f"⛔ {err}")
    if clean_pil is None:
        return _empty("이미지를 먼저 올리거나 랜덤 버튼을 누르세요.")
    if attack not in REGEN_ATTACKS:
        return _empty(f"⚠️ {attack}는 즉석 트리거 미지원(grid 의존). "
                      "BadNets·Blended·LF·SIG에서 사용하세요.")

    clean_pil = clean_pil.convert("RGB").resize((32, 32))
    sv = _clamp_strength(attack, strength)
    if sv is not None:
        bd_pil = core.apply_trigger_strength(attack, clean_pil, sv)
    else:
        bd_pil = core.apply_trigger(attack, clean_pil)
    return _infer_pair(attack, defense, bool(show_cam), clean_pil, bd_pil, None)


def on_param_change(defense, show_cam, cur):
    """방어/Grad-CAM만 바꿀 때: 현재 보고 있는 이미지(cur)로 재추론 → 이미지 유지."""
    if not cur or cur.get("clean_pil") is None:
        return _empty("먼저 샘플을 선택하거나 랜덤/업로드 하세요.")
    attack = cur["attack"]
    ok, attack, defense, _, err = _validate_inputs(attack, defense, 0)
    if not ok:
        return _empty(f"⛔ {err}")
    return _infer_pair(attack, defense, bool(show_cam),
                       cur["clean_pil"], cur["bd_pil"], cur["label"],
                       cur.get("clean_in"), cur.get("bd_in"))


def on_random(attack, defense, show_cam, strength, state):
    """랜덤 샘플로 추론.
    - 재생성 공격: CIFAR 임의 원본 → 즉석 트리거
    - WaNet/IA: 갤러리(dataset 샘플) 중 임의 인덱스 선택
    반환: _infer_pair 9개 + 업로드 미리보기 1개 = 10개"""
    ok, attack2, defense, _, err = _validate_inputs(attack, defense, 0)
    if not ok:
        return _empty(f"⛔ {err}") + (None,)

    if attack2 in REGEN_ATTACKS:
        # CIFAR 1만장 전체에서 진짜 랜덤 → 즉석 트리거
        clean_pil, _ = core.random_clean_sample()
        out = run_custom(attack2, defense, show_cam, strength, clean_pil)
        return out + (clean_pil.resize((160, 160), Image.NEAREST),)
    else:
        # WaNet/IA: bd_test 9000장 전체 풀에서 진짜 랜덤
        clean_pil, bd_pil, label, clean_in, bd_in = core.random_dataset_sample(attack2)
        out = _infer_pair(attack2, defense, bool(show_cam),
                          clean_pil, bd_pil, label, clean_in, bd_in)
        return out + (None,)  # 업로드 미리보기 칸 비움


def on_attack_change(attack):
    """공격 변경 시 갤러리 + 강도 슬라이더 가시성 갱신."""
    triples = load_samples(attack)
    gallery = [(b.resize((160, 160), Image.NEAREST), f"{CLASSES[lab]} → {TARGET_NAME}")
               for (c, b, lab) in triples]
    state = {"attack": attack, "triples": triples}
    live = "✅ 랜덤/업로드 즉석 트리거 지원" if attack in REGEN_ATTACKS else \
           "ℹ️ 이 공격은 갤러리 샘플만 (grid 의존, 즉석 트리거 미지원)"
    info = f"**{attack}** — {ATTACK_INFO[attack]}\n\n{live}"
    # 강도 슬라이더 (SIG/Blended만)
    if attack in STRENGTH_PARAM:
        lbl, default, lo, hi, step = STRENGTH_PARAM[attack]
        sld = gr.update(visible=True, label=f"트리거 강도 — {lbl}",
                        minimum=lo, maximum=hi, step=step, value=default)
    else:
        sld = gr.update(visible=False)
    return gallery, state, info, 0, sld


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="백도어 공격/방어 시연") as demo:
        gr.Markdown(
            "# 🔓 백도어 공격 & 방어 실시간 시연\n"
            "CIFAR-10 / PreActResNet18 · 6개 공격 × 4개 방어(FT·NAD·ANP·ABL)\n"
            f"> 모든 공격은 **all-to-one**: 트리거가 박히면 어떤 이미지든 **`{TARGET_NAME}`**(타깃)로 분류됩니다."
        )
        state = gr.State(None)
        cur = gr.State(None)   # 현재 보고 있는 이미지 쌍 (방어/캠 변경 시 유지용)

        with gr.Row():
            attack = gr.Radio(ATTACKS, value=ATTACKS[0], label="① 공격 선택")
            defense = gr.Radio(["없음"] + core.REPAIR_DEFENSES, value="없음",
                               label="② 방어 선택 (복구형)")
            show_cam = gr.Checkbox(False, label="Grad-CAM 시각화")

        info = gr.Markdown()

        # --- 라이브 입력 패널 ---
        with gr.Accordion("🎬 라이브 입력 (랜덤 / 업로드 / 강도)", open=True):
            with gr.Row():
                random_btn = gr.Button("🎲 랜덤 이미지", variant="primary")
                upload = gr.Image(label="이미지 업로드 (클릭/드래그)", type="pil",
                                  height=120, sources=["upload", "clipboard"])
            strength = gr.Slider(visible=False, label="트리거 강도",
                                 minimum=5, maximum=80, step=5, value=40)

        with gr.Row():
            gallery = gr.Gallery(label="또는 트리거 샘플 클릭", columns=4, rows=2,
                                 height=320, object_fit="contain", allow_preview=False)
        sample_idx = gr.State(0)

        with gr.Row(equal_height=True):
            with gr.Column():
                gr.Markdown("### 🟢 Clean 이미지")
                clean_img = gr.Image(label="원본", height=320, interactive=False)
                clean_probs = gr.Label(label="예측 확률", num_top_classes=3)
            with gr.Column():
                gr.Markdown("### 🔴 Triggered 이미지 (동일 원본 + 트리거)")
                bd_img = gr.Image(label="트리거 적용", height=320, interactive=False)
                bd_probs = gr.Label(label="예측 확률(방어 시 방어모델 기준)", num_top_classes=3)

        summary = gr.Markdown()
        gr.Markdown("### 📊 방어별 ASR 비교 (40샘플 평균) — 낮을수록 방어 성공")
        asr_plot = gr.Label(label="ASR (공격 성공률)")

        with gr.Row():
            cam_clean = gr.Image(label="Grad-CAM (clean)", height=280, interactive=False)
            cam_bd = gr.Image(label="Grad-CAM (triggered)", height=280, interactive=False)

        # 출력 묶음 (9개: 화면8 + cur state)
        outputs = [clean_img, bd_img, clean_probs, bd_probs, summary,
                   cam_clean, cam_bd, asr_plot, cur]
        # 라이브 출력 = 9개 + upload 미리보기
        live_outputs = outputs + [upload]

        # --- 이벤트 ---
        attack.change(on_attack_change, [attack],
                      [gallery, state, info, sample_idx, strength]).then(
                      run_inference, [attack, sample_idx, defense, show_cam, state], outputs)

        gallery.select(lambda evt: evt.index, None, sample_idx).then(
            run_inference, [attack, sample_idx, defense, show_cam, state], outputs)

        # 방어/Grad-CAM 변경: 현재 보고 있는 이미지(cur) 유지하며 재추론
        defense.change(on_param_change, [defense, show_cam, cur], outputs)
        show_cam.change(on_param_change, [defense, show_cam, cur], outputs)

        # 라이브: 랜덤 버튼
        random_btn.click(on_random, [attack, defense, show_cam, strength, state], live_outputs)
        # 라이브: 업로드
        upload.upload(run_custom, [attack, defense, show_cam, strength, upload], outputs)
        # 강도 슬라이더 변경 시 업로드 이미지로 재적용
        strength.release(run_custom, [attack, defense, show_cam, strength, upload], outputs)

        demo.load(on_attack_change, [attack],
                  [gallery, state, info, sample_idx, strength]).then(
                  run_inference, [attack, sample_idx, defense, show_cam, state], outputs)

    return demo


# ---------------------------------------------------------------------------
# 접속 감사 로그 (Starlette 미들웨어 — launch 전에 app_kwargs로 주입)
# ---------------------------------------------------------------------------
def _audit_log(line):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ACCESS_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {line}\n")


def _make_access_middleware():
    from starlette.middleware.base import BaseHTTPMiddleware

    class AccessLogMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            ip = request.client.host if request.client else "?"
            fwd = request.headers.get("x-forwarded-for")  # 터널 경유 실제 IP
            path = request.url.path
            resp = await call_next(request)
            if path == "/login" and request.method == "POST":
                ok = "성공" if resp.status_code == 200 else f"실패({resp.status_code})"
                _audit_log(f"LOGIN {ok} ip={ip} fwd={fwd}")
            elif path in ("/", "") and request.method == "GET":
                _audit_log(f"ACCESS page ip={ip} fwd={fwd} status={resp.status_code}")
            return resp

    return AccessLogMiddleware


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="공개 URL 생성")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-prewarm", action="store_true", help="시작 예열 건너뛰기")
    args = ap.parse_args()

    print(f"[demo] DEVICE={core.DEVICE}", flush=True)

    # 로그인 ID/PW (환경변수)
    demo_user = os.environ.get("DEMO_USER", "demo")
    demo_pass = os.environ.get("DEMO_PASS", "backdoor2026!")
    if demo_pass == "backdoor2026!":
        print("[demo] ⚠️ 기본 비밀번호. 시연 전 DEMO_USER/DEMO_PASS 환경변수 설정 권장.", flush=True)

    # 예열 (시연 중 클릭 멈춤 제거)
    if not args.no_prewarm:
        print("[demo] 예열 시작 (모든 공격·방어 모델 + ASR 미리 로딩)...", flush=True)
        t0 = time.time()
        for msg in core.prewarm():
            print(f"[demo]   {msg}", flush=True)
        print(f"[demo] 예열 완료 ({time.time()-t0:.0f}s)", flush=True)

    demo = build_ui()
    print(f"[demo] launching (share={args.share}, port={args.port}, auth=ON) ...", flush=True)
    print(f"[demo] access log: {ACCESS_LOG}", flush=True)

    # 접속 감사 로그 미들웨어를 launch 전에 주입 (app_kwargs)
    from starlette.middleware import Middleware
    access_mw = [Middleware(_make_access_middleware())]

    demo.launch(share=args.share, server_port=args.port,
                server_name="127.0.0.1", show_error=True,
                auth=(demo_user, demo_pass),
                auth_message="백도어 시연 데모 — 로그인이 필요합니다.",
                app_kwargs={"middleware": access_mw})
