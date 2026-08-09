"""
카메라 기반 Feature Squeezing 방어 + 적대적 공격 생성 테스트 (실시간 모드)

카메라에서 얼굴을 감지·크롭(160×160)한 뒤 Feature Squeezing을 적용하여
적대적 perturbation을 실시간 탐지한다.

탐지 원리:
  정상 얼굴  : squeezing 전후 embedding 변화 작음 → sim_diff 낮음 → ACCEPT
  적대적 공격: squeezing으로 perturbation 제거됨  → sim_diff 높음 → REJECT

키 조작:
    E / ㄷ : 현재 프레임 얼굴 감지 → 등록 (인증 기준 얼굴)
    C / ㅊ : 현재 프레임 캡처 → 방어 탐지 실행
    A / ㅁ : 현재 프레임 얼굴로 적대적 공격 생성 → 방어 탐지 (공격 시뮬레이션)
    Q / ㅂ : 종료

실행:
    python -m src.verification.defenses.verification_defense_temporal_camera
    python -m src.verification.defenses.verification_defense_temporal_camera --enroll-img face.jpg
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw, ImageFont

from src.verification.defenses.facenet_embed import get_embedding, cosine_similarity, get_model, preprocess
from src.verification.defenses.verification_defense_feature_squeezing import (
    squeeze_low_resolution,
    squeeze_color_depth,
    squeeze_median_filter,
)

# ── 상수 ──────────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD  = 0.47966246581077576
FACE_SIZE          = 160
FPS                = 15

# squeezer별 임계값 (카메라 환경 캘리브레이션)
SIM_DIFF_THRESHOLDS = {
    "low_resolution": 0.40,
    "color_depth":    0.08,
    "median_filter":  0.05,
}

# PGD 공격 파라미터
ADV_EPSILON   = 0.03   # 최대 perturbation 크기 (정규화 공간) — 작을수록 squeezing에 취약
ADV_STEPS     = 40     # PGD 반복 횟수
ADV_STEP_SIZE = 0.002  # PGD 스텝 크기

KEYS_ENROLL = {ord('e'), ord('E'), ord('ㄷ')}
KEYS_CHECK  = {ord('c'), ord('C'), ord('ㅊ')}
KEYS_ATTACK = {ord('a'), ord('A'), ord('ㅁ')}
KEYS_QUIT   = {ord('q'), ord('Q'), ord('ㅂ')}

_FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/NotoSansGothic-Regular.ttf",
]

SQUEEZERS = {
    "low_resolution": squeeze_low_resolution,
    "color_depth":    squeeze_color_depth,
    "median_filter":  squeeze_median_filter,
}


# ── 폰트·텍스트 ───────────────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


_FONT_SM = _load_font(20)
_FONT_MD = _load_font(26)
_FONT_LG = _load_font(40)


def _put(frame_bgr: np.ndarray, text: str, y: int,
         color_bgr=(255, 255, 255), font=None) -> np.ndarray:
    if font is None:
        font = _FONT_SM
    pil  = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]
    draw.text((11, y + 1), text, font=font, fill=(0, 0, 0))
    draw.text((10, y),     text, font=font, fill=(r, g, b))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _draw_face_box(frame: np.ndarray, box) -> np.ndarray:
    if box is None:
        return frame
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 180), 2)
    return frame


# ── MTCNN 얼굴 크롭 ───────────────────────────────────────────────────────────

_mtcnn = None


def _get_mtcnn(device=None):
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(image_size=FACE_SIZE, margin=20, keep_all=False, device=device)
    return _mtcnn


def detect_and_crop(frame_bgr: np.ndarray, device=None) -> tuple[Image.Image | None, list | None]:
    """MTCNN으로 얼굴을 감지하고 160×160으로 크롭한다."""
    mtcnn = _get_mtcnn(device)
    pil   = _bgr_to_pil(frame_bgr)
    boxes, _ = mtcnn.detect(pil)
    if boxes is None:
        return None, None
    face = mtcnn(pil)
    if face is None:
        return None, None
    face_np  = ((face.permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(face_np), boxes[0].tolist()


# ── 적대적 공격 생성 (PGD) ────────────────────────────────────────────────────

def generate_adversarial(
    source_img: Image.Image,
    target_emb: torch.Tensor,
    epsilon: float = ADV_EPSILON,
    n_steps: int   = ADV_STEPS,
    step_size: float = ADV_STEP_SIZE,
    device=None,
) -> tuple[Image.Image, float]:
    """
    source_img에 PGD 공격을 적용하여 target_emb(등록 얼굴)와 유사하게 만든다.

    Returns:
        adversarial PIL image (160×160)
        adversarial similarity (공격 후 유사도)
    """
    model, dev = get_model(device)
    target = target_emb.to(dev).unsqueeze(0)

    x_orig = preprocess(source_img).to(dev)   # (1, 3, 160, 160) 정규화 완료
    x_adv  = x_orig.clone()

    for _ in range(n_steps):
        x_adv.requires_grad_(True)
        emb  = model(x_adv)
        emb  = F.normalize(emb, p=2, dim=1)
        loss = -F.cosine_similarity(emb, target)   # 유사도 최대화

        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            x_adv = x_adv - step_size * grad.sign()
            delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
            x_adv = (x_orig + delta).detach()

    with torch.no_grad():
        emb_adv = F.normalize(model(x_adv), p=2, dim=1)
        sim_adv = float(F.cosine_similarity(emb_adv, target).item())

    # 정규화 역변환: pixel = norm * 128.0 + 127.5
    adv_np = x_adv.squeeze(0).permute(1, 2, 0).cpu().numpy()
    adv_np = (adv_np * 128.0 + 127.5).clip(0, 255).astype(np.uint8)

    return Image.fromarray(adv_np), sim_adv


# ── Feature Squeezing 탐지 ────────────────────────────────────────────────────

def feature_squeezing_check(
    face_img: Image.Image,
    enroll_img: Image.Image,
    threshold: float = DEFAULT_THRESHOLD,
    device=None,
) -> dict:
    """160×160 얼굴 크롭에 Feature Squeezing을 적용하여 적대적 공격을 탐지한다."""
    enroll_emb   = get_embedding(enroll_img, device)
    original_emb = get_embedding(face_img, device)
    sim_original = cosine_similarity(original_emb, enroll_emb)

    detections = {}
    for name, fn in SQUEEZERS.items():
        squeezed     = fn(face_img)
        squeezed_emb = get_embedding(squeezed, device)
        sim_squeezed = cosine_similarity(squeezed_emb, enroll_emb)
        sim_diff     = abs(sim_original - sim_squeezed)
        thresh       = SIM_DIFF_THRESHOLDS[name]
        detections[name] = {
            "sim_squeezed": round(sim_squeezed, 6),
            "sim_diff":     round(sim_diff, 6),
            "threshold":    thresh,
            "detected":     sim_diff >= thresh,
        }

    n_detected     = sum(1 for v in detections.values() if v["detected"])
    is_adversarial = n_detected >= 2

    if is_adversarial:
        accepted, reject_reason = False, "adversarial_input"
    elif sim_original < threshold:
        accepted, reject_reason = False, "low_similarity"
    else:
        accepted, reject_reason = True, None

    return {
        "accepted":       accepted,
        "reject_reason":  reject_reason,
        "sim_original":   round(sim_original, 6),
        "detections":     detections,
        "n_detected":     n_detected,
        "is_adversarial": is_adversarial,
    }


# ── 화면 오버레이 ─────────────────────────────────────────────────────────────

def _overlay_result(frame: np.ndarray, result: dict, label: str = "") -> np.ndarray:
    if result["accepted"]:
        verdict, color = "ACCEPT", (0, 200, 0)
    else:
        verdict, color = f"REJECT  [{result['reject_reason']}]", (0, 0, 220)

    if label:
        frame = _put(frame, label, 20, (200, 200, 200), _FONT_SM)

    frame = _put(frame, verdict, 48, color, _FONT_LG)
    frame = _put(frame, f"원본 유사도: {result['sim_original']:.4f}", 100, (255, 220, 0), _FONT_SM)

    y = 126
    for name, info in result["detections"].items():
        flag = "★ 탐지" if info["detected"] else "  정상"
        col  = (0, 80, 255) if info["detected"] else (180, 255, 180)
        frame = _put(frame,
                     f"{flag}  {name}: {info['sim_diff']:.4f}  (임계값 {info['threshold']})",
                     y, col, _FONT_SM)
        y += 26
    return frame


def _print_result(result: dict, label: str = "") -> None:
    if label:
        print(f"\n[{label}]")
    print(f"  원본 유사도   : {result['sim_original']:.6f}")
    for name, info in result["detections"].items():
        flag = "★탐지" if info["detected"] else "  정상"
        print(f"  {flag}  {name}: sim_diff={info['sim_diff']:.6f}  (임계값 {info['threshold']})")
    print(f"  탐지 squeezer : {result['n_detected']}/3")
    print(f"  => {'ACCEPT' if result['accepted'] else 'REJECT: ' + str(result['reject_reason'])}")


# ── 메인 실행 ─────────────────────────────────────────────────────────────────

def run_camera(
    enroll_img_path: str | None = None,
    device=None,
) -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다.")

    enroll_face: Image.Image | None = None
    enroll_emb:  torch.Tensor | None = None

    if enroll_img_path:
        raw = Image.open(enroll_img_path).convert("RGB")
        arr = cv2.cvtColor(np.array(raw), cv2.COLOR_RGB2BGR)
        enroll_face, _ = detect_and_crop(arr, device)
        if enroll_face:
            enroll_emb = get_embedding(enroll_face, device)
            print(f"[등록] {enroll_img_path}")
        else:
            print(f"[경고] 등록 이미지에서 얼굴을 찾지 못했습니다.")

    last_result: dict | None = None
    last_box    = None
    last_label  = ""
    hint = "E/ㄷ: 등록   C/ㅊ: 검사   A/ㅁ: 공격생성   Q/ㅂ: 종료"
    face_status = ""   # "얼굴 없음" 등 상태 메시지

    print("=== Feature Squeezing 방어 (카메라 실시간) ===")
    print("E/ㄷ: 등록  |  C/ㅊ: 탐지 실행  |  A/ㅁ: 공격 생성 테스트  |  Q/ㅂ: 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        disp = frame.copy()
        if last_box:
            _draw_face_box(disp, last_box)

        disp = _put(disp, hint, 10, (255, 255, 255), _FONT_SM)

        if face_status:
            disp = _put(disp, face_status, 40, (0, 100, 255), _FONT_SM)
        elif enroll_face is not None and last_result is None:
            disp = _put(disp, "등록 완료. C/ㅊ: 검사  A/ㅁ: 공격 테스트", 40, (0, 255, 220), _FONT_SM)

        if last_result is not None:
            disp = _overlay_result(disp, last_result, last_label)

        cv2.imshow("Feature Squeezing Defense", disp)
        key = cv2.waitKey(1) & 0xFF

        if key in KEYS_QUIT:
            break

        # ── 등록 ──────────────────────────────────────────────────────────────
        elif key in KEYS_ENROLL:
            ret2, ef = cap.read()
            if ret2:
                face, box = detect_and_crop(ef, device)
                if face is None:
                    face_status = "얼굴을 찾지 못했습니다. 정면을 바라보세요."
                    print("[경고] 얼굴을 찾지 못했습니다.")
                else:
                    enroll_face  = face
                    enroll_emb   = get_embedding(face, device)
                    last_result  = None
                    last_box     = box
                    face_status  = ""
                    hint = "등록 완료.  C/ㅊ: 검사   A/ㅁ: 공격 테스트   Q/ㅂ: 종료"
                    print("[등록 완료] 160×160 얼굴 크롭 저장됨")

        # ── 일반 검사 ─────────────────────────────────────────────────────────
        elif key in KEYS_CHECK:
            if enroll_face is None:
                print("[경고] 먼저 E/ㄷ 로 등록하세요.")
                continue
            ret2, cf = cap.read()
            if not ret2:
                continue

            disp2 = _put(cf.copy(), "얼굴 감지 중...", 40, (0, 255, 220), _FONT_MD)
            cv2.imshow("Feature Squeezing Defense", disp2)
            cv2.waitKey(1)

            face, box = detect_and_crop(cf, device)
            if face is None:
                face_status = "얼굴을 찾지 못했습니다."
                print("[경고] 얼굴을 찾지 못했습니다.")
                continue

            face_status = ""
            last_box    = box
            disp2 = _put(cf.copy(), "분석 중...", 40, (0, 255, 220), _FONT_MD)
            cv2.imshow("Feature Squeezing Defense", disp2)
            cv2.waitKey(1)

            result = feature_squeezing_check(face, enroll_face, device=device)
            last_result = result
            last_label  = "일반 검사"
            _print_result(result, "일반 검사")
            hint = "E/ㄷ: 재등록   C/ㅊ: 재검사   A/ㅁ: 공격 테스트   Q/ㅂ: 종료"

        # ── 적대적 공격 생성 테스트 ───────────────────────────────────────────
        elif key in KEYS_ATTACK:
            if enroll_face is None or enroll_emb is None:
                print("[경고] 먼저 E/ㄷ 로 등록하세요.")
                continue
            ret2, af = cap.read()
            if not ret2:
                continue

            disp2 = _put(af.copy(), "얼굴 감지 중...", 40, (0, 200, 255), _FONT_MD)
            cv2.imshow("Feature Squeezing Defense", disp2)
            cv2.waitKey(1)

            face, box = detect_and_crop(af, device)
            if face is None:
                face_status = "얼굴을 찾지 못했습니다."
                print("[경고] 얼굴을 찾지 못했습니다.")
                continue

            last_box    = box
            face_status = ""

            # 공격 전 유사도 확인
            sim_before = cosine_similarity(get_embedding(face, device), enroll_emb)
            print(f"\n[공격 전] 유사도: {sim_before:.6f}")

            # PGD 공격 생성
            print(f"[PGD 공격 생성 중... ({ADV_STEPS}스텝)]")
            disp2 = _put(af.copy(), f"공격 생성 중... (PGD {ADV_STEPS}스텝)", 40, (0, 200, 255), _FONT_MD)
            cv2.imshow("Feature Squeezing Defense", disp2)
            cv2.waitKey(1)

            adv_face, sim_after = generate_adversarial(face, enroll_emb, device=device)
            print(f"[공격 후] 유사도: {sim_after:.6f}  (등록 얼굴과 유사해졌는지 확인)")

            # 방어 탐지
            result = feature_squeezing_check(adv_face, enroll_face, device=device)
            last_result = result
            last_label  = f"공격 테스트 (공격 전 sim={sim_before:.3f} → 후 sim={sim_after:.3f})"
            _print_result(result, "공격 테스트")
            hint = "E/ㄷ: 재등록   C/ㅊ: 일반 검사   A/ㅁ: 공격 재생성   Q/ㅂ: 종료"

    cap.release()
    cv2.destroyAllWindows()


def run_file_test(enroll_img_path: str, adv_img_path: str, clean_img_path: str | None = None, device=None) -> None:
    """
    기존 공격 파일로 Feature Squeezing 탐지를 테스트한다.

    enroll_img_path : 등록 얼굴 이미지 (target_enroll)
    adv_img_path    : 적대적 공격 이미지 (adversarial)
    clean_img_path  : 원본 clean 이미지 (선택) — 비교용
    """
    print("\n=== Feature Squeezing 파일 테스트 ===")

    enroll_img = Image.open(enroll_img_path).convert("RGB")

    if clean_img_path:
        print("\n[정상 이미지 테스트]")
        clean_img = Image.open(clean_img_path).convert("RGB")
        result_clean = feature_squeezing_check(clean_img, enroll_img, device=device)
        _print_result(result_clean, "정상")

    print("\n[적대적 공격 이미지 테스트]")
    adv_img = Image.open(adv_img_path).convert("RGB")
    result_adv = feature_squeezing_check(adv_img, enroll_img, device=device)
    _print_result(result_adv, "공격")

    if clean_img_path:
        print(f"\n결과 요약:")
        print(f"  정상 이미지 → {'ACCEPT ✓' if result_clean['accepted'] else 'REJECT ✗ (오탐지)'}")
        print(f"  공격 이미지 → {'REJECT ✓ (탐지 성공)' if not result_adv['accepted'] else 'ACCEPT ✗ (탐지 실패)'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="카메라 기반 Feature Squeezing 방어 + 공격 테스트")
    parser.add_argument("--enroll-img",  default=None, help="등록 이미지 경로 (없으면 E/ㄷ 키로 캡처)")
    parser.add_argument("--adv-img",     default=None, help="적대적 공격 이미지 경로 (파일 테스트 모드)")
    parser.add_argument("--clean-img",   default=None, help="정상 이미지 경로 (비교용, 선택)")
    args = parser.parse_args()

    if args.adv_img:
        if not args.enroll_img:
            print("[오류] --adv-img 사용 시 --enroll-img 필수")
        else:
            run_file_test(
                enroll_img_path=args.enroll_img,
                adv_img_path=args.adv_img,
                clean_img_path=args.clean_img,
            )
    else:
        run_camera(enroll_img_path=args.enroll_img)
