"""
Temporal Consistency 방어 실험용 테스트 영상 생성기

두 가지 영상을 생성한다:
  1. static_attack.mp4  — 동일 프레임 반복 (화면/사진 재생 공격 시뮬레이션) → 탐지 대상
  2. dynamic_normal.mp4 — 미세 움직임 포함 (정상 얼굴 시뮬레이션) → 통과 대상

실행:
    python -m src.verification.defenses.make_test_videos
    python -m src.verification.defenses.make_test_videos --source-img path/to/face.jpg --out-dir outputs/test_videos
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageEnhance

DEFAULT_SOURCE_IMG = (
    "pkg_root/facenet_verification_attack_package/samples/0.005/"
    "vf_256d41087135/source/George_W_Bush_0414.jpg"
)
DEFAULT_OUT_DIR = "outputs/test_videos"
FPS       = 15
DURATION  = 3      # seconds
N_FRAMES  = FPS * DURATION   # 45 frames total


def _pil_to_bgr(img: Image.Image, size=(640, 480)) -> np.ndarray:
    img = img.resize(size, Image.LANCZOS)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _augment(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """정상 얼굴의 미세 움직임을 시뮬레이션하는 augmentation."""
    # 미세 회전 (±8°)
    angle = rng.uniform(-8, 8)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
    # 밝기 변화 (±20%)
    img = ImageEnhance.Brightness(img).enhance(1.0 + rng.uniform(-0.20, 0.20))
    # 대비 변화 (±15%)
    img = ImageEnhance.Contrast(img).enhance(1.0 + rng.uniform(-0.15, 0.15))
    # 미세 이동 (±10px)
    dx, dy = int(rng.uniform(-10, 10)), int(rng.uniform(-10, 10))
    arr = np.array(img)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    arr = cv2.warpAffine(arr, M, (arr.shape[1], arr.shape[0]),
                         borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(arr)


def make_video(out_path: str, frames_bgr: list[np.ndarray], fps: int = FPS) -> None:
    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for f in frames_bgr:
        writer.write(f)
    writer.release()


def add_label(frame: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    frame = frame.copy()
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return frame


def generate_static_attack(source_img: Image.Image, n: int = N_FRAMES) -> list[np.ndarray]:
    """동일 프레임을 n번 반복 — 화면/사진 재생 공격 시뮬레이션."""
    base = _pil_to_bgr(source_img)
    return [add_label(base, "[ATTACK] Static Image Replay", (0, 0, 220))] * n


def generate_dynamic_normal(source_img: Image.Image, n: int = N_FRAMES) -> list[np.ndarray]:
    """미세 변동을 추가한 n개 프레임 — 정상 얼굴 시뮬레이션."""
    rng = np.random.default_rng(seed=0)
    frames = []
    for _ in range(n):
        aug = _augment(source_img.copy(), rng)
        f = _pil_to_bgr(aug)
        frames.append(add_label(f, "[NORMAL] Live Face", (0, 200, 0)))
    return frames


def main(source_img_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print(f"[로드] {source_img_path}")
    source_img = Image.open(source_img_path).convert("RGB")

    # 1. 정적 공격 영상
    print(f"[생성] static_attack.mp4 ({N_FRAMES}프레임, {FPS}fps, {DURATION}초)")
    static_frames = generate_static_attack(source_img)
    static_path = os.path.join(out_dir, "static_attack.mp4")
    make_video(static_path, static_frames)
    print(f"       저장 → {static_path}")

    # 2. 정상 얼굴 영상
    print(f"[생성] dynamic_normal.mp4 ({N_FRAMES}프레임, {FPS}fps, {DURATION}초)")
    dynamic_frames = generate_dynamic_normal(source_img)
    dynamic_path = os.path.join(out_dir, "dynamic_normal.mp4")
    make_video(dynamic_path, dynamic_frames)
    print(f"       저장 → {dynamic_path}")

    print("\n완료.")
    print(f"  정적 공격 영상 : {static_path}")
    print(f"  정상 얼굴 영상 : {dynamic_path}")
    print("\n실행 예:")
    print(f"  python -m src.verification.defenses.verification_defense_temporal_camera --video {static_path} --enroll-img {source_img_path}")
    print(f"  python -m src.verification.defenses.verification_defense_temporal_camera --video {dynamic_path} --enroll-img {source_img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Consistency 방어 실험용 테스트 영상 생성")
    parser.add_argument("--source-img", default=DEFAULT_SOURCE_IMG, help="원본 얼굴 이미지 경로")
    parser.add_argument("--out-dir",    default=DEFAULT_OUT_DIR,    help="출력 디렉터리")
    args = parser.parse_args()

    main(args.source_img, args.out_dir)
