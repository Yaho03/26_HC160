"""
Squeeze probe 계측 세션 (EXP-DET-001)

웹캠에서 프레임을 읽어 clean 표본을 기록하고, N프레임마다 같은 프레임에 PGD를
적용해 adversarial 표본을 함께 기록한다. 같은 프레임에서 나온 쌍이므로 조명과
자세가 교란요인으로 작용하지 않는다.

임계값 판정은 하지 않는다. 이 도구는 임계값을 산출하기 위한 데이터를 모은다.

실행:
    python -m src.verification.defenses.probe_capture --subject p01 --frames 300
    python -m src.verification.defenses.probe_capture --subject p01 --attack-every 0

설계 근거는 docs/experiments/EXP-DET-001-camera-squeeze-probe.md 참조.
"""

from __future__ import annotations

import argparse
import hashlib
import secrets
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from src.verification.defenses.facenet_embed import (
    FaceNetBatchEmbedder,
    get_embedding,
)
from src.verification.defenses.probe_attacks import (
    ATTACK_KINDS,
    AttackConfig,
    attack_for_index,
    build_attack_params,
    run_attack,
)
from src.verification.defenses.probe_log import ProbeWriter, write_session_sidecar
from src.verification.defenses.squeeze_probe import (
    TRANSFORM_ORDER,
    TRANSFORM_PARAMS,
    jpeg_headroom,
    probe_crop,
)
from src.verification.defenses.verification_defense_temporal_camera import (
    detect_and_crop,
    generate_adversarial,
)

WEIGHTS_FILENAME = "20180402-114759-vggface2.pt"


def sample_id(frame_idx: int, label: str) -> str:
    """표본 ID. 같은 프레임의 clean과 adversarial을 접미사로 구분한다."""
    suffix = "clean" if label == "clean" else "adv"
    return f"f{frame_idx:06d}_{suffix}"


def decode_fourcc(value: float | int) -> str | None:
    """
    CAP_PROP_FOURCC 정수를 코덱 문자열로 바꾼다.

    JPEG 계열 변환의 의미가 이 값에 따라 달라진다. 카메라가 MJPG를 주면 프레임이
    이미 JPEG 압축된 상태이므로 JPEG 재압축이 거의 무손실이 되고, 그만큼 탐지력이
    떨어진다. raw 형식이면 센서 노이즈가 남아 있어 같은 변환이 실제로 작동한다.
    분석 단계에서 이 구분 없이 JPEG 결과를 해석하면 안 된다.
    """
    code = int(value)
    if code <= 0:
        return None
    chars = [chr((code >> (8 * shift)) & 0xFF) for shift in range(4)]
    text = "".join(chars).strip()
    return text if text.isprintable() and text else None


def should_attack(frame_idx: int, attack_every: int) -> bool:
    """0이면 공격 생성을 끈다."""
    return attack_every > 0 and frame_idx % attack_every == 0


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def _weights_sha256() -> str | None:
    """가중치 파일 해시. 경로는 개인 홈을 포함하므로 사이드카에 넣지 않는다."""
    for root in (Path.home() / ".cache/torch/hub/checkpoints", Path.home() / ".cache/torch/checkpoints"):
        candidate = root / WEIGHTS_FILENAME
        if candidate.exists():
            digest = hashlib.sha256()
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            return digest.hexdigest()
    return None


def session_metadata(
    *,
    session_id: str,
    subject_id: str,
    camera: dict,
    attack: dict,
    counters: dict,
    elapsed_sec: float,
    jpeg_headroom_q75: float | None = None,
    target_frames: int | None = None,
    completed: bool = True,
    interrupted_by: str | None = None,
) -> dict:
    """임계값 artifact가 참조할 provenance를 모은다."""
    return {
        "session_id": session_id,
        "subject_id": subject_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "camera": camera,
        "model": {
            "name": "InceptionResnetV1",
            "pretrained": "vggface2",
            "weights_file": WEIGHTS_FILENAME,
            "weights_sha256": _weights_sha256(),
            "preprocess": "resize160_bilinear,(x-127.5)/128.0",
        },
        "transforms": {name: TRANSFORM_PARAMS[name] for name in TRANSFORM_ORDER},
        "jpeg_headroom_q75": (
            round(jpeg_headroom_q75, 4) if jpeg_headroom_q75 is not None else None
        ),
        "attack": attack,
        "counters": counters,
        "target_frames": target_frames,
        "completed": completed,
        "interrupted_by": interrupted_by,
        "elapsed_sec": round(elapsed_sec, 3),
        "effective_fps": (
            round(counters["frames_read"] / elapsed_sec, 2) if elapsed_sec > 0 else None
        ),
        "notes": (
            "dropped_frames는 read() 실패 횟수이며 드라이버 수준 드롭 카운트가 아니다. "
            "실제 처리 속도는 effective_fps로 판단한다."
        ),
    }


def _establish_enrollment(cap, enroll_img_path, device, preview=None):
    """등록 얼굴을 확보한다. 경로가 없으면 카메라에서 첫 얼굴을 잡는다."""
    if enroll_img_path:
        from PIL import Image

        import numpy as np

        raw = Image.open(enroll_img_path).convert("RGB")
        frame = cv2.cvtColor(np.array(raw), cv2.COLOR_RGB2BGR)
        crop, _ = detect_and_crop(frame, device)
        if crop is None:
            raise RuntimeError(f"등록 이미지에서 얼굴을 찾지 못했다: {enroll_img_path}")
        return crop

    print("[등록] 카메라를 보세요. 얼굴이 잡히면 자동으로 등록합니다.")
    for _ in range(300):
        ok, frame = cap.read()
        if not ok:
            continue
        if preview is not None:
            preview.show(frame, None, {"samples_clean": 0, "samples_adversarial": 0}, 0, "등록 대기")
            cv2.waitKey(1)
        crop, _ = detect_and_crop(frame, device)
        if crop is not None:
            print("[등록] 완료")
            return crop
    raise RuntimeError("등록 얼굴을 확보하지 못했다. 조명과 카메라 권한을 확인하라.")


def capture_session(
    *,
    subject_id: str,
    frames: int,
    attack_every: int,
    out_dir: Path,
    camera_index: int,
    enroll_img_path: str | None,
    epsilon: float,
    steps: int,
    step_size: float,
    attack_kinds: list[str] | None = None,
    no_preview: bool = False,
    width: int | None = None,
    height: int | None = None,
    device=None,
) -> tuple[Path, Path]:
    # 촬영 시작 전에 공격 파라미터를 검증한다. 촬영을 다 하고 실패하면 사람 시간을 버린다.
    attack_kinds = list(attack_kinds or ["pgd"])
    attack_config = AttackConfig(epsilon=epsilon, steps=steps, step_size=step_size)
    attack_params = build_attack_params(attack_kinds, attack_config)

    session_id = secrets.token_hex(6)
    out_dir = Path(out_dir) / session_id
    csv_path = out_dir / "probe.csv"
    sidecar_path = out_dir / "session.json"

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            "카메라를 열 수 없다. macOS에서는 터미널에 카메라 권한을 허용해야 한다."
        )

    # MTCNN 비용이 해상도에 비례한다. 1080p에서는 프레임당 1초를 넘어 세션이 길어진다.
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # 카메라 속성은 release() 전에 읽는다. 릴리즈된 capture는 0을 반환한다.
    camera_info = {
        "index": camera_index,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
        "fps_nominal": cap.get(cv2.CAP_PROP_FPS) or None,
        "fourcc": decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
    }

    embedder = FaceNetBatchEmbedder(device)
    counters = {
        "frames_read": 0,
        "read_failures": 0,
        "frames_without_face": 0,
        "samples_clean": 0,
        "samples_adversarial": 0,
        "rows": 0,
    }

    # 사이드카에 필요한 상태는 try 밖에서 초기화한다. 중단되더라도 provenance 없이
    # CSV만 남는 상황을 막아야 한다.
    headroom: list[float] = []
    attack_index = 0
    attack_counts: dict[str, int] = {kind: 0 for kind in attack_kinds}
    preview = Preview(enabled=not no_preview)
    started = time.perf_counter()
    elapsed = 0.0
    completed = False
    interrupted_by = None

    try:
        enroll_preview = Preview(enabled=not no_preview)
        enroll_crop = _establish_enrollment(cap, enroll_img_path, device, enroll_preview)
        enroll_torch = get_embedding(enroll_crop, device)
        enroll_vector = enroll_torch.numpy().astype("float64")

        print(f"[기록] session={session_id}  목표 {frames}프레임  공격 주기 {attack_every}")
        started = time.perf_counter()
        with ProbeWriter(csv_path, session_id=session_id, subject_id=subject_id) as writer:
            frame_idx = 0
            while counters["samples_clean"] < frames:
                ok, frame = cap.read()
                if not ok:
                    counters["read_failures"] += 1
                    if counters["read_failures"] > 60:
                        print("[중단] 카메라 read 실패가 반복된다.")
                        interrupted_by = "read_failure"
                        break
                    continue

                counters["frames_read"] += 1
                frame_ts_ms = (time.perf_counter() - started) * 1000.0

                crop, box = detect_and_crop(frame, device)
                if crop is None:
                    counters["frames_without_face"] += 1
                    preview.show(frame, None, counters, frames, "얼굴 없음")
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                    continue

                headroom.append(jpeg_headroom(crop))
                reading = probe_crop(crop, enroll_vector, embedder)
                counters["rows"] += writer.write_sample(
                    sample_id=sample_id(frame_idx, "clean"),
                    frame_idx=frame_idx,
                    frame_ts_ms=frame_ts_ms,
                    dropped_frames=counters["read_failures"],
                    label="clean",
                    reading=reading,
                )
                counters["samples_clean"] += 1

                if should_attack(frame_idx, attack_every):
                    kind = attack_for_index(attack_index, attack_kinds)
                    adv_crop, _ = run_attack(
                        kind, crop, enroll_torch, attack_config, device=device
                    )
                    adv_reading = probe_crop(adv_crop, enroll_vector, embedder)
                    counters["rows"] += writer.write_sample(
                        sample_id=sample_id(frame_idx, "adversarial"),
                        frame_idx=frame_idx,
                        frame_ts_ms=frame_ts_ms,
                        dropped_frames=counters["read_failures"],
                        label="adversarial",
                        reading=adv_reading,
                        attack_kind=kind,
                    )
                    counters["samples_adversarial"] += 1
                    attack_counts[kind] += 1
                    attack_index += 1

                preview.show(frame, box, counters, frames, "")
                frame_idx += 1
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    print("[중단] 사용자 취소")
                    interrupted_by = "user_cancel"
                    break

        completed = counters["samples_clean"] >= frames
    except KeyboardInterrupt:
        interrupted_by = "keyboard_interrupt"
        print("\n[중단] 사용자 인터럽트")
    finally:
        elapsed = time.perf_counter() - started
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

        # 중단된 세션도 사이드카를 남긴다. provenance 없는 CSV는 분석에 쓸 수 없다.
        write_session_sidecar(
            sidecar_path,
            session_metadata(
                session_id=session_id,
                subject_id=subject_id,
                camera=camera_info,
                attack={
                    "kinds": attack_kinds,
                    "params": attack_params,
                    "counts": attack_counts,
                    "every": attack_every,
                    "target": "enroll_template",
                },
                counters=counters,
                elapsed_sec=elapsed,
                jpeg_headroom_q75=(
                    sum(headroom) / len(headroom) if headroom else None
                ),
                target_frames=frames,
                completed=completed,
                interrupted_by=interrupted_by,
            ),
        )

    print(f"\n[완료] {counters['rows']}행 기록")
    print(f"  clean {counters['samples_clean']} / adversarial {counters['samples_adversarial']}")
    print(f"  얼굴 미검출 프레임 {counters['frames_without_face']}, read 실패 {counters['read_failures']}")
    print(f"  CSV      {csv_path}")
    print(f"  사이드카 {sidecar_path}")
    return csv_path, sidecar_path


class Preview:
    """
    진행 상황 창. 피험자가 프레임 안에 머무르려면 필요하다.

    창을 열 수 없는 환경에서는 한 번만 알리고 이후 조용히 넘어간다. 계측은 창
    없이도 완전하므로 preview 실패가 세션을 중단시키면 안 된다.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._failed = False

    def show(self, frame, box, counters, target, status) -> None:
        if not self.enabled or self._failed:
            return
        try:
            _draw(frame, box, counters, target, status)
        except cv2.error:
            self._failed = True
            print("[알림] preview 창을 열 수 없어 창 없이 계속한다.")

    def close(self) -> None:
        if not self.enabled or self._failed:
            return
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def _draw(frame, box, counters, target, status) -> None:
    if box is not None:
        x1, y1, x2, y2 = [int(value) for value in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 180), 2)
    text = f"clean {counters['samples_clean']}/{target}  adv {counters['samples_adversarial']}  {status}"
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("EXP-DET-001 probe", frame)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EXP-DET-001 squeeze probe 계측 세션")
    parser.add_argument("--subject", required=True, help="피험자 불투명 ID (예: p01)")
    parser.add_argument("--frames", type=int, default=300, help="기록할 clean 표본 수")
    parser.add_argument("--attack-every", type=int, default=10, help="공격 생성 주기. 0이면 끔")
    parser.add_argument("--out-dir", default="outputs/probe", help="세션 출력 디렉터리")
    parser.add_argument("--camera", type=int, default=0, help="카메라 인덱스")
    parser.add_argument("--enroll-img", default=None, help="등록 이미지. 없으면 카메라에서 캡처")
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--step-size", type=float, default=0.002)
    parser.add_argument(
        "--attack-kinds",
        default="pgd",
        help=f"쉼표로 구분한 공격 종류. 기회마다 번갈아 쓴다. 사용 가능: {','.join(ATTACK_KINDS)}",
    )
    parser.add_argument("--width", type=int, default=1280, help="캡처 폭. MTCNN 속도에 직결")
    parser.add_argument("--height", type=int, default=720, help="캡처 높이")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="진행 창을 열지 않는다. GUI가 없는 환경에서 사용한다",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    capture_session(
        subject_id=args.subject,
        frames=args.frames,
        attack_every=args.attack_every,
        out_dir=Path(args.out_dir),
        camera_index=args.camera,
        enroll_img_path=args.enroll_img,
        epsilon=args.epsilon,
        steps=args.steps,
        step_size=args.step_size,
        attack_kinds=[k.strip() for k in args.attack_kinds.split(",") if k.strip()],
        no_preview=args.no_preview,
        width=args.width,
        height=args.height,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
