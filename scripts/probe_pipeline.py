"""
계측 세션 도착 시 실행하는 분석 파이프라인

지금까지 손으로 치던 다음 단계를 한 명령으로 묶는다.

1. zip 압축 해제. 세션 ID 기준 경로로만 넣는다
2. 사이드카 검증
3. 기존 세션과 합치기
4. threshold artifact 재생성
5. 피험자 분리 검증 (population, per_user)
6. 표 출력과 JSON 저장

명령은 docs/14_LOCAL_RUNBOOK.md 10절과
docs/experiments/EXP-DET-001-camera-squeeze-probe.md 7.1~7.3절의 것을 그대로 쓴다.
로직은 기존 모듈을 import하며 복사하지 않는다.

zip 파일명에 실명이 들어 있을 수 있으므로 경로와 출력에 원본 파일명을 넣지 않는다.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from collections import namedtuple
from pathlib import Path

# 스크립트로 직접 실행할 때도 저장소 루트를 찾도록 한다.
# python scripts/probe_pipeline.py 와 python -m scripts.probe_pipeline 둘 다 지원한다.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.verification.defenses.probe_log import _require_opaque

EXIT_OK = 0
EXIT_ERROR = 1        # 실행 오류
EXIT_VALIDATION = 2   # 검증 실패. CI에서 원인을 가릴 수 있게 구분한다

# 노트북 자가 검사와 같은 기준이다.
MIN_HEADROOM = 1.0
MIN_FACE_RATE = 0.50
MIN_CLEAN = 100

REQUIRED_MEMBERS = {"probe.csv", "session.json"}

Finding = namedtuple("Finding", "check message")


class ZipContentError(ValueError):
    """압축 파일에 예상하지 않은 항목이 있다."""


# ── 순수 함수 ─────────────────────────────────────────────────────────────────


def check_sidecar(meta: dict) -> list[Finding]:
    """
    사이드카가 분석에 쓸 만한지 본다. 파일을 읽지 않으므로 테스트 가능하다.

    기준 미달을 오류로 보지 않고 findings로 돌려준다. 판단은 호출자가 한다.
    """
    findings = []
    counters = meta.get("counters", {})

    if not meta.get("completed"):
        findings.append(
            Finding("completed", "세션이 목표 프레임 전에 중단됐다")
        )

    headroom = meta.get("jpeg_headroom_q75")
    if headroom is None or headroom < MIN_HEADROOM:
        findings.append(
            Finding(
                "jpeg_headroom_q75",
                f"입력이 이미 압축된 것 같다 (headroom {headroom}, 기준 {MIN_HEADROOM} 이상)",
            )
        )

    read = counters.get("frames_read", 0)
    without_face = counters.get("frames_without_face", 0)
    face_rate = (read - without_face) / read if read else 0.0
    if face_rate < MIN_FACE_RATE:
        findings.append(
            Finding(
                "face_rate",
                f"얼굴이 잡힌 프레임이 {face_rate:.0%}다 (기준 {MIN_FACE_RATE:.0%} 이상)",
            )
        )

    clean = counters.get("samples_clean", 0)
    if clean < MIN_CLEAN:
        findings.append(
            Finding(
                "samples_clean",
                f"clean 표본이 {clean}개다 (목표 FPR 1%에는 {MIN_CLEAN}개 이상 필요)",
            )
        )
    return findings


def session_dir_name(meta: dict) -> str:
    """
    세션 디렉터리 이름. zip 파일명이 아니라 사이드카의 세션 ID로만 만든다.

    subject_id도 함께 검증한다. 실명이 들어오면 여기서 막힌다.
    """
    _require_opaque("subject_id", meta.get("subject_id", ""))
    return _require_opaque("session_id", meta.get("session_id", ""))


def stage_ok(name: str, payload: dict | None = None) -> dict:
    """실행했고 결과가 있다. 결과가 비어 있어도 skipped와 구분된다."""
    return {"stage": name, "status": "ok", "reason": None, **(payload or {})}


def stage_skipped(name: str, reason: str) -> dict:
    """실행하지 않았다. 사유를 반드시 남긴다."""
    return {"stage": name, "status": "skipped", "reason": reason}


def format_report(report: dict) -> str:
    """
    건너뛴 단계를 최상단에 먼저 보인다. 하위 항목에 묻히면 완주한 것처럼 읽힌다.
    """
    stages = report.get("stages", [])
    lines = []

    skipped = [s for s in stages if s["status"] == "skipped"]
    if skipped:
        lines.append("건너뛴 단계")
        for stage in skipped:
            lines.append(f"  {stage['stage']}: SKIPPED ({stage['reason']})")
        lines.append("")

    lines.append("단계별 결과")
    for stage in stages:
        if stage["status"] == "skipped":
            continue
        detail = {k: v for k, v in stage.items() if k not in ("stage", "status", "reason")}
        lines.append(f"  {stage['stage']}: {stage['status']}  {_brief(detail)}")
    return "\n".join(lines)


def _brief(detail: dict) -> str:
    parts = []
    for key, value in detail.items():
        if isinstance(value, (list, tuple)):
            parts.append(f"{key}={len(value)}건")
        elif isinstance(value, dict):
            parts.append(f"{key}={len(value)}항목")
        else:
            parts.append(f"{key}={value}")
    return "  ".join(parts)


# ── 파일 처리 ─────────────────────────────────────────────────────────────────


def extract_session(archive_path, destination_root) -> Path:
    """
    세션 zip을 세션 ID 디렉터리로 푼다.

    예상 항목만 허용한다. 경로 탈출과 얼굴 이미지 같은 예상 밖 파일을 막는다.
    """
    archive_path = Path(archive_path)
    destination_root = Path(destination_root)

    with zipfile.ZipFile(archive_path) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        flattened = {Path(n).name for n in names}

        for name in names:
            if Path(name).name != name.lstrip("./"):
                # 디렉터리를 하나 감싼 형태는 허용하되 상위 이동은 막는다
                if ".." in Path(name).parts or Path(name).is_absolute():
                    raise ZipContentError(f"경로 탈출 항목: {name}")

        unexpected = flattened - REQUIRED_MEMBERS
        if unexpected:
            raise ZipContentError(
                f"예상하지 않은 항목 {sorted(unexpected)}. "
                f"허용: {sorted(REQUIRED_MEMBERS)}"
            )
        missing = REQUIRED_MEMBERS - flattened
        if missing:
            raise ZipContentError(f"필수 항목 누락: {sorted(missing)}")

        with tempfile.TemporaryDirectory() as staging:
            staging = Path(staging)
            for name in names:
                target = staging / Path(name).name
                with archive.open(name) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)

            meta = json.loads((staging / "session.json").read_text(encoding="utf-8"))
            destination = destination_root / session_dir_name(meta)
            destination.mkdir(parents=True, exist_ok=True)
            for member in REQUIRED_MEMBERS:
                shutil.copy2(staging / member, destination / member)

    return destination


# ── 파이프라인 ────────────────────────────────────────────────────────────────

FEATURES = [
    ("jpeg_q75", "self_consistency"), ("jpeg_q50", "self_consistency"),
    ("blur2.0", "template_shift"), ("jpeg_q30", "self_consistency"),
    ("median3", "self_consistency"), ("median5", "self_consistency"),
]

# 공격 패키지 attack_handoff_index.csv에 기록된 FaceNet 신원 임계값
IDENTITY_THRESHOLD = 0.47966246581077576


def _load_meta(session_dir: Path) -> dict:
    return json.loads((session_dir / "session.json").read_text(encoding="utf-8"))


def run_pipeline(
    *,
    archives,
    session_root: Path,
    existing_sessions,
    output_dir: Path,
    window_frames: int = 3,
    target_fpr: float = 0.01,
    adaptive_detection_rate: float = 0.0,
    assume_yes: bool = False,
    interactive: bool = True,
) -> tuple[dict, int]:
    """
    단계별 결과와 종료 코드를 돌려준다.

    검증 실패는 EXIT_VALIDATION, 실행 오류는 EXIT_ERROR로 구분한다. CI에서 원인을
    가리려면 두 코드가 달라야 한다.
    """
    from src.verification.defenses.cross_subject import (
        InsufficientSubjectsError,
        leave_one_subject_out,
    )
    from src.verification.defenses.probe_analyze import load_probe_rows
    from src.verification.defenses.probe_merge import merge_probe_csvs
    from src.verification.defenses.probe_threshold import build_artifact

    stages = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 압축 해제
    new_dirs = []
    for archive in archives:
        new_dirs.append(extract_session(archive, session_root))
    stages.append(stage_ok("extract", {"sessions": [d.name for d in new_dirs]}))

    # 2) 검증
    all_findings = {}
    for directory in new_dirs:
        findings = check_sidecar(_load_meta(directory))
        if findings:
            all_findings[directory.name] = [f._asdict() for f in findings]
    stages.append(stage_ok("validate", {"sessions_with_findings": all_findings}))

    if all_findings and not assume_yes:
        for session, findings in all_findings.items():
            print(f"\n[검증 경고] 세션 {session}")
            for item in findings:
                print(f"  - {item['message']}")
        if not interactive:
            print("\n비대화형 실행이므로 중단한다. 계속하려면 --yes를 준다.")
            return {"stages": stages}, EXIT_VALIDATION
        answer = input("\n계속 진행하겠는가? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            return {"stages": stages}, EXIT_VALIDATION

    # 3) 합치기
    sources = [Path(p) for p in existing_sessions] + [d / "probe.csv" for d in new_dirs]
    merged = merge_probe_csvs(sources, output_dir / "probe.csv")
    rows = load_probe_rows(merged)
    subjects = sorted({r["subject_id"] for r in rows})
    stages.append(
        stage_ok("merge", {"rows": len(rows), "subjects": subjects,
                           "sessions": len({r["session_id"] for r in rows})})
    )

    # 4) artifact
    sidecars = [_load_meta(d) for d in new_dirs]
    for path in existing_sessions:
        sidecar = Path(path).parent / "session.json"
        if sidecar.exists():
            sidecars.append(json.loads(sidecar.read_text(encoding="utf-8")))
    artifact = build_artifact(
        merged, sidecars, target_fpr=target_fpr, window_frames=window_frames,
        identity_threshold=IDENTITY_THRESHOLD,
        adaptive_detection_rate=adaptive_detection_rate,
    )
    (output_dir / "detector_threshold.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stages.append(
        stage_ok("artifact", {"threshold": artifact["threshold"],
                              "limitations": artifact["limitations"]})
    )

    # 5) 피험자 분리 검증
    for mode in ("population", "per_user"):
        try:
            results = leave_one_subject_out(
                rows, features=FEATURES, target_fpr=target_fpr,
                window_frames=window_frames, normalization=mode,
            )
        except InsufficientSubjectsError as error:
            stages.append(stage_skipped(f"cross_subject_{mode}", str(error)))
        except Exception as error:  # noqa: BLE001 - 단계 실패가 전체를 죽이지 않게 한다
            stages.append(
                {"stage": f"cross_subject_{mode}", "status": "error",
                 "reason": f"{type(error).__name__}: {error}"}
            )
        else:
            stages.append(stage_ok(f"cross_subject_{mode}", {"results": results}))

    return {"stages": stages}, EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="계측 세션 zip을 받아 합치고 artifact와 분리 검증까지 실행한다"
    )
    parser.add_argument("archives", nargs="+", help="세션 zip 경로")
    parser.add_argument("--session-root", default="outputs/probe_remote")
    parser.add_argument(
        "--existing", action="append", default=[],
        help="합칠 기존 probe.csv. 반복 지정한다",
    )
    parser.add_argument("--output-dir", default="outputs/probe_combined")
    parser.add_argument("--window-frames", type=int, default=3)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--adaptive-detection-rate", type=float, default=0.0)
    parser.add_argument(
        "--yes", action="store_true", help="검증 경고가 있어도 묻지 않고 계속한다"
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report, code = run_pipeline(
            archives=[Path(a) for a in args.archives],
            session_root=Path(args.session_root),
            existing_sessions=args.existing,
            output_dir=Path(args.output_dir),
            window_frames=args.window_frames,
            target_fpr=args.target_fpr,
            adaptive_detection_rate=args.adaptive_detection_rate,
            assume_yes=args.yes,
            interactive=sys.stdin.isatty(),
        )
    except (ZipContentError, ValueError) as error:
        print(f"[오류] {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_ERROR

    print()
    print(format_report(report))

    destination = Path(args.output_dir) / "pipeline_report.json"
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\n리포트 저장: {destination}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
