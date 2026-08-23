from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from src.face_auth.adapters.capture_base import FrameSource
from src.face_auth.adapters.in_memory_store import InMemoryStore
from src.face_auth.adapters.opencv_capture import (
    CaptureSourceError,
    video_source,
    webcam_source,
)
from src.face_auth.adapters.opencv_preview import (
    OpenCVPreview,
    PreviewUnavailableError,
)
from src.face_auth.application.enrollment_service import (
    create_template,
    load_template,
    save_template,
)
from src.face_auth.application.evidence_service import build_capture_manifest
from src.face_auth.application.evaluation_service import EvaluationService
from src.face_auth.application.session_service import SessionService
from src.face_auth.application.token_service import TokenService
from src.face_auth.domain.policy import PolicyEngine
from src.face_auth.domain.types import (
    FramePacket,
    GateStatus,
    SecurityProfile,
    SessionState,
)
from src.face_auth.inference.face_detector import MTCNNFaceDetector
from src.face_auth.inference.active_liveness import ActiveLivenessGate
from src.face_auth.inference.adversarial_detector import (
    AdversarialDetectorConfig,
    TransformConsistencyDetector,
)
from src.face_auth.inference.camera_motion import CameraMotionConfig, CameraMotionGate
from src.face_auth.inference.continuity import ContinuityConfig, IdentityContinuityGate
from src.face_auth.inference.feature_squeeze import FeatureSqueezeInspector
from src.face_auth.inference.full_pipeline import FullEvidencePipeline
from src.face_auth.inference.head_pose import FivePointHeadPoseEstimator
from src.face_auth.inference.pad_adapter import create_pad_scorer
from src.face_auth.inference.passive_pad import PassivePADConfig, PassivePADGate
from src.face_auth.inference.pipeline import BaselineEvidencePipeline
from src.face_auth.inference.quality import QualityConfig, QualityGate
from src.face_auth.inference.verifier import (
    FaceNetEmbedder,
    MultiFrameVerifier,
    VerificationConfig,
)


def _source(args):
    return video_source(args.video) if args.video else webcam_source(args.camera)


@dataclass(frozen=True)
class CaptureBatch:
    frames: tuple[FramePacket, ...]
    cancelled: bool = False


def _collect(
    source: FrameSource,
    frame_count: int,
    *,
    preview: OpenCVPreview | None = None,
    purpose: str = "CAPTURE",
) -> CaptureBatch:
    frames = []
    try:
        while len(frames) < frame_count:
            frame = source.read()
            if frame is None:
                break
            frames.append(frame)
            if preview is not None and not preview.show(
                frame,
                captured_frames=len(frames),
                target_frames=frame_count,
                purpose=purpose,
            ):
                return CaptureBatch(tuple(frames), cancelled=True)
    finally:
        source.close()
        if preview is not None:
            preview.close()
    return CaptureBatch(tuple(frames))


def _capture(args, purpose: str) -> CaptureBatch:
    preview = OpenCVPreview() if _preview_enabled(args) else None
    return _collect(_source(args), args.frames, preview=preview, purpose=purpose)


def _preview_enabled(args) -> bool:
    if args.preview is not None:
        return args.preview
    return args.camera is not None


def _capture_failure(
    error: RuntimeError,
    *,
    session_id: str | None = None,
) -> int:
    if isinstance(error, CaptureSourceError):
        reason_code = error.reason_code
        hint = (
            "Allow camera access for the Python or terminal process and check "
            "the camera index."
            if reason_code == "CAMERA_UNAVAILABLE"
            else "Check that the video path exists and uses a supported codec."
        )
    else:
        reason_code = "PREVIEW_UNAVAILABLE"
        hint = "Run in a desktop session or pass --no-preview for headless capture."
    payload = {
        "status": "CAPTURE_ERROR",
        "reason_code": reason_code,
        "message": str(error),
        "hint": hint,
    }
    if session_id is not None:
        payload["session_id"] = session_id
        payload["state"] = SessionState.ERROR.value
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 3


def _capture_cancelled(*, captured_frames: int, session_id: str | None = None) -> int:
    payload = {
        "status": "CAPTURE_CANCELLED",
        "reason_code": "USER_CANCELLED",
        "captured_frames": captured_frames,
    }
    if session_id is not None:
        payload["session_id"] = session_id
        payload["state"] = SessionState.RETRYABLE.value
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 130


def enroll(args) -> int:
    detector = MTCNNFaceDetector(device=args.device)
    embedder = FaceNetEmbedder(device=args.device)
    quality = _quality_gate(args)
    try:
        capture = _capture(args, "ENROLLMENT")
    except (CaptureSourceError, PreviewUnavailableError) as error:
        return _capture_failure(error)
    if capture.cancelled:
        return _capture_cancelled(captured_frames=len(capture.frames))
    frames = capture.frames
    crops = []
    for frame in frames:
        faces = detector.detect(frame.image_bgr)
        if len(faces) != 1:
            continue
        if (
            quality.evaluate(frame.image_bgr, faces[0].bbox).result.status
            is not GateStatus.PASS
        ):
            continue
        crops.append(faces[0].crop)

    try:
        template = create_template(crops, embedder, min_frames=args.min_valid_frames)
    except ValueError as error:
        print(
            json.dumps(
                {
                    "status": "ENROLLMENT_REJECTED",
                    "reason": str(error),
                    "captured_frames": len(frames),
                    "valid_frames": len(crops),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    save_template(template, args.output)
    print(
        json.dumps(
            {
                "status": "ENROLLED",
                "template_path": str(Path(args.output).resolve()),
                "template_version": template.template_version,
                "model_version": template.model_version,
                "valid_frames": len(crops),
                "warning": "Prototype template storage is not encrypted.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def authenticate(args) -> int:
    profile = SecurityProfile(args.profile)
    if profile is SecurityProfile.FULL and not args.pad_model:
        raise SystemExit("FULL profile requires --pad-model")
    template = load_template(args.template)
    detector = MTCNNFaceDetector(device=args.device)
    embedder = FaceNetEmbedder(device=args.device)
    verifier = MultiFrameVerifier(
        template.embedding,
        VerificationConfig(
            threshold=args.threshold,
            threshold_version=args.threshold_version,
            model_version=template.model_version,
            min_probe_frames=args.min_valid_frames,
        ),
    )
    pipeline = BaselineEvidencePipeline(
        detector,
        embedder,
        verifier,
        quality=_quality_gate(args),
        min_valid_frames=args.min_valid_frames,
    )

    store = InMemoryStore()
    challenge_kinds = (
        ("HEAD_LEFT", "HEAD_RIGHT")
        if profile is SecurityProfile.FULL
        else ("HEAD_LEFT", "HEAD_RIGHT", "BLINK")
    )
    sessions = SessionService(
        store,
        policy_version=args.policy_version,
        challenge_kinds=challenge_kinds,
    )
    tokens = TokenService(store)
    evaluation = EvaluationService(
        store, sessions, PolicyEngine(args.policy_version), tokens
    )
    session = sessions.create_session(
        user_id=args.user_id,
        purpose=args.purpose,
        transaction_context_hash=args.context_hash,
        security_profile=profile,
    )
    challenge = sessions.issue_challenge(session.session_id)
    sessions.move(session.session_id, SessionState.CAPTURING)
    try:
        capture = _capture(args, "AUTHENTICATION")
    except (CaptureSourceError, PreviewUnavailableError) as error:
        sessions.move(session.session_id, SessionState.ERROR)
        return _capture_failure(error, session_id=session.session_id)
    if capture.cancelled:
        sessions.move(session.session_id, SessionState.RETRYABLE)
        return _capture_cancelled(
            captured_frames=len(capture.frames), session_id=session.session_id
        )
    frames = capture.frames
    if not frames:
        sessions.move(session.session_id, SessionState.ERROR)
        print(
            json.dumps(
                {
                    "session_id": session.session_id,
                    "state": session.state.value,
                    "decision": "ERROR",
                    "reason_codes": ["CAMERA_ERROR"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 3
    manifest = build_capture_manifest(session, frames)
    sessions.move(session.session_id, SessionState.EVIDENCE_RECEIVED)
    sessions.move(session.session_id, SessionState.EVALUATING)
    if profile is SecurityProfile.FULL:
        full_pipeline = _full_pipeline(args, pipeline, embedder, template.embedding)
        full_observation = full_pipeline.evaluate(
            frames,
            challenge_kind=challenge.kind,
            challenge_start_frame_id=-1,
        )
        observation = full_observation.baseline
        gate_results = list(full_observation.gate_results)
    else:
        observation = pipeline.evaluate(frames)
        gate_results = list(observation.gate_results)
    outcome = evaluation.evaluate(session.session_id, gate_results)

    payload = {
        "session_id": session.session_id,
        "security_profile": session.security_profile.value,
        "state": session.state.value,
        "challenge": challenge.kind,
        "decision": outcome.decision.action.value,
        "reason_codes": list(outcome.decision.reason_codes),
        "total_frames": observation.total_frames,
        "valid_face_frames": observation.valid_face_frames,
        "attempt_id": manifest.attempt_id,
        "evidence_digest": manifest.evidence_digest,
        "token_id": outcome.token.token_id if outcome.token else None,
        "warning": (
            "FULL is a reference profile; thresholds must be calibrated on target cameras."
            if profile is SecurityProfile.FULL
            else "BASELINE_ONLY does not include PAD, active liveness, or continuity."
        ),
        "gates": [
            {
                "gate": result.gate,
                "status": result.status.value,
                "score": result.score,
                "threshold": result.threshold,
                "reason_codes": list(result.reason_codes),
                "model_version": result.model_version,
                "threshold_version": result.threshold_version,
            }
            for result in outcome.decision.gate_results
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if outcome.token else 2


def _full_pipeline(args, baseline, embedder, template_embedding):
    pad_scorer = create_pad_scorer(
        runtime=args.pad_runtime,
        model_path=args.pad_model,
        model_version=args.pad_model_version,
        input_size=args.pad_input_size,
        live_class_index=args.pad_live_class_index,
        output_kind=args.pad_output_kind,
        device=args.device or "cpu",
        providers=args.pad_provider,
    )
    adversarial = None
    if args.adversarial_threshold is not None:
        adversarial = FeatureSqueezeInspector(
            embedder,
            TransformConsistencyDetector(
                AdversarialDetectorConfig(
                    max_cosine_distance=args.adversarial_threshold,
                    threshold_version=args.adversarial_threshold_version,
                )
            ),
        )
    return FullEvidencePipeline(
        baseline,
        pad_scorer,
        PassivePADGate(
            PassivePADConfig(
                live_threshold=args.pad_live_threshold,
                model_version=args.pad_model_version,
                threshold_version=args.pad_threshold_version,
                min_frames=args.min_valid_frames,
            )
        ),
        FivePointHeadPoseEstimator(),
        ActiveLivenessGate(),
        IdentityContinuityGate(
            template_embedding,
            ContinuityConfig(
                min_anchor_similarity=args.continuity_threshold,
                window_size=args.continuity_window,
                failures_required=args.continuity_failures,
                threshold_version=args.continuity_threshold_version,
            ),
        ),
        camera_motion_gate=CameraMotionGate(
            CameraMotionConfig(
                max_normalized_motion=args.camera_motion_threshold,
                threshold_version=args.camera_motion_threshold_version,
            )
        ),
        adversarial_inspector=adversarial,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Face-authentication reference prototype"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    enroll_parser = subparsers.add_parser(
        "enroll", help="Create a separate multi-frame template"
    )
    _capture_arguments(enroll_parser)
    enroll_parser.add_argument("--output", required=True)
    enroll_parser.set_defaults(handler=enroll)

    auth_parser = subparsers.add_parser("authenticate", help="Run a face-auth profile")
    _capture_arguments(auth_parser)
    auth_parser.add_argument("--template", required=True)
    auth_parser.add_argument("--threshold", required=True, type=float)
    auth_parser.add_argument("--threshold-version", required=True)
    auth_parser.add_argument("--policy-version", default="face-auth-policy-v1")
    auth_parser.add_argument("--user-id", required=True)
    auth_parser.add_argument("--purpose", default="HIGH_RISK_ACTION")
    auth_parser.add_argument("--context-hash")
    auth_parser.add_argument(
        "--profile",
        choices=[profile.value for profile in SecurityProfile],
        default=SecurityProfile.BASELINE_ONLY.value,
    )
    full = auth_parser.add_argument_group("FULL profile")
    full.add_argument("--pad-model")
    full.add_argument("--pad-model-version", default="unversioned-pad-model")
    full.add_argument(
        "--pad-runtime", choices=["torchscript", "onnx"], default="torchscript"
    )
    full.add_argument(
        "--pad-provider",
        action="append",
        help="ONNX Runtime execution provider; repeat to set fallback order",
    )
    full.add_argument("--pad-live-threshold", type=float, default=0.80)
    full.add_argument("--pad-threshold-version", default="pad-threshold-unvalidated")
    full.add_argument("--pad-input-size", type=int)
    full.add_argument("--pad-live-class-index", type=int)
    full.add_argument(
        "--pad-output-kind", choices=["logits", "probability"]
    )
    full.add_argument("--continuity-threshold", type=float, default=0.65)
    full.add_argument("--continuity-window", type=int, default=5)
    full.add_argument("--continuity-failures", type=int, default=3)
    full.add_argument("--continuity-threshold-version", default="continuity-3-of-5-v1")
    full.add_argument("--camera-motion-threshold", type=float, default=0.035)
    full.add_argument("--camera-motion-threshold-version", default="camera-motion-v1")
    full.add_argument("--adversarial-threshold", type=float)
    full.add_argument(
        "--adversarial-threshold-version", default="feature-squeeze-unvalidated"
    )
    auth_parser.set_defaults(handler=authenticate)
    return parser


def _capture_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video")
    source.add_argument("--camera", type=int)
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument(
        "--preview",
        dest="preview",
        action="store_true",
        help="show the OpenCV capture preview",
    )
    preview.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        help="disable the preview for an intentional headless run",
    )
    parser.set_defaults(preview=None)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--min-valid-frames", type=int, default=5)
    parser.add_argument("--device")
    parser.add_argument("--min-blur-variance", type=float, default=40.0)
    parser.add_argument("--min-brightness", type=float, default=35.0)
    parser.add_argument("--max-brightness", type=float, default=220.0)


def _quality_gate(args) -> QualityGate:
    return QualityGate(
        QualityConfig(
            min_blur_variance=args.min_blur_variance,
            min_mean_brightness=args.min_brightness,
            max_mean_brightness=args.max_brightness,
        )
    )


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
