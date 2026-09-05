from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.common.reproducibility import git_state, stable_json_sha256
from src.experiments.artifact_registration import (
    ArtifactRegistrationError,
    load_registration_context,
    preflight_registration,
    register_completed_output,
)
from src.experiments.run_manifest import RunManifest
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
from src.face_auth.application.decision_artifact import (
    DecisionArtifactError,
    build_decision_artifact,
    validate_decision_output,
    write_decision_artifact,
)
from src.face_auth.application.evidence_service import build_capture_manifest
from src.face_auth.application.evaluation_service import EvaluationService
from src.face_auth.application.session_service import SessionService
from src.face_auth.application.token_service import TokenService
from src.face_auth.domain.policy import PolicyEngine
from src.face_auth.domain.types import (
    DecisionAction,
    FramePacket,
    GateResult,
    GateStatus,
    SecurityProfile,
    SessionState,
)
from src.face_auth.inference.face_detector import MTCNNFaceDetector
from src.face_auth.inference.active_liveness import ActiveLivenessGate
from src.face_auth.inference.adversarial_detector import (
    TemplateShiftDetector,
    TemplateShiftDetectorConfig,
    AdversarialDetectorConfig,
    TransformConsistencyDetector,
)
from src.face_auth.inference.camera_motion import CameraMotionConfig, CameraMotionGate
from src.face_auth.inference.continuity import ContinuityConfig, IdentityContinuityGate
from src.face_auth.inference.content_replay import (
    ContentReplayConfig,
    ContentReplayGate,
    ContentReplayMonitor,
)
from src.face_auth.inference.feature_squeeze import (
    FeatureSqueezeConfig,
    FeatureSqueezeInspector,
)
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
    challenge_start_frame_id: int | None = None
    live_veto: GateResult | None = None


class ChallengeBindingError(RuntimeError):
    """Raised when FULL evidence is not bound to a displayed challenge."""


def _collect(
    source: FrameSource,
    frame_count: int,
    *,
    preview: OpenCVPreview | None = None,
    purpose: str = "CAPTURE",
    instruction: str | None = None,
    live_replay_monitor: ContentReplayMonitor | None = None,
    monitor_start_frame_id: int | None = None,
) -> CaptureBatch:
    frames = []
    challenge_start_frame_id = None
    try:
        while len(frames) < frame_count:
            frame = source.read()
            if frame is None:
                break
            frames.append(frame)
            effective_marker = (
                challenge_start_frame_id
                if challenge_start_frame_id is not None
                else monitor_start_frame_id
            )
            live_veto = None
            if (
                live_replay_monitor is not None
                and effective_marker is not None
                and frame.frame_id >= effective_marker
            ):
                live_result = live_replay_monitor.update(frame)
                if (
                    frame.frame_id > effective_marker
                    and live_result.status is GateStatus.FAIL
                ):
                    live_veto = live_result
            if preview is not None:
                continue_capture = preview.show(
                    frame,
                    captured_frames=len(frames),
                    target_frames=frame_count,
                    purpose=purpose,
                    instruction=instruction,
                    alert="REPLAY DETECTED" if live_veto is not None else None,
                    wait_ms=900 if live_veto is not None else 1,
                )
                if instruction is not None and challenge_start_frame_id is None:
                    challenge_start_frame_id = frame.frame_id
                    if live_replay_monitor is not None:
                        live_replay_monitor.update(frame)
                if live_veto is not None:
                    return CaptureBatch(
                        tuple(frames),
                        challenge_start_frame_id=challenge_start_frame_id,
                        live_veto=live_veto,
                    )
                if not continue_capture:
                    return CaptureBatch(
                        tuple(frames),
                        cancelled=True,
                        challenge_start_frame_id=challenge_start_frame_id,
                    )
            elif live_veto is not None:
                return CaptureBatch(
                    tuple(frames),
                    challenge_start_frame_id=challenge_start_frame_id,
                    live_veto=live_veto,
                )
    finally:
        source.close()
        if preview is not None:
            preview.close()
    return CaptureBatch(
        tuple(frames), challenge_start_frame_id=challenge_start_frame_id
    )


def _capture(
    args,
    purpose: str,
    *,
    instruction: str | None = None,
    live_replay_monitor: ContentReplayMonitor | None = None,
) -> CaptureBatch:
    preview_enabled = _preview_enabled(args)
    preview = OpenCVPreview() if preview_enabled else None
    return _collect(
        _source(args),
        args.frames,
        preview=preview,
        purpose=purpose,
        instruction=instruction,
        live_replay_monitor=live_replay_monitor,
        monitor_start_frame_id=(
            None
            if preview_enabled
            else getattr(args, "challenge_start_frame_id", None)
        ),
    )


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


def _challenge_instruction(challenge_kind: str) -> str:
    instructions = {
        "HEAD_LEFT": "TURN HEAD LEFT",
        "HEAD_RIGHT": "TURN HEAD RIGHT",
        "BLINK": "BLINK ONCE",
    }
    try:
        return instructions[challenge_kind]
    except KeyError as error:
        raise ChallengeBindingError(
            f"Unsupported challenge kind: {challenge_kind}"
        ) from error


def _resolve_challenge_start_frame_id(args, capture: CaptureBatch) -> int:
    if (
        capture.challenge_start_frame_id is not None
        and args.challenge_start_frame_id is not None
    ):
        raise ChallengeBindingError(
            "Do not provide --challenge-start-frame-id when the preview records "
            "the displayed challenge boundary."
        )
    marker = capture.challenge_start_frame_id
    if marker is None:
        marker = args.challenge_start_frame_id
    if marker is None:
        raise ChallengeBindingError(
            "FULL headless or recorded-video capture requires "
            "--challenge-start-frame-id from the external challenge presenter."
        )

    frame_ids = {frame.frame_id for frame in capture.frames}
    if marker not in frame_ids:
        raise ChallengeBindingError(
            "challenge_start_frame_id must identify a captured frame."
        )
    post_challenge_frames = sum(
        frame.frame_id > marker for frame in capture.frames
    )
    if post_challenge_frames < args.min_valid_frames:
        raise ChallengeBindingError(
            "Not enough post-challenge frames for the configured minimum."
        )
    return marker


def _challenge_binding_failure(error: ChallengeBindingError, session_id: str) -> int:
    print(
        json.dumps(
            {
                "status": "CHALLENGE_BINDING_ERROR",
                "reason_code": "CHALLENGE_BOUNDARY_INVALID",
                "message": str(error),
                "session_id": session_id,
                "state": SessionState.ERROR.value,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 3


def _validate_challenge_presentation(args) -> None:
    preview_enabled = _preview_enabled(args)
    if preview_enabled and args.challenge_start_frame_id is not None:
        raise ChallengeBindingError(
            "Do not provide --challenge-start-frame-id when the preview records "
            "the displayed challenge boundary."
        )
    if not preview_enabled and args.challenge_start_frame_id is None:
        raise ChallengeBindingError(
            "FULL headless or recorded-video capture requires "
            "--challenge-start-frame-id from the external challenge presenter."
        )


def _content_replay_config(args) -> ContentReplayConfig:
    return ContentReplayConfig(
        max_near_duplicate_run=args.content_replay_max_run,
        max_mean_absolute_difference=args.content_replay_max_difference,
        threshold_version=args.content_replay_threshold_version,
    )


def _capture_configuration_failure(error: ValueError, session_id: str) -> int:
    print(
        json.dumps(
            {
                "status": "CAPTURE_CONFIG_ERROR",
                "reason_code": "INVALID_CONTENT_REPLAY_CONFIG",
                "message": str(error),
                "session_id": session_id,
                "state": SessionState.ERROR.value,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 3


def _prepare_decision_output(args) -> None:
    args._registration_context = None
    if args.decision_output is None:
        if args.registration_context is not None:
            raise DecisionArtifactError(
                "--registration-context requires --decision-output"
            )
        return
    protected_inputs = [args.template]
    if args.video is not None:
        protected_inputs.append(args.video)
    if args.registration_context is not None:
        protected_inputs.append(args.registration_context)
    validate_decision_output(
        args.decision_output,
        overwrite=args.overwrite_decision_output,
        protected_inputs=protected_inputs,
    )
    if args.registration_context is None:
        return
    if args.overwrite_decision_output:
        raise DecisionArtifactError(
            "registered decision outputs are immutable; remove "
            "--overwrite-decision-output"
        )
    if args.device is None:
        raise DecisionArtifactError(
            "registered decision outputs require an explicit --device"
        )
    try:
        context = load_registration_context(args.registration_context)
        preflight_registration(args.decision_output)
        code_state = git_state(args.registration_repo_dir)
    except (ArtifactRegistrationError, OSError, subprocess.CalledProcessError) as error:
        raise DecisionArtifactError(
            f"unable to prepare decision registration: {error}"
        ) from error
    args._registration_context = context
    args._registration_code_state = code_state
    args._registration_started_at = RunManifest.utc_now()


def _persist_decision_artifact(args, payload: dict) -> None:
    if args.decision_output is None:
        return
    context = args._registration_context
    if context is not None:
        try:
            current_code_state = git_state(args.registration_repo_dir)
        except (OSError, subprocess.CalledProcessError) as error:
            raise DecisionArtifactError(
                f"unable to recheck decision producer state: {error}"
            ) from error
        if current_code_state != args._registration_code_state:
            raise DecisionArtifactError(
                "repository state changed during authentication"
            )
    write_decision_artifact(
        args.decision_output,
        payload,
        overwrite=args.overwrite_decision_output,
    )
    if context is None:
        return
    try:
        register_completed_output(
            args.decision_output,
            context=context,
            kind="decision",
            created_at=payload["created_at"],
            config_sha256=_authentication_config_sha256(args),
            git_commit=args._registration_code_state["git_commit"],
            dirty_worktree=args._registration_code_state["dirty_worktree"],
            device={"type": args.device},
            started_at=args._registration_started_at,
            ended_at=RunManifest.utc_now(),
        )
    except (ArtifactRegistrationError, OSError) as error:
        Path(args.decision_output).unlink(missing_ok=True)
        raise DecisionArtifactError(
            f"unable to register decision artifact: {error}"
        ) from error


def _authentication_config_sha256(args) -> str:
    names = (
        "profile",
        "threshold",
        "threshold_version",
        "policy_version",
        "purpose",
        "frames",
        "min_valid_frames",
        "device",
        "min_blur_variance",
        "min_brightness",
        "max_brightness",
        "pad_model_version",
        "pad_runtime",
        "pad_provider",
        "pad_live_threshold",
        "pad_threshold_version",
        "pad_input_size",
        "pad_live_class_index",
        "pad_output_kind",
        "continuity_threshold",
        "continuity_window",
        "continuity_failures",
        "continuity_threshold_version",
        "camera_motion_threshold",
        "camera_motion_threshold_version",
        "content_replay_max_run",
        "content_replay_max_difference",
        "content_replay_threshold_version",
        "adversarial_threshold",
        "adversarial_threshold_version",
        "adversarial_template_threshold",
        "adversarial_template_threshold_version",
        # 변환 범위도 preprocessing의 일부다. ADR-003에 따라 해시에 포함한다.
        "adversarial_randomize",
        "adversarial_range_preset",
        "adversarial_families",
    )
    configuration = {name: getattr(args, name) for name in names}
    configuration["source_kind"] = "video" if args.video is not None else "camera"
    return stable_json_sha256(configuration)


def _decision_artifact_failure(
    error: DecisionArtifactError, session_id: str | None = None
) -> int:
    payload = {
        "status": "DECISION_ARTIFACT_ERROR",
        "reason_code": "DECISION_ARTIFACT_WRITE_FAILED",
        "message": str(error),
    }
    if session_id is not None:
        payload["session_id"] = session_id
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 3


def _live_security_veto(
    *,
    args,
    session_id: str,
    challenge_kind: str,
    policy_version: str,
    manifest,
    gate_result: GateResult,
) -> int:
    artifact = build_decision_artifact(
        session_id=session_id,
        attempt_id=manifest.attempt_id,
        security_profile=SecurityProfile.FULL,
        state=SessionState.SECURITY_DENIED,
        result_status="LIVE_SECURITY_VETO",
        decision=DecisionAction.SECURITY_DENIED,
        policy_version=policy_version,
        reason_codes=gate_result.reason_codes,
        challenge_kind=challenge_kind,
        challenge_start_frame_id=manifest.challenge_start_frame_id,
        evidence_digest=manifest.evidence_digest,
        total_frames=manifest.frame_count,
        valid_face_frames=None,
        gate_results=(gate_result,),
        token_id=None,
        decision_id=_registered_decision_id(args),
    )
    try:
        _persist_decision_artifact(args, artifact)
    except DecisionArtifactError as error:
        return _decision_artifact_failure(error, session_id)
    print(
        json.dumps(
            {
                "status": "LIVE_SECURITY_VETO",
                "session_id": session_id,
                "security_profile": SecurityProfile.FULL.value,
                "state": SessionState.SECURITY_DENIED.value,
                "decision": "SECURITY_DENIED",
                "policy_version": policy_version,
                "challenge": challenge_kind,
                "challenge_start_frame_id": manifest.challenge_start_frame_id,
                "reason_codes": list(gate_result.reason_codes),
                "total_frames": manifest.frame_count,
                "attempt_id": manifest.attempt_id,
                "evidence_digest": manifest.evidence_digest,
                "token_id": None,
                "gate": {
                    "gate": gate_result.gate,
                    "status": gate_result.status.value,
                    "score": gate_result.score,
                    "threshold": gate_result.threshold,
                    "reason_codes": list(gate_result.reason_codes),
                    "threshold_version": gate_result.threshold_version,
                },
                "warning": (
                    "Live replay thresholds require target-camera validation."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 2


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
    try:
        _prepare_decision_output(args)
    except DecisionArtifactError as error:
        return _decision_artifact_failure(error)
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
    instruction = (
        _challenge_instruction(challenge.kind)
        if profile is SecurityProfile.FULL
        else None
    )
    live_replay_monitor = None
    if profile is SecurityProfile.FULL:
        try:
            _validate_challenge_presentation(args)
            live_replay_monitor = ContentReplayMonitor(
                _content_replay_config(args)
            )
        except ChallengeBindingError as error:
            sessions.move(session.session_id, SessionState.ERROR)
            return _challenge_binding_failure(error, session.session_id)
        except ValueError as error:
            sessions.move(session.session_id, SessionState.ERROR)
            return _capture_configuration_failure(error, session.session_id)
    try:
        capture = _capture(
            args,
            "AUTHENTICATION",
            instruction=instruction,
            live_replay_monitor=live_replay_monitor,
        )
    except (CaptureSourceError, PreviewUnavailableError) as error:
        sessions.move(session.session_id, SessionState.ERROR)
        return _capture_failure(error, session_id=session.session_id)
    if capture.live_veto is not None:
        challenge_start_frame_id = (
            capture.challenge_start_frame_id
            if capture.challenge_start_frame_id is not None
            else args.challenge_start_frame_id
        )
        manifest = build_capture_manifest(
            session,
            list(capture.frames),
            challenge_start_frame_id=challenge_start_frame_id,
        )
        sessions.move(session.session_id, SessionState.SECURITY_DENIED)
        return _live_security_veto(
            args=args,
            session_id=session.session_id,
            challenge_kind=challenge.kind,
            policy_version=args.policy_version,
            manifest=manifest,
            gate_result=capture.live_veto,
        )
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
    challenge_start_frame_id = None
    if profile is SecurityProfile.FULL:
        try:
            challenge_start_frame_id = _resolve_challenge_start_frame_id(
                args, capture
            )
        except ChallengeBindingError as error:
            sessions.move(session.session_id, SessionState.ERROR)
            return _challenge_binding_failure(error, session.session_id)
    manifest = build_capture_manifest(
        session,
        frames,
        challenge_start_frame_id=challenge_start_frame_id,
    )
    sessions.move(session.session_id, SessionState.EVIDENCE_RECEIVED)
    sessions.move(session.session_id, SessionState.EVALUATING)
    if profile is SecurityProfile.FULL:
        full_pipeline = _full_pipeline(args, pipeline, embedder, template.embedding)
        full_observation = full_pipeline.evaluate(
            frames,
            challenge_kind=challenge.kind,
            challenge_start_frame_id=challenge_start_frame_id,
        )
        observation = full_observation.baseline
        gate_results = list(full_observation.gate_results)
    else:
        observation = pipeline.evaluate(frames)
        gate_results = list(observation.gate_results)
    outcome = evaluation.evaluate(session.session_id, gate_results)

    artifact = build_decision_artifact(
        session_id=session.session_id,
        attempt_id=manifest.attempt_id,
        security_profile=session.security_profile,
        state=session.state,
        result_status="POLICY_DECISION",
        decision=outcome.decision.action,
        policy_version=outcome.decision.policy_version,
        reason_codes=outcome.decision.reason_codes,
        challenge_kind=challenge.kind,
        challenge_start_frame_id=manifest.challenge_start_frame_id,
        evidence_digest=manifest.evidence_digest,
        total_frames=observation.total_frames,
        valid_face_frames=observation.valid_face_frames,
        gate_results=outcome.decision.gate_results,
        token_id=outcome.token.token_id if outcome.token else None,
        decision_id=_registered_decision_id(args),
    )
    try:
        _persist_decision_artifact(args, artifact)
    except DecisionArtifactError as error:
        return _decision_artifact_failure(error, session.session_id)

    payload = {
        "session_id": session.session_id,
        "security_profile": session.security_profile.value,
        "state": session.state.value,
        "challenge": challenge.kind,
        "challenge_start_frame_id": manifest.challenge_start_frame_id,
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


def _registered_decision_id(args) -> str | None:
    context = getattr(args, "_registration_context", None)
    return context.artifact_id if context is not None else None


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
        # 두 detector는 서로 다른 것을 잰다. transform-consistency는 조작 흔적의
        # 유무를, template-shift는 그 조작이 등록자로 위장하는 방향인지를 본다.
        # template 게이트는 자체 임계값이 주어질 때만 켜진다.
        template_detector = None
        if args.adversarial_template_threshold is not None:
            template_detector = TemplateShiftDetector(
                TemplateShiftDetectorConfig(
                    max_template_shift=args.adversarial_template_threshold,
                    threshold_version=args.adversarial_template_threshold_version,
                ),
                template_embedding,
            )
        # 계열과 프리셋을 명시적으로 넘긴다. 연구 트랙 모듈의 기본값에 기대면 그쪽
        # 변경이 인증 동작을 조용히 바꾼다.
        squeeze_config = FeatureSqueezeConfig(
            randomized=args.adversarial_randomize,
            families=tuple(
                name.strip() for name in args.adversarial_families.split(",") if name.strip()
            ),
            range_preset=args.adversarial_range_preset,
        )
        adversarial = FeatureSqueezeInspector(
            embedder,
            TransformConsistencyDetector(
                AdversarialDetectorConfig(
                    max_cosine_distance=args.adversarial_threshold,
                    threshold_version=args.adversarial_threshold_version,
                )
            ),
            config=squeeze_config,
            template_detector=template_detector,
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
        content_replay_gate=ContentReplayGate(_content_replay_config(args)),
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
        "--decision-output",
        help="write a privacy-minimized authentication decision JSON artifact",
    )
    auth_parser.add_argument(
        "--overwrite-decision-output",
        action="store_true",
        help="replace an existing decision artifact",
    )
    auth_parser.add_argument(
        "--registration-context",
        help=(
            "write immutable artifact-reference and run-manifest sidecars from "
            "an explicit registration context"
        ),
    )
    auth_parser.add_argument(
        "--registration-repo-dir",
        default=".",
        help="repository used to record the decision producer commit",
    )
    auth_parser.add_argument(
        "--profile",
        choices=[profile.value for profile in SecurityProfile],
        default=SecurityProfile.BASELINE_ONLY.value,
    )
    full = auth_parser.add_argument_group("FULL profile")
    full.add_argument("--pad-model")
    full.add_argument("--pad-model-version", default="unversioned-pad-model")
    full.add_argument(
        "--challenge-start-frame-id",
        type=int,
        help=(
            "captured frame where an external challenge UI first presented the "
            "FULL challenge; required for recorded-video or headless capture"
        ),
    )
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
    full.add_argument("--content-replay-max-run", type=int, default=2)
    full.add_argument("--content-replay-max-difference", type=float, default=0.30)
    full.add_argument(
        "--content-replay-threshold-version", default="content-replay-v2"
    )
    full.add_argument("--adversarial-threshold", type=float)
    full.add_argument(
        "--adversarial-template-threshold",
        type=float,
        help=(
            "등록 템플릿 기준 이동량 게이트의 임계값. 생략하면 이 게이트를 켜지 않는다. "
            "--adversarial-threshold 와 함께 지정해야 한다"
        ),
    )
    full.add_argument(
        "--adversarial-template-threshold-version",
        default="template-shift-unvalidated",
    )
    full.add_argument(
        "--adversarial-randomize",
        action="store_true",
        help=(
            "squeeze 변환 파라미터를 매 프레임 무작위로 뽑는다. 연구 트랙에서 고정 "
            "변환의 BPDA 탐지율이 0%%였고 랜덤화로 84%%가 됐으나, face_auth 경로에서는 "
            "아직 측정하지 않았다"
        ),
    )
    full.add_argument(
        "--adversarial-range-preset",
        default="narrow",
        help="랜덤화 파라미터 범위 프리셋. 인증 config 해시에 포함된다",
    )
    full.add_argument(
        "--adversarial-families",
        default="jpeg,bit,blur",
        help="쉼표로 구분한 변환 계열. 기본값은 배포된 3종이다",
    )
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
