import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.common.reproducibility import sha256_file
from src.face_auth.application.decision_artifact import (
    DecisionArtifactError,
    build_artifact_reference,
    build_decision_artifact,
    validate_decision_output,
    write_decision_artifact,
)
from src.face_auth.cli import CaptureBatch, authenticate, build_parser
from src.face_auth.domain.types import (
    DecisionAction,
    GateResult,
    GateStatus,
    SecurityProfile,
    SessionState,
    FramePacket,
)


def decision_payload(**overrides):
    values = {
        "session_id": "session-1",
        "attempt_id": "attempt-1",
        "security_profile": SecurityProfile.FULL,
        "state": SessionState.SECURITY_DENIED,
        "result_status": "LIVE_SECURITY_VETO",
        "decision": DecisionAction.SECURITY_DENIED,
        "policy_version": "policy-v1",
        "reason_codes": ("FRAME_SEQUENCE_INVALID",),
        "challenge_kind": "HEAD_LEFT",
        "challenge_start_frame_id": 3,
        "evidence_digest": "a" * 64,
        "total_frames": 4,
        "valid_face_frames": None,
        "gate_results": (
            GateResult(
                "content_replay",
                GateStatus.FAIL,
                score=3.0,
                threshold=2.0,
                reason_codes=("FRAME_SEQUENCE_INVALID",),
                threshold_version="content-replay-v2",
                latency_ms=1.25,
            ),
        ),
        "token_id": None,
        "decision_id": "decision-1",
        "created_at": "2026-08-23T00:00:00Z",
    }
    values.update(overrides)
    return build_decision_artifact(**values)


class DecisionArtifactTest(unittest.TestCase):
    def test_live_veto_payload_matches_schema_shape_and_omits_identity_data(self):
        payload = decision_payload()
        schema = json.loads(
            (Path("schemas") / "authentication-decision.schema.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(set(payload), set(schema["required"]))
        self.assertEqual(
            set(payload["gates"][0]),
            set(schema["$defs"]["gate"]["required"]),
        )
        self.assertEqual(payload["decision"]["token_id"], None)
        self.assertNotIn("user_id", json.dumps(payload))
        self.assertNotIn("nonce", json.dumps(payload))

    def test_verified_decision_requires_a_token(self):
        with self.assertRaisesRegex(DecisionArtifactError, "require a token_id"):
            decision_payload(
                result_status="POLICY_DECISION",
                decision=DecisionAction.VERIFIED,
                state=SessionState.VERIFIED,
            )

    def test_non_verified_decision_rejects_a_token(self):
        with self.assertRaisesRegex(DecisionArtifactError, "only VERIFIED"):
            decision_payload(token_id="token-1")

    def test_terminal_state_must_match_decision(self):
        with self.assertRaisesRegex(DecisionArtifactError, "state must match"):
            decision_payload(state=SessionState.ERROR)

    def test_writer_is_canonical_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "decision.json"
            payload = decision_payload()
            write_decision_artifact(output, payload)

            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)
            with self.assertRaisesRegex(DecisionArtifactError, "overwrite"):
                write_decision_artifact(output, payload)

            changed = decision_payload(decision_id="decision-2")
            write_decision_artifact(output, changed, overwrite=True)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["decision_id"],
                "decision-2",
            )

    def test_output_cannot_replace_an_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "template.npz"
            source.write_bytes(b"template")
            with self.assertRaisesRegex(DecisionArtifactError, "differ"):
                validate_decision_output(source, protected_inputs=(source,))

    def test_artifact_reference_binds_written_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "decision.json"
            payload = decision_payload()
            write_decision_artifact(output, payload)
            reference = build_artifact_reference(
                output,
                artifact_id=payload["decision_id"],
                relative_uri="artifacts/run-1/decision.json",
                created_at=payload["created_at"],
                producer_run_id="run-1",
            )

            self.assertEqual(reference["kind"], "decision")
            self.assertEqual(reference["sha256"], sha256_file(output))
            self.assertEqual(reference["bytes"], output.stat().st_size)

    def test_artifact_reference_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "decision.json"
            write_decision_artifact(output, decision_payload())
            with self.assertRaisesRegex(DecisionArtifactError, "traversal"):
                build_artifact_reference(
                    output,
                    artifact_id="decision-1",
                    relative_uri="../decision.json",
                    created_at="2026-08-23T00:00:00Z",
                )

    def test_authenticate_writes_a_verified_policy_decision(self):
        passing_gates = (
            GateResult("frame_integrity", GateStatus.PASS),
            GateResult("quality", GateStatus.PASS),
            GateResult("single_face", GateStatus.PASS),
            GateResult(
                "identity",
                GateStatus.PASS,
                score=0.91,
                threshold=0.7,
                model_version="facenet-v1",
                threshold_version="identity-v1",
            ),
        )
        frames = tuple(
            FramePacket(index, float(index), np.full((8, 8, 3), index, np.uint8))
            for index in range(5)
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "decision.json"
            args = build_parser().parse_args(
                [
                    "authenticate",
                    "--video",
                    "probe.mp4",
                    "--template",
                    "template.npz",
                    "--threshold",
                    "0.7",
                    "--threshold-version",
                    "identity-v1",
                    "--user-id",
                    "user-1",
                    "--decision-output",
                    str(output),
                ]
            )
            pipeline = Mock()
            pipeline.evaluate.return_value = SimpleNamespace(
                total_frames=5,
                valid_face_frames=5,
                gate_results=passing_gates,
            )
            with (
                patch(
                    "src.face_auth.cli.load_template",
                    return_value=SimpleNamespace(
                        embedding=np.ones(512, dtype=np.float32),
                        model_version="facenet-v1",
                    ),
                ),
                patch("src.face_auth.cli.MTCNNFaceDetector"),
                patch("src.face_auth.cli.FaceNetEmbedder"),
                patch(
                    "src.face_auth.cli.BaselineEvidencePipeline",
                    return_value=pipeline,
                ),
                patch(
                    "src.face_auth.cli._capture",
                    return_value=CaptureBatch(frames),
                ),
                redirect_stdout(io.StringIO()) as stdout,
            ):
                exit_code = authenticate(args)

            artifact = json.loads(output.read_text(encoding="utf-8"))
            console = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(artifact["result_status"], "POLICY_DECISION")
            self.assertEqual(artifact["decision"]["action"], "VERIFIED")
            self.assertEqual(artifact["decision"]["token_id"], console["token_id"])
            self.assertEqual(artifact["evidence"]["valid_face_frames"], 5)
            self.assertNotIn("user-1", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
