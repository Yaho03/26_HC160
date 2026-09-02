import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.contracts.validation import check_schema_shape
from src.verification.defenses.probe_log import PROBE_COLUMNS
try:  # CI는 unittest로 돌고 jsonschema를 잠금 의존성에 두지 않는다.
    import jsonschema as _jsonschema  # noqa: F401

    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False

from src.verification.defenses.probe_threshold import (
    InsufficientCleanSamplesError,
    build_artifact,
    derive_limitations,
)


def _write_probe(path, *, n_clean, n_adversarial, sessions=("s1",), subjects=("p01",), attack_kinds=("pgd",)):
    rows = []
    index = 0
    for session in sessions:
        for subject in subjects:
            for i in range(n_clean):
                rows.append((session, subject, f"f{i:06d}_clean", i, "clean", 0.95 - i * 0.0001, ""))
            for i in range(n_adversarial):
                rows.append((session, subject, f"f{i:06d}_adv", i, "adversarial",
                             0.5 - i * 0.001, attack_kinds[i % len(attack_kinds)]))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROBE_COLUMNS)
        writer.writeheader()
        for session, subject, sample_id, frame, label, cos_ot, kind in rows:
            for transform in ("blur0.8", "jpeg_q75"):
                writer.writerow({
                    "session_id": session, "subject_id": subject, "sample_id": sample_id,
                    "frame_idx": frame, "frame_ts_ms": 0.0, "dropped_frames": 0,
                    "label": label, "attack_kind": kind, "transform": transform,
                    "cos_orig_enroll": 0.8, "cos_transformed_enroll": 0.8,
                    "cos_orig_transformed": cos_ot, "embed_ms": 10.0,
                })
            index += 1
    return Path(path)


def _sidecar(**overrides):
    meta = {
        "session_id": "s1", "subject_id": "p01",
        "model": {"name": "InceptionResnetV1", "pretrained": "vggface2",
                  "weights_file": "w.pt", "weights_sha256": "a" * 64,
                  "preprocess": "resize160"},
        "transforms": {"blur0.8": {"radius": 0.8}, "jpeg_q75": {"quality": 75}},
        "attack": {"kind": "pgd_targeted_enroll", "epsilon": 0.03, "steps": 40, "every": 5},
        "jpeg_headroom_q75": 2.2,
        "counters": {"samples_clean": 300, "samples_adversarial": 60},
    }
    meta.update(overrides)
    return meta


class BuildArtifactTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.probe = _write_probe(Path(self._dir.name) / "probe.csv", n_clean=300, n_adversarial=60)

    def tearDown(self):
        self._dir.cleanup()

    def test_calibration_and_evaluation_are_separated(self):
        artifact = build_artifact(
            self.probe, [_sidecar()], target_fpr=0.01, top_k=2, window_frames=1
        )

        self.assertEqual(artifact["calibration"]["n_clean"], 300)
        self.assertEqual(artifact["evaluation"]["n_adversarial"], 60)
        self.assertNotIn("n_adversarial", artifact["calibration"])

    def test_counts_are_in_aggregation_units(self):
        """집계를 켜면 표본 수가 윈도 수가 된다. 프레임 수와 혼동하면 안 된다."""
        artifact = build_artifact(
            self.probe, [_sidecar()], target_fpr=0.01, top_k=2, window_frames=3
        )

        self.assertEqual(artifact["calibration"]["n_clean_windows"], 300 - 3 + 1)
        self.assertEqual(artifact["evaluation"]["n_adversarial"], 60 - 3 + 1)

    def test_achieved_fpr_never_exceeds_the_target(self):
        artifact = build_artifact(self.probe, [_sidecar()], target_fpr=0.01, top_k=2)
        self.assertLessEqual(artifact["calibration"]["achieved_fpr"], 0.01)

    def test_records_model_and_transform_provenance(self):
        artifact = build_artifact(self.probe, [_sidecar()], target_fpr=0.01, top_k=2)

        self.assertEqual(artifact["model"]["weights_sha256"], "a" * 64)
        self.assertIn("blur0.8", artifact["transforms"])
        self.assertRegex(artifact["calibration"]["probe_csv_sha256"], r"^[0-9a-f]{64}$")

    def test_refuses_when_clean_samples_cannot_reach_the_target_fpr(self):
        """표본이 부족하면 임계값을 만들지 않는다. 조용히 FPR 0으로 낮추지 않는다."""
        small = _write_probe(Path(self._dir.name) / "small.csv", n_clean=20, n_adversarial=5)
        with self.assertRaises(InsufficientCleanSamplesError):
            build_artifact(small, [_sidecar()], target_fpr=0.01, top_k=2)

    def test_selection_method_and_direction_are_fixed(self):
        artifact = build_artifact(self.probe, [_sidecar()], target_fpr=0.01, top_k=2)

        self.assertEqual(artifact["selection_method"], "target_fpr")
        self.assertEqual(artifact["score_direction"], "higher_is_adversarial")


class SchemaConformanceTest(unittest.TestCase):
    """생성기와 스키마가 어긋나면 여기서 잡힌다."""

    def _schema(self):
        root = Path(__file__).resolve().parents[2] / "schemas"
        return json.loads(
            (root / "detector-threshold-artifact.schema.json").read_text(encoding="utf-8")
        )

    def _artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            probe = _write_probe(Path(directory) / "probe.csv", n_clean=300, n_adversarial=60)
            return build_artifact(probe, [_sidecar()], target_fpr=0.01, top_k=2)

    def test_generated_artifact_matches_the_schema_shape(self):
        """저장소는 jsonschema를 잠금 의존성에 두지 않는다. 구조 검사는 어디서나 돈다."""
        check_schema_shape(self._artifact(), self._schema())

    @unittest.skipUnless(_HAS_JSONSCHEMA, "jsonschema가 설치돼 있지 않다")
    def test_generated_artifact_validates_with_jsonschema_when_available(self):
        import jsonschema

        jsonschema.validate(self._artifact(), self._schema())

    def test_shape_check_rejects_a_missing_required_field(self):
        """검사기가 실제로 위반을 잡는지 확인한다. 통과만 보면 무의미하다."""
        artifact = self._artifact()
        del artifact["threshold"]
        with self.assertRaises(ValueError):
            check_schema_shape(artifact, self._schema())

    def test_shape_check_rejects_an_unknown_field(self):
        artifact = self._artifact()
        artifact["unexpected"] = 1
        with self.assertRaises(ValueError):
            check_schema_shape(artifact, self._schema())

    def test_limitations_survive_into_the_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            probe = _write_probe(Path(directory) / "probe.csv", n_clean=300, n_adversarial=60)
            artifact = build_artifact(probe, [_sidecar()], target_fpr=0.01, top_k=2)

        self.assertGreaterEqual(len(artifact["limitations"]), 3)


class LimitationsTest(unittest.TestCase):
    def test_single_subject_and_session_are_reported(self):
        limitations = derive_limitations([_sidecar()], subjects={"p01"}, sessions={"s1"})
        joined = " ".join(limitations)

        self.assertIn("피험자", joined)
        self.assertIn("세션", joined)

    def test_single_attack_kind_is_reported(self):
        limitations = derive_limitations([_sidecar()], subjects={"p01"}, sessions={"s1"})
        self.assertTrue(any("공격" in item for item in limitations))

    def test_adaptive_attack_is_always_reported_as_unevaluated(self):
        limitations = derive_limitations(
            [_sidecar(), _sidecar(session_id="s2", subject_id="p02")],
            subjects={"p01", "p02"},
            sessions={"s1", "s2"},
        )
        self.assertTrue(any("adaptive" in item.lower() for item in limitations))


if __name__ == "__main__":
    unittest.main()


class AggregationTest(unittest.TestCase):
    """임계값은 적용 단위와 함께 기록해야 한다.

    face_auth 게이트는 최근 max_frames개 중 최악값을 쓴다. 프레임 단위로 정한
    임계값을 세션에 그대로 쓰면 실현 FPR이 윈도 크기만큼 배가된다.
    """

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.probe = _write_probe(
            Path(self._dir.name) / "probe.csv", n_clean=300, n_adversarial=60
        )

    def tearDown(self):
        self._dir.cleanup()

    def test_artifact_records_the_aggregation_unit(self):
        artifact = build_artifact(
            self.probe, [_sidecar()], target_fpr=0.01, top_k=2, window_frames=3
        )

        self.assertEqual(artifact["aggregation"]["unit"], "session")
        self.assertEqual(artifact["aggregation"]["window_frames"], 3)
        self.assertEqual(artifact["aggregation"]["rule"], "max")

    def test_window_one_is_recorded_as_frame_unit(self):
        artifact = build_artifact(
            self.probe, [_sidecar()], target_fpr=0.01, top_k=2, window_frames=1
        )
        self.assertEqual(artifact["aggregation"]["unit"], "frame")

    def test_threshold_is_calibrated_on_aggregated_clean_windows(self):
        """세션 단위 임계값은 프레임 단위보다 높아야 한다."""
        frame = build_artifact(
            self.probe, [_sidecar()], target_fpr=0.01, top_k=2, window_frames=1
        )
        session = build_artifact(
            self.probe, [_sidecar()], target_fpr=0.01, top_k=2, window_frames=3
        )

        self.assertGreater(session["threshold"], frame["threshold"])
        self.assertLessEqual(session["calibration"]["achieved_fpr"], 0.01)

    def test_clean_cost_budget_is_reported(self):
        artifact = build_artifact(
            self.probe, [_sidecar()], target_fpr=0.01, top_k=2, window_frames=3
        )
        evaluation = artifact["evaluation"]

        self.assertIsNotNone(evaluation["clean_tar_delta_pp"])
        self.assertLessEqual(evaluation["clean_tar_delta_pp"], 0.0)
        self.assertIsInstance(evaluation["meets_clean_cost_budget"], bool)


class PerAttackKindTest(unittest.TestCase):
    """07 7절: 공격 성공률을 단일 평균으로 숨기지 않고 종류별로 보고한다."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._dir.cleanup()

    def test_each_kind_is_reported_separately(self):
        probe = _write_probe(
            Path(self._dir.name) / "probe.csv",
            n_clean=300, n_adversarial=60, attack_kinds=("pgd", "fgsm"),
        )
        artifact = build_artifact(probe, [_sidecar()], target_fpr=0.01, top_k=2, window_frames=1)
        by_kind = artifact["evaluation"]["tpr_by_attack_kind"]

        self.assertEqual(set(by_kind), {"pgd", "fgsm"})
        for kind, bucket in by_kind.items():
            self.assertEqual(bucket["total"], 30, kind)

    def test_numerator_and_denominator_are_reported(self):
        """종류별 표본이 적으면 점추정만으로 판단할 수 없다."""
        probe = _write_probe(
            Path(self._dir.name) / "probe.csv",
            n_clean=300, n_adversarial=60, attack_kinds=("pgd", "fgsm"),
        )
        artifact = build_artifact(probe, [_sidecar()], target_fpr=0.01, top_k=2, window_frames=1)

        for bucket in artifact["evaluation"]["tpr_by_attack_kind"].values():
            self.assertIn("detected", bucket)
            self.assertIn("total", bucket)


class LimitationHonestyTest(unittest.TestCase):
    """측정한 것을 미측정이라고 적으면 한계 목록 전체의 신뢰가 떨어진다."""

    def test_measured_clean_cost_is_not_listed_as_unmeasured(self):
        limitations = derive_limitations(
            [_sidecar()], subjects={"p01"}, sessions={"s1"}, clean_tar_delta_pp=-0.5
        )
        self.assertFalse(any("측정하지 않았다" in item and "TAR" in item for item in limitations))

    def test_unmeasured_clean_cost_is_listed(self):
        limitations = derive_limitations(
            [_sidecar()], subjects={"p01"}, sessions={"s1"}, clean_tar_delta_pp=None
        )
        self.assertTrue(any("TAR delta" in item for item in limitations))

    def test_budget_overrun_is_listed_as_a_limitation(self):
        limitations = derive_limitations(
            [_sidecar()], subjects={"p01"}, sessions={"s1"}, clean_tar_delta_pp=-3.02
        )
        self.assertTrue(any("초과" in item for item in limitations))
