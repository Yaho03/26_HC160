"""계측 세션 도착 시 실행하는 파이프라인.

지금까지 손으로 치던 압축 해제, 검증, 합치기, artifact 생성, 피험자 분리 검증을
한 명령으로 묶는다. 검증 로직은 순수 함수로 두어 파일과 카메라 없이 테스트한다.
"""

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.probe_pipeline import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_VALIDATION,
    ZipContentError,
    check_sidecar,
    extract_session,
    format_report,
    session_dir_name,
    stage_ok,
    stage_skipped,
)


def _meta(**overrides):
    meta = {
        "session_id": "abc123def456", "subject_id": "p03", "completed": True,
        "jpeg_headroom_q75": 1.9,
        "counters": {"frames_read": 220, "frames_without_face": 20,
                     "samples_clean": 200, "samples_adversarial": 40},
    }
    meta.update(overrides)
    return meta


class CheckSidecarTest(unittest.TestCase):
    def test_healthy_session_produces_no_findings(self):
        self.assertEqual(check_sidecar(_meta()), [])

    def test_incomplete_session_is_flagged(self):
        findings = check_sidecar(_meta(completed=False))
        self.assertEqual([f.check for f in findings], ["completed"])

    def test_low_headroom_is_flagged(self):
        findings = check_sidecar(_meta(jpeg_headroom_q75=0.08))
        self.assertEqual([f.check for f in findings], ["jpeg_headroom_q75"])
        self.assertIn("0.08", findings[0].message)

    def test_missing_headroom_is_flagged(self):
        self.assertEqual(
            [f.check for f in check_sidecar(_meta(jpeg_headroom_q75=None))],
            ["jpeg_headroom_q75"],
        )

    def test_low_face_detection_rate_is_flagged(self):
        meta = _meta(counters={"frames_read": 200, "frames_without_face": 150,
                               "samples_clean": 200, "samples_adversarial": 40})
        self.assertEqual([f.check for f in check_sidecar(meta)], ["face_rate"])

    def test_too_few_clean_samples_is_flagged(self):
        meta = _meta(counters={"frames_read": 100, "frames_without_face": 5,
                               "samples_clean": 40, "samples_adversarial": 8})
        self.assertEqual([f.check for f in check_sidecar(meta)], ["samples_clean"])

    def test_multiple_problems_are_all_reported(self):
        meta = _meta(completed=False, jpeg_headroom_q75=0.05,
                     counters={"frames_read": 100, "frames_without_face": 90,
                               "samples_clean": 10, "samples_adversarial": 2})
        self.assertEqual(
            {f.check for f in check_sidecar(meta)},
            {"completed", "jpeg_headroom_q75", "face_rate", "samples_clean"},
        )

    def test_zero_frames_read_does_not_divide_by_zero(self):
        meta = _meta(counters={"frames_read": 0, "frames_without_face": 0,
                               "samples_clean": 0, "samples_adversarial": 0})
        checks = {f.check for f in check_sidecar(meta)}
        self.assertIn("face_rate", checks)


class SessionDirNameTest(unittest.TestCase):
    def test_uses_the_session_id(self):
        self.assertEqual(session_dir_name(_meta()), "abc123def456")

    def test_rejects_a_non_opaque_session_id(self):
        """zip 파일명에 실명이 있을 수 있다. 경로는 세션 ID로만 만든다."""
        from src.verification.defenses.probe_log import OpaqueIdError

        with self.assertRaises(OpaqueIdError):
            session_dir_name(_meta(session_id="../../etc/passwd"))

    def test_rejects_a_non_opaque_subject_id(self):
        from src.verification.defenses.probe_log import OpaqueIdError

        with self.assertRaises(OpaqueIdError):
            session_dir_name(_meta(subject_id="이도현"))


class ExtractSessionTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def _zip(self, name, members):
        path = self.root / name
        with zipfile.ZipFile(path, "w") as archive:
            for member, content in members.items():
                archive.writestr(member, content)
        return path

    def test_extracts_into_a_session_id_directory(self):
        archive = self._zip("이름_p03_1.zip", {
            "probe.csv": "session_id\n", "session.json": json.dumps(_meta()),
        })
        destination = extract_session(archive, self.root / "out")

        self.assertEqual(destination.name, "abc123def456")
        self.assertTrue((destination / "probe.csv").exists())

    def test_original_filename_never_reaches_the_path(self):
        archive = self._zip("실명_p03.zip", {
            "probe.csv": "x\n", "session.json": json.dumps(_meta()),
        })
        destination = extract_session(archive, self.root / "out")

        self.assertNotIn("실명", str(destination))

    def test_rejects_unexpected_members(self):
        archive = self._zip("s.zip", {
            "probe.csv": "x\n", "session.json": json.dumps(_meta()),
            "face.png": "binary",
        })
        with self.assertRaises(ZipContentError):
            extract_session(archive, self.root / "out")

    def test_rejects_path_traversal_members(self):
        archive = self._zip("s.zip", {
            "probe.csv": "x\n", "session.json": json.dumps(_meta()),
            "../escape.txt": "x",
        })
        with self.assertRaises(ZipContentError):
            extract_session(archive, self.root / "out")

    def test_rejects_archive_missing_required_members(self):
        archive = self._zip("s.zip", {"probe.csv": "x\n"})
        with self.assertRaises(ZipContentError):
            extract_session(archive, self.root / "out")


class StageStatusTest(unittest.TestCase):
    """건너뜀과 실행했으나 결과 없음을 구분해야 자동 집계가 가능하다."""

    def test_skipped_carries_a_reason(self):
        stage = stage_skipped("cross_subject", "피험자 1명, 최소 2명 필요")
        self.assertEqual(stage["status"], "skipped")
        self.assertIn("2명", stage["reason"])

    def test_ok_with_empty_result_is_not_skipped(self):
        stage = stage_ok("cross_subject", {"results": []})
        self.assertEqual(stage["status"], "ok")
        self.assertIsNone(stage["reason"])
        self.assertEqual(stage["results"], [])

    def test_the_two_are_distinguishable_in_json(self):
        skipped = json.loads(json.dumps(stage_skipped("x", "이유")))
        empty = json.loads(json.dumps(stage_ok("x", {"results": []})))
        self.assertNotEqual(skipped["status"], empty["status"])


class ReportTest(unittest.TestCase):
    def test_skips_appear_before_stage_details(self):
        """하위 항목에 묻히면 완주한 것처럼 읽힌다."""
        report = format_report({
            "stages": [
                stage_ok("merge", {"rows": 100}),
                stage_skipped("cross_subject", "피험자 1명, 최소 2명 필요"),
            ]
        })
        lines = report.splitlines()
        skip_line = next(i for i, l in enumerate(lines) if "SKIPPED" in l)
        merge_line = next(i for i, l in enumerate(lines) if "merge" in l and "SKIPPED" not in l)

        self.assertLess(skip_line, merge_line)

    def test_skip_reason_is_shown(self):
        report = format_report({
            "stages": [stage_skipped("cross_subject", "피험자 1명, 최소 2명 필요")]
        })
        self.assertIn("최소 2명 필요", report)

    def test_report_without_skips_has_no_skip_banner(self):
        report = format_report({"stages": [stage_ok("merge", {"rows": 100})]})
        self.assertNotIn("SKIPPED", report)


class ExitCodeTest(unittest.TestCase):
    """CI에서 검증 실패와 실행 오류를 구분할 수 있어야 한다."""

    def test_codes_are_distinct(self):
        self.assertEqual(len({EXIT_OK, EXIT_ERROR, EXIT_VALIDATION}), 3)

    def test_ok_is_zero(self):
        self.assertEqual(EXIT_OK, 0)

    def test_validation_failure_is_two(self):
        self.assertEqual(EXIT_VALIDATION, 2)


if __name__ == "__main__":
    unittest.main()


class ScriptEntrypointTest(unittest.TestCase):
    """모듈 실행과 스크립트 실행이 둘 다 되어야 한다.

    테스트는 저장소 루트에서 모듈로 돌기 때문에 sys.path 문제가 드러나지 않는다.
    실제 실행 경로를 함께 확인한다.
    """

    def test_runs_as_a_script_from_any_directory(self):
        import subprocess
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / "probe_pipeline.py"), "--help"],
            capture_output=True, text=True, cwd=tempfile.gettempdir(),
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runs_as_a_module(self):
        import subprocess
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "-m", "scripts.probe_pipeline", "--help"],
            capture_output=True, text=True, cwd=str(root),
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class ExitCodeIntegrationTest(unittest.TestCase):
    """CI가 원인을 가릴 수 있어야 하므로 실제 실행에서 코드를 확인한다."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def _zip(self, name, members):
        path = self.root / name
        with zipfile.ZipFile(path, "w") as archive:
            for member, content in members.items():
                archive.writestr(member, content)
        return path

    def _run(self, archive, *extra):
        import subprocess
        from pathlib import Path as _Path

        repo = _Path(__file__).resolve().parents[2]
        return subprocess.run(
            [sys.executable, str(repo / "scripts" / "probe_pipeline.py"), str(archive),
             "--session-root", str(self.root / "sessions"),
             "--output-dir", str(self.root / "out"), *extra],
            capture_output=True, text=True, stdin=subprocess.DEVNULL,
        ).returncode

    def test_bad_archive_returns_error_code(self):
        archive = self._zip("junk.zip", {"face.png": "x"})
        self.assertEqual(self._run(archive, "--yes"), EXIT_ERROR)

    def test_failed_validation_in_non_interactive_returns_validation_code(self):
        meta = _meta(completed=False, jpeg_headroom_q75=0.05,
                     counters={"frames_read": 10, "frames_without_face": 9,
                               "samples_clean": 5, "samples_adversarial": 1})
        archive = self._zip("bad.zip", {
            "probe.csv": "x\n", "session.json": json.dumps(meta),
        })
        self.assertEqual(self._run(archive), EXIT_VALIDATION)
