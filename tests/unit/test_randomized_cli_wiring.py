"""랜덤화 CLI 배선.

ADR-003은 임계값을 model·preprocessing에 묶인 버전 artifact로 다루라고 한다. 변환
범위도 preprocessing의 일부이므로 프리셋 이름과 계열 목록이 인증 config 해시에
들어가야 한다. 해시에 없으면 같은 설정으로 기록된 두 결정이 실제로는 다른 변환을
쓴 것일 수 있다.
"""

import unittest

from src.face_auth.cli import _authentication_config_sha256, build_parser


def _args(*extra):
    return build_parser().parse_args([
        "authenticate", "--video", "v.mp4", "--template", "t.npz",
        "--threshold", "0.5", "--threshold-version", "v1", "--user-id", "u1",
        *extra,
    ])


class ParserTest(unittest.TestCase):
    def test_randomization_is_off_by_default(self):
        self.assertFalse(_args().adversarial_randomize)

    def test_flag_turns_randomization_on(self):
        self.assertTrue(_args("--adversarial-randomize").adversarial_randomize)

    def test_preset_defaults_to_narrow(self):
        self.assertEqual(_args().adversarial_range_preset, "narrow")

    def test_families_default_to_the_shipped_transforms(self):
        self.assertEqual(_args().adversarial_families, "jpeg,bit,blur")


class ConfigHashTest(unittest.TestCase):
    def test_randomization_flag_changes_the_hash(self):
        self.assertNotEqual(
            _authentication_config_sha256(_args()),
            _authentication_config_sha256(_args("--adversarial-randomize")),
        )

    def test_preset_changes_the_hash(self):
        self.assertNotEqual(
            _authentication_config_sha256(_args("--adversarial-randomize")),
            _authentication_config_sha256(
                "--adversarial-randomize", "--adversarial-range-preset", "wide"
            ) if False else _authentication_config_sha256(
                _args("--adversarial-randomize", "--adversarial-range-preset", "wide")
            ),
        )

    def test_families_change_the_hash(self):
        self.assertNotEqual(
            _authentication_config_sha256(_args("--adversarial-randomize")),
            _authentication_config_sha256(
                _args("--adversarial-randomize", "--adversarial-families", "blur")
            ),
        )

    def test_same_arguments_give_the_same_hash(self):
        self.assertEqual(
            _authentication_config_sha256(_args("--adversarial-randomize")),
            _authentication_config_sha256(_args("--adversarial-randomize")),
        )


if __name__ == "__main__":
    unittest.main()
