import unittest

from src.face_auth.evaluation.pad_capture_cli import _parser, _validate_args


def required_arguments():
    return [
        "--artifact-root",
        "local_pad_dataset",
        "--manifest",
        "local_pad_dataset/manifest.csv",
        "--sample-id",
        "sample_00000001",
        "--label",
        "attack",
        "--attack-species",
        "print",
        "--subject-token",
        "subject_00000001",
        "--session-token",
        "session_00000001",
        "--device-token",
        "device_00000001",
        "--split",
        "calibration",
    ]


class PADCaptureCLIArgumentsTest(unittest.TestCase):
    def test_default_capture_is_long_enough_for_minimum_frames(self):
        args = _parser().parse_args(required_arguments())
        _validate_args(args)
        self.assertEqual(round(args.duration_seconds * args.fps), 75)

    def test_short_capture_is_rejected_before_camera_is_opened(self):
        args = _parser().parse_args(
            required_arguments()
            + ["--duration-seconds", "1", "--fps", "5", "--min-frames", "10"]
        )
        with self.assertRaisesRegex(SystemExit, "too short"):
            _validate_args(args)


if __name__ == "__main__":
    unittest.main()
