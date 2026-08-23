import json
import tempfile
import unittest
from pathlib import Path

from src.attack_scenarios.manifest import InsertVideoSpec, load_manifest


class ScenarioManifestTest(unittest.TestCase):
    def test_relative_insert_video_manifest_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.json"
            path.write_text(
                json.dumps(
                    {
                        "scenario_id": "ATK-05-test",
                        "base_video": "base.mp4",
                        "output_video": "out.mp4",
                        "events": [
                            {
                                "type": "insert_video",
                                "at_index": 3,
                                "path": "attack.mp4",
                                "max_frames": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = load_manifest(path)
            self.assertEqual(manifest.scenario_id, "ATK-05-test")
            self.assertIsInstance(manifest.events[0], InsertVideoSpec)
            self.assertEqual(manifest.events[0].max_frames, 2)

    def test_absolute_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.json"
            path.write_text(
                json.dumps(
                    {
                        "scenario_id": "bad",
                        "base_video": "/tmp/base.mp4",
                        "output_video": "out.mp4",
                        "events": [
                            {
                                "type": "repeat_frame",
                                "source_index": 0,
                                "at_index": 1,
                                "count": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
