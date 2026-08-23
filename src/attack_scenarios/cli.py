from __future__ import annotations

import argparse
import json

from src.attack_scenarios.manifest import load_manifest
from src.attack_scenarios.video_builder import build_scenario


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic attack scenario video"
    )
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    frame_count, dimensions = build_scenario(manifest)
    print(
        json.dumps(
            {
                "scenario_id": manifest.scenario_id,
                "output_video": str(manifest.output_video),
                "frame_count": frame_count,
                "width": dimensions[0],
                "height": dimensions[1],
                "fps": manifest.fps,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
