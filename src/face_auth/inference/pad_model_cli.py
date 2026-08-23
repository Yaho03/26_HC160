from __future__ import annotations

import argparse
import json

from src.face_auth.inference.pad_model_registry import (
    load_pad_model_artifact,
    verify_pad_model_artifact,
)


def main() -> int:
    args = _parser().parse_args()
    try:
        model = load_pad_model_artifact(args.registry)
        result = verify_pad_model_artifact(model, args.model)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a local PAD model against its source registry"
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--model", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
