import json
import unittest
from pathlib import Path

from src.contracts.validation import (
    ContractError,
    validate_attack_result,
    validate_defense_result,
    validate_relative_uri,
    validate_verification_pair,
)


class ContractValidationTest(unittest.TestCase):
    def test_all_schema_files_are_valid_json(self):
        schema_root = Path(__file__).resolve().parents[2] / "schemas"
        schemas = sorted(schema_root.glob("*.schema.json"))
        self.assertGreaterEqual(len(schemas), 6)
        for path in schemas:
            with self.subTest(path=path.name):
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(data["$schema"], "https://json-schema.org/draft/2020-12/schema")

    def test_absolute_and_parent_paths_are_rejected(self):
        for invalid in ("/content/data/a.png", "C:\\data\\a.png", "../secret.png"):
            with self.subTest(path=invalid), self.assertRaises(ContractError):
                validate_relative_uri(invalid)
        self.assertEqual(validate_relative_uri("artifacts/run-1/a.png"), "artifacts/run-1/a.png")

    def test_pair_cannot_repeat_same_sample(self):
        with self.assertRaises(ContractError):
            validate_verification_pair(
                {
                    "schema_version": "1.0",
                    "pair_id": "pair-1",
                    "protocol_id": "protocol-1",
                    "left_sample_id": "sample-1",
                    "right_sample_id": "sample-1",
                    "same_identity": True,
                    "split": "test",
                }
            )

    def test_attack_success_semantics_are_enforced(self):
        valid = {
            "accepted_before": False,
            "accepted_after": True,
            "success_from_reject": True,
            "elapsed_ms": 10.0,
        }
        validate_attack_result(valid)
        invalid = dict(valid, accepted_before=True)
        with self.assertRaises(ContractError):
            validate_attack_result(invalid)

    def test_clean_defense_row_cannot_claim_defense_success(self):
        row = {
            "input_kind": "clean",
            "attack_result_id": None,
            "clean_pair_id": "pair-1",
            "accepted_before": True,
            "accepted_after": True,
            "defense_success": True,
            "elapsed_ms": 1.0,
        }
        with self.assertRaises(ContractError):
            validate_defense_result(row)


if __name__ == "__main__":
    unittest.main()
