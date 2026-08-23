import unittest

from src.forensics.privacy import artifact_reference, pseudonym, sanitize_identity_and_paths
from src.forensics.summarize_defense_handoff import build_overview, summarize, validate


def defense_row(defense: str, basis: str, sample_id: str = "vf_demo", **overrides):
    row = {
        "sample_id": sample_id,
        "defense": defense,
        "evaluation_basis": basis,
        "attack_success_before_defense": "true",
        "defense_bypassed": "false",
        "defense_success": "true",
    }
    row.update(overrides)
    return row


class ForensicsPrivacyTest(unittest.TestCase):
    def test_pseudonyms_and_artifact_references_are_deterministic_and_idempotent(self):
        self.assertEqual(pseudonym("Tony_Blair", "target"), pseudonym("Tony_Blair", "target"))
        self.assertEqual(
            pseudonym(pseudonym("Tony_Blair", "target"), "target"),
            pseudonym("Tony_Blair", "target"),
        )
        self.assertEqual(
            artifact_reference(artifact_reference("/private/data/Tony_Blair.jpg")),
            artifact_reference("/private/data/Tony_Blair.jpg"),
        )

    def test_publishable_row_contains_no_direct_identity_or_path(self):
        sanitized = sanitize_identity_and_paths(
            {
                "source_identity": "John_Ashcroft",
                "target_identity": "Tony_Blair",
                "adv_file": "/Users/example/Tony_Blair.jpg",
            }
        )
        self.assertNotIn("Ashcroft", sanitized["source_identity"])
        self.assertNotIn("Blair", sanitized["target_identity"])
        self.assertNotIn("/", sanitized["adv_file"])


class DefenseDenominatorTest(unittest.TestCase):
    def test_preprocessing_and_training_keep_separate_bases_and_denominators(self):
        rows = [
            defense_row("jpeg", "artifact_reload", defense_bypassed="true", defense_success="false"),
            defense_row("jpeg", "artifact_reload", sample_id="vf_ineligible", attack_success_before_defense="false"),
            defense_row("adv_training", "legacy_record"),
        ]
        preprocessing = summarize(rows, {"jpeg"})
        training = summarize(rows, {"adv_training"})
        overview = build_overview(rows, preprocessing, training)
        self.assertEqual(preprocessing[0]["eligible_attacks"], 1)
        self.assertEqual(training[0]["eligible_attacks"], 1)
        self.assertIsNone(overview["combined_bypass_rate"])
        self.assertEqual(
            overview["evaluation_groups"]["preprocessing"]["evaluation_basis"],
            "artifact_reload",
        )
        self.assertEqual(
            overview["evaluation_groups"]["adversarial_training"]["evaluation_basis"],
            "legacy_record",
        )

    def test_validate_rejects_mixed_evaluation_basis(self):
        rows = [
            defense_row("jpeg", "legacy_record"),
            defense_row("smoothing", "artifact_reload"),
            defense_row("bitdepth", "artifact_reload"),
            defense_row("adv_training", "legacy_record"),
        ]
        with self.assertRaisesRegex(ValueError, "evaluation_basis"):
            validate(rows)

    def test_zero_eligible_denominator_is_undefined(self):
        rows = [
            defense_row("jpeg", "artifact_reload", attack_success_before_defense="false"),
        ]
        preprocessing = summarize(rows, {"jpeg"})
        overview = build_overview(rows, preprocessing, [])
        self.assertIsNone(
            overview["evaluation_groups"]["preprocessing"]["defenses"]["jpeg"]["defense_bypass_rate"]
        )


if __name__ == "__main__":
    unittest.main()
