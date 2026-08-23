import json
import unittest
from datetime import datetime, timezone

from src.forensics.build_attack_forensics import (
    build_context,
    build_sessions,
    risk_level,
    rule_findings,
    score_from_hits,
    strongest_defense_state,
)


def attack_row(**overrides):
    row = {
        "sample_id": "vf_test",
        "pair_id": "pair-1",
        "attack": "targeted_zoo_facenet_verification",
        "source_name": "source-a",
        "target_name": "target-b",
        "threshold": "0.50",
        "similarity_before": "0.20",
        "similarity_after_attack": "0.51",
        "accepted_before": "false",
        "accepted_after_attack": "true",
        "attack_success_before_defense": "true",
        "queries_used": "120",
        "l2": "0.0",
        "linf": "0.0",
    }
    row.update(overrides)
    return row


class AttackForensicsTest(unittest.TestCase):
    def test_rule_reasons_are_specific_and_zero_norm_is_not_missing(self):
        rows = [attack_row(), attack_row(sample_id="vf_test_2")]
        findings = rule_findings(
            rows[0],
            strongest_defense_state([]),
            build_context(rows),
            threshold_margin=0.01,
        )
        by_id = {finding["rule_id"]: finding["reason"] for finding in findings}
        self.assertIn("FA-R002", by_id)
        self.assertIn("attempts=2", by_id["FA-R002"])
        self.assertIn("FA-R003", by_id)
        self.assertIn("queries_used=120", by_id["FA-R003"])
        self.assertIn("FA-R008", by_id)
        self.assertIn("l2=0.0", by_id["FA-R008"])

    def test_borderline_rule_requires_repeated_attempts(self):
        row = attack_row()
        findings = rule_findings(
            row,
            strongest_defense_state([]),
            build_context([row]),
            threshold_margin=0.01,
        )
        self.assertNotIn("FA-R002", {finding["rule_id"] for finding in findings})

    def test_sessions_keep_authentication_and_risk_separate(self):
        row = attack_row(
            accepted_after_attack="false",
            attack_success_before_defense="false",
            similarity_before="0.35",
            similarity_after_attack="0.49",
        )
        sessions = build_sessions([row], {}, datetime(2026, 1, 1, tzinfo=timezone.utc))
        self.assertFalse(sessions[0]["accepted_after_attack"])
        self.assertGreater(sessions[0]["risk_score"], 0)
        reasons = json.loads(sessions[0]["rule_reasons"])
        self.assertTrue(all(set(item) == {"rule_id", "reason"} for item in reasons))

    def test_missing_required_numeric_field_fails_closed(self):
        row = attack_row(threshold="")
        with self.assertRaisesRegex(ValueError, "threshold"):
            build_sessions([row], {}, datetime(2026, 1, 1, tzinfo=timezone.utc))

    def test_risk_score_is_deterministic_and_bounded(self):
        hits = [f"FA-R00{number}" for number in range(1, 9)]
        first = score_from_hits(hits, True, True, True)
        second = score_from_hits(hits, True, True, True)
        self.assertEqual(first, second)
        self.assertEqual(first, 100)
        self.assertEqual(risk_level(first), "critical")


if __name__ == "__main__":
    unittest.main()
