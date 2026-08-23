import unittest

from src.face_auth.domain.state_machine import (
    InvalidStateTransition,
    can_transition,
    transition,
)
from src.face_auth.domain.types import SessionState


class StateMachineTest(unittest.TestCase):
    def test_happy_path(self):
        path = [
            SessionState.CREATED,
            SessionState.CHALLENGE_ISSUED,
            SessionState.CAPTURING,
            SessionState.EVIDENCE_RECEIVED,
            SessionState.EVALUATING,
            SessionState.VERIFIED,
            SessionState.CONSUMED,
        ]
        current = path[0]
        for target in path[1:]:
            self.assertTrue(can_transition(current, target))
            current = transition(current, target)
        self.assertEqual(current, SessionState.CONSUMED)

    def test_terminal_state_cannot_be_reopened(self):
        with self.assertRaises(InvalidStateTransition):
            transition(SessionState.SECURITY_DENIED, SessionState.CHALLENGE_ISSUED)

    def test_verified_cannot_return_to_capture(self):
        with self.assertRaises(InvalidStateTransition):
            transition(SessionState.VERIFIED, SessionState.CAPTURING)

    def test_live_security_veto_can_deny_during_capture(self):
        self.assertTrue(
            can_transition(SessionState.CAPTURING, SessionState.SECURITY_DENIED)
        )
        self.assertEqual(
            transition(SessionState.CAPTURING, SessionState.SECURITY_DENIED),
            SessionState.SECURITY_DENIED,
        )


if __name__ == "__main__":
    unittest.main()
