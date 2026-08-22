# Session and Token Contract

## State Flow

```text
CREATED -> CHALLENGE_ISSUED -> CAPTURING -> EVIDENCE_RECEIVED
        -> EVALUATING -> VERIFIED | RETRYABLE | SECURITY_DENIED | ERROR
        -> CONSUMED | EXPIRED
```

## Session Creation

Input:

```json
{
  "user_id": "user-1",
  "purpose": "HIGH_RISK_ACTION",
  "transaction_context_hash": "sha256-context",
  "security_profile": "FULL"
}
```

Output contains a server-generated `session_id`, nonce, challenge, policy version, and expiry. Challenge issuance must happen before accepted evidence capture.

## Gate Contract

Every gate returns:

- gate name;
- `PASS`, `FAIL`, `NOT_EVALUATED`, or `ERROR`;
- score and threshold when applicable;
- stable reason codes;
- model and threshold versions;
- latency in milliseconds.

## Verification Token

Only a `VERIFIED` session may issue a token. Consume must atomically verify:

- token has not been consumed;
- token and session are not expired;
- user ID matches;
- purpose matches;
- transaction context hash matches.

The token also carries the challenge nonce. A capture manifest binds the same nonce to ordered frame metadata and a SHA-256 digest of the captured frame bytes. In this local prototype the capture and verifier share one trust boundary; a production client must additionally sign or attest this evidence.

Any mismatch requires a new authentication session.

## FULL Profile Required Gates

- `frame_integrity`: frame IDs and monotonic capture times are ordered;
- `quality`: enough usable, well-lit, sufficiently sharp face frames exist;
- `single_face`: no frame silently selects one face from multiple faces;
- `identity`: validation-calibrated multi-frame similarity passes;
- `camera_motion`: global movement remains within the calibrated retry boundary;
- `content_replay`: repeated/frozen frame content is not observed;
- `passive_pad`: a target-device-calibrated PAD model passes;
- `active_liveness`: the post-challenge head movement is observed;
- `continuity`: recent embeddings remain bound to the enrollment template.

The optional `adversarial` gate may veto a decision when enabled. A missing PAD model is a configuration failure, not a pass.
