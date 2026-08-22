# Financial Face Authentication Reference Prototype

## Scope

This repository implements a contest-grade reference prototype, not a production banking system. The prototype demonstrates:

- separate enrollment and authentication flows;
- session-issued nonce and randomized liveness challenge;
- webcam and recorded-video evidence through one pipeline;
- quality, single-face, identity, PAD, liveness, continuity, and adversarial gates;
- explicit `VERIFIED`, `RETRYABLE`, `SECURITY_DENIED`, and `ERROR` outcomes;
- a short-lived result token bound to a purpose/context and consumable once;
- reproducible normal and attack scenarios.

Mobile SDKs, hardware-backed attestation, KMS/HSM, and production deployment are extension work.

## Security Profiles

| Profile | Required gates | Meaning |
|---|---|---|
| `BASELINE_ONLY` | frame integrity, quality, single face, identity | Development vertical slice only; must not be described as complete financial authentication |
| `FULL` | baseline + camera motion + content replay + passive PAD + active liveness + continuity | Target demonstration profile |

An evaluated optional gate may veto either profile. A required `ERROR` or `NOT_EVALUATED` gate cannot produce `VERIFIED`.

## Build Order

1. Revalidate existing metrics and data provenance.
2. Implement and test domain state, policy, session, and one-time token rules.
3. Run the baseline pipeline against recorded videos.
4. Connect the webcam through the same `FrameSource` contract.
5. Add calibrated PAD, challenge liveness, and identity continuity.
6. Add transform-consistency adversarial inspection as a secondary trigger.

## Current Implementation

Implemented:

- session state machine;
- stable gate statuses and decision actions;
- baseline and full security-profile policy;
- challenge/nonce issuance;
- purpose/context-bound one-time token;
- fail-closed handling for missing and errored required gates;
- separate multi-frame enrollment and template storage;
- webcam and recorded-video capture adapters;
- MTCNN all-face detection and FaceNet multi-frame verification;
- camera-motion and repeated-content detection;
- TorchScript PAD adapter with no heuristic fallback;
- randomized head-turn liveness from five landmarks;
- template-anchored identity continuity;
- optional transform-consistency adversarial inspection;
- deterministic insertion/replay scenario manifests;
- validation-only threshold calibration;
- tests for state, policy, inference gates, expiry, context mismatch, and token replay.

Next:

- obtain and validate a PAD model on the target webcam/phone cameras;
- collect subject/session/device-disjoint genuine, print, screen, and replay evidence;
- calibrate every hard threshold on validation data;
- automate APCER/BPCER, FMR/FNMR, insertion detection, and latency reports;
- add encrypted template storage and a persistent session/token adapter.
