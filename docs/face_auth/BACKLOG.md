# Face Authentication Backlog

Issues remain as reviewable specifications until approved for creation in GitHub.

## Phase 0

| ID | Work | Acceptance criteria |
|---|---|---|
| DOC-001 | Scope, trust boundary, threat model | P0/P1 and prototype claims agreed |
| EVAL-001 | Revalidate existing 100%/98.1% claims | leakage and clean false positives reported |
| DEP-001 | Reproducible environment | imports, versions, checksum/seed instructions recorded |
| EXP-001 | Disjoint dataset manifests | subject/session/device/attack split recorded |
| ARCH-001 | State and gate contracts | transition and fail-closed tests pass |
| SEC-001 | Enrollment/template/token lifecycle | registration separated; replay/context tests pass |
| ATK-001 | Scenario runner | deterministic video/insertion/protocol scenarios |

## Phase 1

| ID | Work | Acceptance criteria |
|---|---|---|
| FR-101 | Session, nonce, challenge, expiry | duplicate and expiry behavior tested |
| FR-102 | Webcam/video source and bounded queue | one pipeline and drop accounting |
| FR-103 | Quality gate | retryable quality failures separated from errors |
| FR-104 | Detection, alignment, multi-face gate | never silently selects one of multiple faces |
| FR-105 | Multi-frame enrollment and verification | validation threshold and versions recorded |
| FR-106 | Policy engine | required failures/errors cannot verify |
| FR-107 | One-time verification token | replay, context change, and expiry rejected |
| EXP-002 | Baseline E2E scenarios | genuine/impostor/multi-face/blur automated |

## Phase 2

| ID | Work | Acceptance criteria |
|---|---|---|
| FR-201 | Passive PAD and calibration | APCER/BPCER by print/screen species |
| FR-202 | Random active liveness | pre-challenge action and replay rejected |
| FR-203 | Tracking and identity continuity | occlusion reacquisition and person switch tested |
| EXP-201 | Print/screen/replay experiment | species-level results reported |
| EXP-202 | Mid-frame/person-switch experiment | insertion-length detection and delay reported |

## Phase 3

| ID | Work | Acceptance criteria |
|---|---|---|
| FR-301 | Transform-consistency secondary inspection | embedding dispersion and clean calibration |
| FR-302 | Correct adversarial-training split | holdout ASR, FAR, and FRR reported |
| EXP-301 | Digital/screen/print comparison | transfer results reported separately |
| PERF-001 | Runtime optimization | FPS, drops, and P95 latency reported |
| SEC-301 | Attestation provider contract | mock adapter and mobile extension boundary |
