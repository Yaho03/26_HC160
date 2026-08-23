# Face Authentication Backlog

Issues remain as reviewable specifications until approved for creation in GitHub.

## Status legend

- **Implemented**: code and automated contract tests exist.
- **Partial**: a code path exists, but an external artifact, target-device calibration, or required experiment is missing.
- **Pending**: no complete implementation/evidence exists yet.

## Implementation snapshot (commit `c4c1e15`)

| ID | Status | Evidence or remaining blocker |
|---|---|---|
| DOC-001 | Implemented | Scope, API contract, threat model, module map, and limitations documented. |
| EVAL-001 | Partial | Legacy limitations are documented; old claims have not been rerun on corrected held-out data. |
| DEP-001 | Partial | Python packages have a clean CI lock; approved external model artifacts and their distribution checksums remain. |
| DEP-002 | Implemented | The linux/amd64 Python 3.11 CI resolves all transitive face-auth packages from a SHA-256 lock and rejects direct-pin drift. |
| EXP-001 | Partial | Dataset manifest and leakage validator exist; final subject/session/device-disjoint dataset is not built. |
| ARCH-001 | Implemented | State, gate, policy, and fail-closed integration tests pass. |
| SEC-001 | Partial | Enrollment separation and token lifecycle exist; encrypted persistent template storage does not. |
| ATK-001 | Implemented | Manifest-driven replay/insertion runner and tests exist. |
| CI-001 | Implemented | Python 3.11 workflow installs pinned face-auth dependencies and runs unit/integration suites on relevant changes. |
| FR-101 | Implemented | Nonce, challenge, state, expiry, and replay behavior are tested. |
| FR-102 | Implemented | Webcam/video share `FrameSource`; bounded latest-frame buffering and drops are tested. |
| FR-103 | Implemented | Quality failure is retryable and distinct from model errors. |
| FR-104 | Implemented | MTCNN retains all detections and multi-face evidence fails closed. |
| FR-105 | Partial | Multi-frame template/verification plus provenance-bound FaceNet score export and threshold calibration exist; final approved identity threshold artifact is pending. |
| FR-106 | Implemented | Required failures/errors/NOT_EVALUATED cannot verify. |
| FR-107 | Implemented | Token context, expiry, replay, and one-time consume are tested. |
| FR-108 | Implemented | Webcam preview, capture progress, user cancellation, headless override, and structured camera/preview failures are tested. |
| EXP-002 | Implemented for synthetic tests | Genuine/impostor/multi-face/quality/error cases are automated; target-device evidence remains. |
| FR-201 | Partial | TorchScript/ONNX PAD adapters, source-bound report schema, immutable output, manifest validator, video evaluator, APCER/BPCER and species metrics exist; validated checkpoint and held-out physical data are missing. |
| FR-202 | Partial | Random challenge logic, live instruction display, and post-display frame binding exist; physical replay/accessibility evaluation is pending. |
| FR-203 | Partial | Tracking and template continuity exist; held-out switch/occlusion study is pending. |
| FR-204 | Implemented | FULL capture incrementally detects repeated/frozen content, binds the captured prefix, and terminates with a no-token security veto. |
| FR-205 | Implemented | Evidence-bound terminal decisions use one privacy-minimized schema, immutable-by-default output, and a run-manifest-compatible artifact reference. |
| EXP-201 | Pending | Physical print/screen/replay dataset and report are absent. |
| EXP-202 | Partial | Synthetic insertion scenario passes; physical/person-switch matrix and delay report are absent. |
| FR-301 | Partial | Optional transform-consistency veto exists; clean/adaptive calibration is pending. |
| FR-302 | Pending | Correct held-out adversarial-training rerun is not implemented. |
| EXP-301 | Pending | Digital/screen/print transfer comparison is absent. |
| PERF-001 | Pending | Named-hardware FPS/drop/P95 report is absent. |
| SEC-301 | Pending | Mobile attestation provider boundary is documented but not implemented. |

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
| FR-108 | Live camera capture UX | preview/progress, cancellation, headless mode, and camera failure behavior tested |
| EXP-002 | Baseline E2E scenarios | genuine/impostor/multi-face/blur automated |

## Phase 2

| ID | Work | Acceptance criteria |
|---|---|---|
| FR-201 | Passive PAD and calibration | APCER/BPCER by print/screen species |
| FR-202 | Random active liveness | pre-challenge action and replay rejected |
| FR-203 | Tracking and identity continuity | occlusion reacquisition and person switch tested |
| FR-204 | Streaming replay veto | first threshold violation stops capture and denies without downstream model execution |
| FR-205 | Authentication decision artifact | normal and streaming-veto outcomes share one schema and bind policy/gate versions to evidence without biometric payloads |
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
