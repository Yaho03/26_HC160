# HC160 Implementation Status

## 1. Snapshot

This document describes the repository at commit `d1b96ba` on branch `codex/realtime-face-auth-v2`.

- Research foundation and artifact contracts: implemented.
- Privacy-conscious dataset manifest workflow: implemented.
- Verification threshold calibration and clean-report workflow: implemented at the score-record layer.
- Session-based face-authentication reference prototype: implemented.
- Physical PAD validation, production persistence, mobile attestation, and release claims: not complete.

Verification performed on 2026-08-22:

- Python 3.9 full suite: 113 tests passed.
- Python 3.9 research suite: 35 tests passed.
- Python 3.13 research suite: 35 tests passed.
- Recorded-video FaceNet smoke test: completed; see `face_auth/SMOKE_TEST_REPORT_2026-08-22.md`.
- PAD report wiring smoke: calibration/test reports were generated with a constant-output test model; its print APCER was intentionally exposed as `1.0`, which validates reporting behavior rather than PAD accuracy.

These counts identify the snapshot only. Use the current test command rather than treating the numbers as permanent.

## 2. Meaning of status labels

| Label | Meaning |
|---|---|
| Implemented | Code path and automated contract tests exist. |
| Smoke-tested | A limited local execution was completed on named inputs. |
| Experiment pending | Code exists, but required disjoint data and final metrics do not. |
| External artifact blocked | A model, dataset, or device evidence is not present in the repository. |
| Production extension | Intentionally outside this contest-grade local prototype. |

`Implemented` does not mean the security claim is validated. A feature becomes reportable only after its experiment protocol, data provenance, thresholds, sample counts, confidence intervals, and limitations are recorded.

## 3. Current implementation matrix

| Area | Status | Implementation evidence | Remaining evidence or work |
|---|---|---|---|
| Dataset snapshot | Implemented | `src/datasets/manifest.py`, `manifest_cli.py`, dataset schemas | Build and publish an approved LFW snapshot manifest without raw identities. |
| Split leakage checks | Implemented | media-hash and optional identity-disjoint checks | Run on the final subject/session/device split. |
| Verification metrics | Implemented | `src/evaluation/verification_metrics.py` | Apply to model-produced pair scores from a frozen dataset. |
| Verification calibration | Implemented | `verification_calibration.py`, `verification_baseline_cli.py` | Generate real FaceNet calibration/test score manifests and EXP-VER-001 outputs. |
| Threshold provenance | Implemented | threshold and clean-report schemas | Record actual checkpoint and preprocessing artifact hashes. |
| Session state and policy | Implemented | `src/face_auth/domain/`, unit and integration tests | Persistent adapter and concurrency/transaction testing. |
| Enrollment separation | Implemented | `enrollment_service.py`, separate `enroll` CLI | Encrypt templates and add authorized registration/revocation. |
| Evidence binding | Implemented locally | nonce-bound capture manifest and digest | Signed or attested evidence from a separate client trust boundary. |
| Baseline video authentication | Smoke-tested | MTCNN, FaceNet, quality and identity pipeline | Calibrate identity and quality thresholds on target devices. |
| Camera input | Implemented | shared OpenCV `FrameSource` contract | Device matrix, long-run capture, drop and latency experiments. |
| Repeated-content detection | Smoke-tested | codec-tolerant content replay gate | Genuine/attack false-positive study across codecs and cameras. |
| Camera-motion gate | Implemented | background motion estimator | Target-camera calibration; currently retry-oriented evidence only. |
| Passive PAD | Evaluation harness implemented; external artifact blocked | TorchScript adapter, manifest validator, video evaluator, APCER/BPCER and attack-species metrics exist | Approved model, license/checksum, and held-out physical evaluation. |
| Active liveness | Implemented | randomized challenge and head-turn/blink logic | Physical replay study, accessibility alternatives, threshold calibration. |
| Identity continuity | Implemented | template-anchored temporal gate | Person-switch and occlusion evaluation on held-out sessions. |
| Adversarial inspection | Implemented as optional veto | transform-consistency and feature-squeeze modules | Clean calibration, adaptive attack evaluation, latency cost. |
| Scenario generation | Implemented | manifest-driven insertion/replay builder | Broader scenario catalog and real physical attack capture. |
| Legacy adversarial training | Historical only | existing defense outputs and scripts | Correct disjoint train/validation/test rerun plus adaptive attack. |
| UI or service API | Not implemented | local CLI only | Read-only demo/API may be added after validated artifacts exist. |
| Secure persistence | Production extension | in-memory store and local NPZ only | Database transactions, encryption, KMS/HSM, retention and audit. |
| Mobile/device attestation | Production extension | trust boundary documented | iOS/Android client, server verification and hardware-backed keys. |

## 4. Two calibration workflows that must not be confused

The repository intentionally has two calibration layers:

| Path | Purpose | Input | Output |
|---|---|---|---|
| `src/evaluation/verification_calibration.py` | Research verification operating threshold and clean baseline | Pair-level cosine scores from disjoint calibration/test manifests | Versioned threshold artifact and EXP-VER-001 clean report |
| `src/face_auth/evaluation/calibration.py` | Prototype gate thresholds such as PAD, motion, replay, and adversarial checks | Validation CSV containing clean/attack gate values | Local threshold bundle for the face-auth CLI |

The second workflow does not replace the first. A face-auth configuration should reference a research-validated identity threshold plus separately validated gate thresholds. Neither workflow may tune on test rows.

## 5. Current claims boundary

The repository can currently demonstrate:

- deterministic state, policy, token, evidence, and artifact contracts;
- a recorded-video baseline authentication vertical slice;
- fail-closed composition of FULL-profile gates;
- reproducible synthetic replay/insertion scenarios;
- model-independent verification calibration without test leakage.

It cannot currently claim:

- production financial authentication readiness;
- a validated PAD accuracy or universal replay detection rate;
- robustness to compromised camera drivers, virtual cameras, or rooted devices;
- fairness or demographic parity;
- certified adversarial robustness;
- the legacy defense percentages as held-out verification security.

## 6. Recommended next evidence

1. Build approved subject/session/device-disjoint manifests with EXP-DATA-001.
2. Generate FaceNet calibration and test score JSONL files with pinned model and preprocessing artifacts.
3. Run EXP-VER-001 and freeze the identity threshold artifact.
4. Acquire a license-compatible PAD checkpoint and record its checksum and preprocessing contract.
5. Capture held-out genuine, print, screen, replay, insertion, and person-switch sessions.
6. Calibrate quality, PAD, liveness, continuity, replay, motion, and adversarial thresholds on validation only.
7. Report FMR/FNMR, APCER/BPCER, attack-species results, clean cost, latency, errors, and confidence intervals.

## 7. Integration gaps found during documentation review

These are not reasons to discard the implementation. They are the shortest path from a working prototype to defensible experiment artifacts.

| Priority | Gap | Required follow-up |
|---|---|---|
| P0 before a PAD result | The PAD evaluator records model SHA-256 but not the manifest hash or every source-video hash in its report. | Bind the PAD manifest and input video artifacts by SHA-256 and reference a run manifest. |
| P0 before a PAD result | The PAD CLI currently writes its output path directly and can replace an existing report. | Add refusal-by-default or immutable run-ID output directories. Until then, always choose a new output path. |
| P0 before a FULL claim | No approved PAD checkpoint or held-out physical dataset is present. | Acquire, license-check, hash, calibrate, and evaluate the model before enabling a reportable FULL profile. |
| P0 before verification results | Score generation from the pinned embedding model is not yet connected to the new EXP-VER-001 JSONL workflow. | Add a model-specific score exporter with model/preprocessing artifact hashes. |
| P1 | GitHub CI runs dependency-free research tests only. | Add a separate pinned face-auth CI job or documented heavyweight validation workflow. |
| P1 | Face-auth CLI decisions and PAD reports do not yet have complete JSON Schemas/run-manifest integration. | Define output schemas and register all emitted artifacts. |
| P1 | `requirements-face-auth.txt` is version-pinned but not hash-locked; target is Python 3.11 while the complete local run was Python 3.9. | Validate a clean Python 3.11 environment and publish a locked environment artifact. |
| P1 | Templates, sessions, and tokens use local NPZ/in-memory adapters. | Add encrypted, transactional persistence before any multi-process or remote-service use. |

For publishable PAD runs, pass a repository-relative manifest path so the current report does not disclose a local absolute path.
