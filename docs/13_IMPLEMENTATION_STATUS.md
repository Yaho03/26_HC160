# HC160 Implementation Status

## 1. Snapshot

This document describes the repository at commit `a5fe5ad` on branch `codex/realtime-face-auth-v2`.

- Research foundation and artifact contracts: implemented.
- Privacy-conscious dataset manifest workflow: implemented.
- Verification threshold calibration and clean-report workflow: implemented at the score-record layer.
- Session-based face-authentication reference prototype: implemented.
- Physical PAD validation, production persistence, mobile attestation, and release claims: not complete.

Verification performed on 2026-08-22:

- Python 3.9 full suite: 144 tests passed.
- Python 3.9 research suite: 37 tests passed.
- Python 3.13 research suite: 37 tests passed.
- Recorded-video FaceNet smoke test: completed; see `face_auth/SMOKE_TEST_REPORT_2026-08-22.md`.
- PAD report wiring smoke: calibration/test reports were generated with a constant-output test model; its print APCER was intentionally exposed as `1.0`, which validates reporting behavior rather than PAD accuracy.
- Physical capture readiness: not started. Consented subjects, print media, display-replay devices, and held-out PAD evidence are not prepared yet; this work remains tracked as a separate follow-up experiment.

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
| Verification metrics | Implemented | `src/evaluation/verification_metrics.py` | Apply to approved model-produced pair scores from a frozen dataset. |
| FaceNet score export | Implemented; experiment pending | Explicit checkpoint loader, frozen preprocessing contract, pair/image validation, score/export schemas and provenance sidecar | Run on the approved identity-disjoint calibration/test pair manifests. |
| Verification calibration | Implemented | `verification_calibration.py`, provenance-verifying `verification_baseline_cli.py` | Generate final EXP-VER-001 threshold and clean report from approved exports. |
| Threshold provenance | Implemented | Threshold and clean-report schemas now bind model, preprocessing and score-export SHA-256 values | Register final outputs in the run/artifact manifest workflow. |
| Session state and policy | Implemented | `src/face_auth/domain/`, unit and integration tests | Persistent adapter and concurrency/transaction testing. |
| Enrollment separation | Implemented | `enrollment_service.py`, separate `enroll` CLI | Encrypt templates and add authorized registration/revocation. |
| Evidence binding | Implemented locally | nonce-bound capture manifest and digest | Signed or attested evidence from a separate client trust boundary. |
| Baseline video authentication | Smoke-tested | MTCNN, FaceNet, quality and identity pipeline | Calibrate identity and quality thresholds on target devices. |
| Camera input | Implemented | shared OpenCV `FrameSource` contract | Device matrix, long-run capture, drop and latency experiments. |
| Live camera interaction | Implemented | memory-only preview, progress overlay, user cancellation, headless override, and structured device/preview errors | macOS permission grant and target-camera manual smoke test. |
| Repeated-content detection | Smoke-tested | codec-tolerant content replay gate | Genuine/attack false-positive study across codecs and cameras. |
| Camera-motion gate | Implemented | background motion estimator | Target-camera calibration; currently retry-oriented evidence only. |
| Passive PAD | Capture/evaluation harness implemented; external artifact blocked | TorchScript/ONNX adapters, fail-closed model registry, physical capture CLI/protocol, manifest validator, source-bound evaluator, APCER/BPCER and attack-species metrics exist | Approved model, license/checksum, authorized capture sessions, and held-out physical evaluation. |
| PAD report provenance | Implemented | Run ID, Git state, manifest/model/source-video SHA-256, byte counts, immutable-by-default output, and `pad-evaluation-report.schema.json` | Register each completed report in the repository-wide run/artifact manifest workflow. |
| Active liveness | Implemented | randomized challenge, live instruction overlay, displayed-frame boundary binding, and head-turn/blink logic | Physical replay study, accessibility alternatives, threshold calibration. |
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
2. Build frozen calibration/test pair manifests and export FaceNet score JSONL plus provenance sidecars with the pinned checkpoint.
3. Run EXP-VER-001 and freeze the identity threshold artifact.
4. Acquire a license-compatible PAD checkpoint and record its checksum and preprocessing contract.
5. Capture held-out genuine, print, screen, replay, insertion, and person-switch sessions.
6. Calibrate quality, PAD, liveness, continuity, replay, motion, and adversarial thresholds on validation only.
7. Report FMR/FNMR, APCER/BPCER, attack-species results, clean cost, latency, errors, and confidence intervals.

## 7. Integration gaps found during documentation review

These are not reasons to discard the implementation. They are the shortest path from a working prototype to defensible experiment artifacts.

| Priority | Gap | Required follow-up |
|---|---|---|
| P0 before a FULL claim | No approved PAD checkpoint or held-out physical dataset is present. | Acquire, license-check, hash, calibrate, and evaluate the model before enabling a reportable FULL profile. |
| P1 | GitHub CI runs dependency-free research tests only. | Add a separate pinned face-auth CI job or documented heavyweight validation workflow. |
| P1 | PAD reports have a JSON Schema and run/source provenance, but face-auth decisions and repository-wide run/artifact registration remain incomplete. | Define the decision schema and register all emitted reports in run manifests and artifact references. |
| P1 | `requirements-face-auth.txt` is version-pinned but not hash-locked; target is Python 3.11 while the complete local run was Python 3.9. | Validate a clean Python 3.11 environment and publish a locked environment artifact. |
| P1 | Templates, sessions, and tokens use local NPZ/in-memory adapters. | Add encrypted, transactional persistence before any multi-process or remote-service use. |

PAD reports no longer serialize local manifest or model paths. Formal runs refuse a dirty Git worktree and an existing output path by default; `--allow-dirty` and `--overwrite` are explicit non-default escape hatches for local debugging only.
