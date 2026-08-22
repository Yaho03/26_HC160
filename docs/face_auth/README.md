# Face Authentication Documentation

## Purpose

`src/face_auth/` is a local, session-based reference prototype for evaluating face-authentication security controls. It is not a banking backend, mobile SDK, or production biometric product.

## Runtime flow

```text
separate enrollment capture
  -> local multi-frame template

session creation
  -> nonce + randomized challenge
  -> webcam or recorded-video capture
  -> frame integrity + quality + all-face detection
  -> identity verification
  -> FULL-only motion/replay/PAD/liveness/continuity gates
  -> fail-closed policy decision
  -> context-bound one-time result token
  -> consume or expire
```

## Module map

| Path | Responsibility |
|---|---|
| `domain/types.py` | Session, gate, decision, frame, manifest, and token value types. |
| `domain/state_machine.py` | Allowed session transitions and terminal-state protection. |
| `domain/policy.py` | BASELINE_ONLY/FULL required gates and failure precedence. |
| `application/session_service.py` | Session and challenge lifecycle. |
| `application/evaluation_service.py` | Gate aggregation, decision, and token issuance. |
| `application/token_service.py` | Purpose/context-bound, expiring, one-time token behavior. |
| `application/enrollment_service.py` | Separate multi-frame template creation and local NPZ storage. |
| `application/evidence_service.py` | Nonce-bound ordered-frame evidence digest. |
| `adapters/capture_base.py` | Common frame-source and bounded latest-frame buffer contract. |
| `adapters/opencv_capture.py` | Recorded-video and webcam input. |
| `inference/pipeline.py` | Baseline integrity, quality, all-face, and identity evidence. |
| `inference/full_pipeline.py` | FULL-profile gate composition. |
| `inference/pad_adapter.py` | Explicit TorchScript PAD model adapter; no heuristic pass fallback. |
| `inference/active_liveness.py` | Post-challenge head-turn or blink evidence. |
| `inference/continuity.py` | Template-anchored multi-frame identity consistency. |
| `inference/content_replay.py` | Frozen/repeated-content signal. |
| `inference/camera_motion.py` | Background global-motion quality signal. |
| `inference/adversarial_detector.py` | Transform-consistency optional veto. |
| `evaluation/calibration.py` | Prototype gate-threshold calibration from validation CSV. |
| `evaluation/pad_manifest.py` | Opaque, relative-path PAD video manifest validation. |
| `evaluation/pad_evaluator.py` | Per-video face/quality/PAD evaluation with explicit excluded outcomes. |
| `evaluation/pad_metrics.py` | APCER, BPCER, ACER, attack-species metrics and Wilson intervals. |
| `evaluation/pad_cli.py` | Reproducible labeled-video PAD report command. |
| `cli.py` | Separate `enroll` and `authenticate` commands. |

Attack-video generation is separate under `src/attack_scenarios/`.

## Security profiles

| Profile | Required gates | Appropriate use |
|---|---|---|
| `BASELINE_ONLY` | frame integrity, quality, single face, identity | Development, integration, and recorded-video baseline only. |
| `FULL` | baseline plus camera motion, content replay, passive PAD, active liveness, continuity | Reference security composition after every model and threshold is validated. |

The optional adversarial gate can veto either profile when configured. A missing or errored required gate cannot produce `VERIFIED`.

## Quick commands

```bash
python -m src.face_auth.cli enroll --help
python -m src.face_auth.cli authenticate --help
python -m unittest discover -s tests/unit -v
python -m unittest discover -s tests/integration -v
```

Complete examples and troubleshooting are in `../14_LOCAL_RUNBOOK.md`.

## Documentation map

- `API_CONTRACT.md` — state, gate, evidence, and token contract.
- `THREAT_MODEL.md` — protected assets, in-scope threats, and trust-boundary limits.
- `IMPLEMENTATION_PLAN.md` — implemented structure and remaining build work.
- `BACKLOG.md` — requirement-level work and current status.
- `EXPERIMENT_PLAN.md` — split rules, evaluation groups, and metrics.
- `SMOKE_TEST_REPORT_2026-08-22.md` — limited recorded-video evidence and its interpretation.
- `../13_IMPLEMENTATION_STATUS.md` — repository-wide implementation/evidence matrix.

## Non-negotiable limitations

- Local NPZ templates are not encrypted.
- The in-memory session/token store is not a durable transactional database.
- A Python camera process cannot attest the operating system, driver, or virtual camera.
- The repository contains no validated PAD checkpoint.
- Example thresholds are not release thresholds.
- Synthetic replay success does not establish physical print/screen/deepfake performance.
- `VERIFIED` is an authentication result, not authorization to complete a financial transaction.
