# Experiment Plan

## Split Rules

- subject-disjoint train/validation/test;
- enrollment and probe session disjoint;
- at least one holdout capture device when data permits;
- holdout attack family, epsilon, or physical medium;
- thresholds selected on validation only;
- no threshold changes after observing test results.

## Evaluation Groups

- genuine;
- zero-effort impostor;
- print;
- static screen;
- screen replay;
- mid-session insertion and identity switch;
- digital adversarial;
- screen-transferred adversarial;
- print-transferred adversarial;
- blur, low light, overexposure, shake, and face loss;
- frame reorder, duplicate requests, token replay, and context change.

## Metrics

| Area | Metrics |
|---|---|
| Identity | FMR/FNMR, FAR/FRR, EER, ROC/DET |
| PAD | APCER, BPCER, ACER per attack species |
| Adversarial | ASR before/after and clean FRR delta |
| Continuity | detection rate by insertion length and P50/P95/P99 delay |
| Protocol | replay, expiry, and context-change rejection |
| Runtime | capture/analysis FPS, frame drops, gate and session latency |

Every result must include sample counts and dataset, model, threshold, and policy versions.

## PAD evaluation harness

The current implementation adds a labeled-video PAD workflow under `src/face_auth/evaluation/`:

- manifest rows use opaque sample, subject, session, and device tokens;
- video references must be safe relative POSIX paths;
- bona-fide rows require `attack_species=none`;
- attack rows require a concrete print, screen, replay, or transferred-adversarial species;
- calibration and test are explicit split values;
- single-face and quality checks run before PAD scoring;
- multi-face, insufficient-valid-frame, camera, and model failures remain excluded outcomes rather than PAD passes or failures.

Primary definitions:

```text
APCER = evaluated attack samples classified as live / evaluated attack samples
BPCER = evaluated bona-fide samples classified as attack / evaluated bona-fide samples
ACER  = (APCER + BPCER) / 2, only when both rates are defined
```

Reports must show APCER by attack species, Wilson 95% intervals, and counts for `NOT_EVALUATED` and `ERROR`. A low APCER is not interpretable if a large fraction of attacks were excluded before PAD scoring.

The harness is implementation evidence only until it is run with a licensed, hashed checkpoint and subject/session/device-disjoint physical captures. Before an official run, add manifest and per-video content hashes to artifact provenance and use a unique output path because the current PAD CLI does not refuse overwrite.
