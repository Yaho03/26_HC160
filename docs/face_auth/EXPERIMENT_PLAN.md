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
