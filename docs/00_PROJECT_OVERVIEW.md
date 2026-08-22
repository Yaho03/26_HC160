# HC160 Project Overview

## 1. Project identity

HC160 is a reproducible research and demonstration project for evaluating targeted adversarial impersonation attacks, layered defenses, and detection signals against face-verification systems in a financial-authentication scenario.

The repository is **not** a production banking authentication system and must not be presented as one. Its authoritative output is an experiment run with pinned data, model, threshold, configuration, and artifact metadata. A future UI may display completed runs, but it must not redefine research metrics.

## 2. Primary research question

Can a defense reduce targeted reject-to-accept attack success while preserving clean verification performance under a fixed, independently calibrated threshold?

Primary evaluation priorities:

1. targeted attack success rate among pairs rejected before attack;
2. FAR/FMR and FRR/FNMR on untouched clean test pairs;
3. clean TAR preservation after defense;
4. attack budget, query cost, and end-to-end latency;
5. reproducibility across seeds and environments.

## 3. Tracks

| Track | Status | Role |
|---|---|---|
| LFW-10 ResNet-50 classification | Legacy baseline | Preserves the first five-attack/four-defense experiments. Classification accuracy is not a financial-authentication metric. |
| ResNet feature verification | Bridge baseline | Demonstrates pair generation, cosine scoring, EER calibration, and targeted PGD using the trained classifier backbone. |
| FaceNet VGGFace2 verification | Primary verification candidate | Matches the committed verification-defense artifacts. Its missing batch attack-generation provenance must be restored before it becomes fully reproducible. |
| Real-time face-auth reference prototype | Development track | Defines session, challenge, gate, policy, and one-time result-token behavior. It is intentionally separate from research metric computation. |
| Generative purification | Future extension | Defensive extension only after clean and adversarial verification baselines are valid. |

## 4. Target users

- team members implementing and reproducing experiments;
- reviewers evaluating attack, defense, and detection evidence;
- demonstration operators showing already validated runs;
- security researchers studying limitations, not production customers.

## 5. In scope

- versioned dataset and pair manifests;
- classification baselines kept for historical comparison;
- face embeddings and verification threshold calibration;
- targeted white-box and black-box attacks;
- preprocessing, trained, ensemble, temporal, and detection defenses;
- clean-performance preservation and adaptive-attack evaluation;
- reproducible run manifests, reports, and tests;
- a bounded reference flow for session and token security.

## 6. Out of scope until separately approved

- production banking deployment or security certification;
- real customer biometric data;
- mobile device attestation, KMS/HSM, or production account systems;
- claims about population fairness without an appropriate dataset;
- generative face impersonation or deepfake creation;
- microservices, message brokers, or distributed infrastructure;
- automatic ingestion of unverified experiment results into a public dashboard.

## 7. Success definition

The project is complete as a research system when:

- every reported number resolves to a run ID and denominator;
- calibration and test sets are separate;
- attack and defense artifacts pass versioned contracts;
- trained defenses are evaluated on held-out data and adaptive attacks;
- every defense reports clean-performance degradation;
- a clean checkout can run CPU tests and a documented smoke experiment;
- security, privacy, licensing, and known limitations are disclosed.

## 8. Normative document order

When documents conflict, use this precedence:

1. `01_RESEARCH_REQUIREMENTS.md`
2. `04_DATA_AND_ARTIFACT_CONTRACT.md`
3. `05_FACE_VERIFICATION_SPEC.md`
4. `06_ATTACK_SPEC.md` and `07_DEFENSE_AND_DETECTION_SPEC.md`
5. `08_EXPERIMENT_PLAN.md` and `09_EVALUATION_METRICS.md`
6. historical handoff and progress documents
