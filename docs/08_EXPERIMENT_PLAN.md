# HC160 Experiment Plan

## 1. Common run record

Every experiment records purpose, hypothesis, input manifests, split roles, model/checkpoint, threshold, attack/defense configuration, control variables, metrics, comparison criteria, outputs, reproduction command, and known limitations.

## 2. Experiment registry

| ID | Purpose and hypothesis | Inputs and controls | Primary outputs and comparison gate |
|---|---|---|---|
| EXP-DATA-001 | Freeze dataset provenance and prove split separation. | Dataset archive, manifest builder; no model. | Valid manifests, zero prohibited overlaps. |
| EXP-VER-001 | Establish clean verification baseline. | Calibration and untouched test pairs; frozen model/preprocessing. | FAR, FRR, EER, ROC-AUC, TAR@FAR, threshold artifact. |
| EXP-VER-002 | Explain classification vs verification behavior. | Same legacy LFW-10 source; separate metrics. | Comparative report without treating accuracy as verification. |
| EXP-ATK-001 | Measure white-box targeted vulnerability. | Reject-before test pairs; FGSM/PGD budgets. | Reject-to-accept ASR, norms, latency, CI. |
| EXP-ATK-002 | Measure black-box efficiency. | Same eligible pairs; fixed query budgets. | ASR-query curve, elapsed time, budget compliance. |
| EXP-DEF-001 | Compare preprocessing defenses. | Attack results plus clean pairs. | ASR reduction versus clean TAR/FRR delta. |
| EXP-DEF-002 | Evaluate trained defense generalization. | Disjoint attack train/validation/test rows. | Held-out and adaptive ASR, clean metrics. |
| EXP-DEF-003 | Evaluate ensemble value. | Fixed component outputs. | Improvement over strongest component and latency cost. |
| EXP-DET-001 | Evaluate squeezing detector. | Balanced clean/attack validation and test rows. | Detector ROC-AUC, TPR/FPR, verification impact. |
| EXP-TEMP-001 | Evaluate real temporal/replay behavior. | Genuine and attack video sessions. | Detection by attack species, false positives, delay. |
| EXP-GEN-001 | Evaluate defensive generative purification. | Held-out clean/attack pairs; fixed denoiser. | ASR, clean TAR, identity drift, p95 latency. |
| EXP-TRN-001 | Measure attack transferability. | Source and target embedding models. | Cross-model ASR matrix. |
| EXP-PERF-001 | Establish runtime budget. | Named CPU/GPU and batch sizes. | p50/p95 latency, throughput, memory. |
| EXP-REP-001 | Verify repeatability. | At least three seeds and a clean checkout. | Metric variance and artifact/run-manifest completeness. |

## 3. Mandatory controls

- Threshold is frozen before test.
- Eligible pair IDs are identical for attacks being compared.
- Defense comparisons use identical attack artifacts.
- Clean and adversarial preprocessing versions match.
- Failed loads and model errors are reported separately.
- Hardware, batch size, warm-up, and timing boundaries are recorded.
- Final reports include numerator, denominator, and confidence interval.

## 4. Traceability examples

| Requirement | Module target | Test target | Experiment | Artifact/report |
|---|---|---|---|---|
| DATA-001 | `src/contracts`, dataset builder | schema/hash test | EXP-DATA-001 | dataset manifest |
| VER-001 | `src/evaluation/verification_calibration.py` | `tests/research/test_verification_calibration.py` | EXP-VER-001 | threshold artifact + clean report |
| ATK-001 | centralized evaluation | transition test | EXP-ATK-001 | attack results |
| DEF-001 | defense evaluator | clean/attack denominator test | EXP-DEF-001 | trade-off table |
| DEF-002 | experiment split validator | leakage regression | EXP-DEF-002 | held-out report |
| DET-001 | detection evaluator | detector/auth separation | EXP-DET-001 | detector ROC |
| EXP-001 | run-manifest builder | manifest validation | EXP-REP-001 | run manifest |
| SEC-001 | artifact policy | sensitive-file scan | EXP-DATA-001 | release checklist |
