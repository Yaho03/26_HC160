# HC160 Research Requirements

## 1. Requirement language

- **MUST**: required for a result to be considered valid.
- **SHOULD**: expected unless a documented ADR approves an exception.
- **MAY**: optional extension.

## 2. Requirements

### Data and preprocessing

| ID | Strength | Requirement | Acceptance evidence |
|---|---|---|---|
| DATA-001 | MUST | Every dataset snapshot has an immutable manifest and checksum. | Schema-valid manifest; all referenced files match hashes. |
| DATA-002 | MUST | Train, calibration, development, and test roles are explicit. | Automated split-overlap test passes. |
| DATA-003 | MUST | Image references are artifact-relative, never machine-absolute. | Contract validation rejects `/content`, home, and drive-specific paths. |
| DATA-004 | MUST | Dataset license, source, and redistribution limits are recorded. | Dataset metadata contains non-empty license fields. |

### Models and verification

| ID | Strength | Requirement | Acceptance evidence |
|---|---|---|---|
| MODEL-001 | MUST | Every checkpoint records architecture, source, license, dataset, configuration, Git commit, seed, and SHA-256. | Checkpoint sidecar validation passes. |
| VER-001 | MUST | Thresholds are selected on calibration data only and frozen for test evaluation. | Threshold artifact identifies disjoint calibration manifest. |
| VER-002 | MUST | Clean verification reports FAR/FMR, FRR/FNMR, EER, ROC-AUC, and TAR at supported FAR levels. | Metric tests and clean test report. |
| VER-003 | MUST | Model-specific preprocessing and threshold versions are inseparable. | Run validation rejects mismatched model/preprocessing/threshold IDs. |

### Attacks

| ID | Strength | Requirement | Acceptance evidence |
|---|---|---|---|
| ATK-001 | MUST | Targeted impersonation success means a reject-to-accept transition. | `success_from_reject == !accepted_before && accepted_after`. |
| ATK-002 | MUST | Threat model and perturbation/query budgets are stored per result. | Attack contract validation passes. |
| ATK-003 | MUST | Canonical adversarial images use a lossless format. | PNG/tensor artifact hash is present; JPEG is marked as a derived transform. |
| ATK-004 | SHOULD | In-house implementations are cross-checked against a reference implementation on a smoke subset. | Comparison run and tolerance report. |

### Defenses and detection

| ID | Strength | Requirement | Acceptance evidence |
|---|---|---|---|
| DEF-001 | MUST | Every defense is evaluated on both attack and clean inputs. | Attack ASR and clean TAR/FRR delta appear in the same report. |
| DEF-002 | MUST | Trained defenses use disjoint training, validation, and test attack samples. | Leakage test passes; split IDs are recorded. |
| DEF-003 | MUST | A trained or differentiable defense is evaluated under an adaptive attack. | Adaptive-attack run references the defended checkpoint. |
| DEF-004 | MUST | Transform, trained, ensemble, temporal, and detector components use distinct semantics. | Interface type is recorded in defense metadata. |
| DET-001 | MUST | Detector alerts are not silently treated as authentication accuracy. | Detection metrics and authentication decisions are reported separately. |

### Experiments and reports

| ID | Strength | Requirement | Acceptance evidence |
|---|---|---|---|
| EXP-001 | MUST | Every completed run records configuration, code, environment, seed, device, inputs, outputs, and reproduction command. | Run manifest validation passes. |
| EXP-002 | MUST | Every metric includes numerator, denominator, unit, and grouping. | Aggregate-result validation passes. |
| EXP-003 | MUST | Randomized experiments use at least three seeds unless explicitly exempted. | Seed count appears in report or an approved limitation is cited. |
| RPT-001 | MUST | Reports distinguish measured facts, limitations, and inference. | Report checklist passes. |
| RPT-002 | MUST | Legacy published results are preserved and never silently regenerated. | Legacy artifact hashes remain unchanged. |

### Security, ethics, and optional UI

| ID | Strength | Requirement | Acceptance evidence |
|---|---|---|---|
| SEC-001 | MUST | Raw faces, embeddings, and checkpoints are treated as sensitive artifacts. | Git policy check and artifact sensitivity labels. |
| SEC-002 | MUST | Reports state that the system is not production financial authentication. | Required disclaimer is present. |
| SEC-003 | MUST | Generative AI work is restricted to defensive purification unless separately approved. | Experiment registry scope check. |
| UI-001 | MAY | A UI may display completed, validated runs. | Displayed values match artifact hashes and run IDs. |
| UI-002 | MUST if UI exists | UI and research pipeline remain separate. | UI cannot mutate thresholds or overwrite completed runs. |

## 3. Traceability rule

Each implemented requirement must link:

```text
requirement ID -> implementation -> configuration -> automated test
               -> experiment ID -> run/artifact -> report section
```

The initial traceability table lives in `08_EXPERIMENT_PLAN.md` and should later be generated from run manifests.
