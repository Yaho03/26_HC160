# HC160 System Architecture

## 1. Architectural principle

The reproducible research pipeline is authoritative. Interactive demonstrations and future persistence layers are consumers of immutable run artifacts.

```text
Dataset + license metadata
        -> immutable sample manifest
        -> pair and split manifests
        -> model/checkpoint + preprocessing version
        -> calibration threshold artifact
        -> clean baseline
        -> attack run
        -> defense/detection run
        -> centralized evaluation
        -> tables, figures, and report
        -> optional read-only demo
```

## 2. Research layers

| Layer | Responsibility | Must not do |
|---|---|---|
| Dataset | Build manifests and fixed splits. | Select a threshold or modify test labels. |
| Model | Load a versioned classifier or embedding model. | Infer paths or thresholds from local machines. |
| Verification | Embed pairs, calibrate thresholds, and make accept/reject decisions. | Tune on test results. |
| Attack | Produce bounded adversarial probes and attack metadata. | Redefine verification metrics. |
| Defense | Transform inputs or models and emit defense results. | Count clean rejection as defense success. |
| Detection | Emit suspicion signals and evidence. | Claim authentication accuracy from detection rate. |
| Evaluation | Compute all metric numerators and denominators. | Mutate source artifacts. |
| Reporting | Render validated aggregates. | Recalculate hidden metrics with different definitions. |
| Demo | Display a selected validated run. | Change threshold/configuration of completed runs. |

## 3. Existing-code mapping

| Existing path | Architectural role |
|---|---|
| `src/datasets/` | Dataset preparation; must gain manifest output. |
| `src/training/` | Legacy classification training. |
| `src/attacks/` | Legacy classification attacks. |
| `src/verification/` | Verification bridge. |
| `src/verification/defenses/` | FaceNet verification defense prototypes. |
| `src/defenses/` | Legacy classification defenses. |
| `src/reports/` | Legacy report builders; future consumers of centralized metrics. |
| `src/face_auth/` | Session/policy/token reference prototype. |
| `outputs/` | Historical committed summaries plus future immutable run directories. |

## 4. Research and service boundary

The `src/face_auth` reference prototype controls session state, challenges, fail-closed gate policy, and one-time token consumption. It does not make legacy experiment CSVs production evidence. Model gates must return versioned `GateResult` values, while research evaluation independently measures model behavior on fixed datasets.

An optional service may later provide:

- session creation and evidence submission;
- a read-only experiment browser;
- artifact metadata lookup;
- controlled demo scenario execution.

It does not require microservices, a message broker, or a time-series database.

## 5. Trust boundaries

- Files outside the immutable manifest are untrusted inputs.
- Model weights are accepted only when their hash and metadata match.
- A client-provided timestamp, user identity, or decision is never authoritative.
- A camera-only Python prototype cannot attest the operating system, driver, or virtual camera.
- Completed run directories are append-only; corrections create a new run ID.

## 6. Failure behavior

- Contract violation: fail before model execution.
- Missing required artifact: run status `BLOCKED_INPUT`, not a zero-valued metric.
- Model exception: `ERROR`, not authentication rejection or defense success.
- Low-quality capture: retryable outcome when policy allows it.
- Missing required security gate: fail closed and do not issue a result token.
