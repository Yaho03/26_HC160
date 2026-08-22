# HC160 Local Runbook

## 1. Workspace and branch

Run commands from the repository root:

```text
<workspace>/test_HC160
```

Expected development branch:

```text
codex/realtime-face-auth-v2
```

Confirm the branch and review local changes before running or committing work:

```bash
git status -sb
```

## 2. Python environments

- Target runtime: Python 3.11.
- Locally verified full-prototype runtime: Python 3.9 with `requirements-face-auth.txt`.
- Research contract tests also run with the system Python 3.13 because they have no ML dependency.

Create an isolated environment and install face-auth dependencies when required:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-face-auth.txt
```

Install the optional ONNX PAD runtime only when evaluating an approved ONNX checkpoint:

```bash
python -m pip install -r requirements-pad-onnx.txt
```

Do not commit `.venv`, downloaded weights, raw faces, embeddings, templates, or PAD models.

## 3. Validation commands

Fast dependency-free research validation:

```bash
python -m unittest discover -s tests/research -v
```

Full face-auth and research validation after installing dependencies:

```bash
python -m unittest discover -s tests -v
```

At the documented implementation snapshot the full command passed 126 tests on Python 3.9. Warning output from the test-only TorchScript trace is not a failure; any failed or errored test is.

## 4. Dataset manifest workflow

The manifest builder accepts only artifact-ready files under pseudonymous identity directories:

```text
<artifact-root>/<split>/id_<16-or-more-lowercase-hex>/<image>.png
```

See `03_DATASET_AND_PREPROCESSING.md` for the complete build command. Validate an existing manifest and referenced media with:

```bash
python -m src.datasets.manifest_cli validate \
  --manifest data/manifests/DATASET.jsonl \
  --artifact-root external_artifacts/DATASET \
  --require-identity-disjoint
```

Omit `--require-identity-disjoint` only when the protocol does not claim open-set subject separation. Media hashes remain forbidden from crossing split roles.

## 5. Verification baseline workflow

Model inference must first emit score JSONL rows matching `schemas/verification-score.schema.json`. Calibration and test rows must have different pair IDs and identical protocol, model, and preprocessing IDs.

```bash
python -m src.evaluation.verification_baseline_cli \
  --calibration-scores outputs/verification/calibration-scores.jsonl \
  --test-scores outputs/verification/test-scores.jsonl \
  --selection-method target_far \
  --target-far 0.001 \
  --calibration-manifest-id pairs-calibration-v1 \
  --calibration-manifest-sha256 CALIBRATION_MANIFEST_SHA256 \
  --test-manifest-id pairs-test-v1 \
  --test-manifest-sha256 TEST_MANIFEST_SHA256 \
  --threshold-artifact-id thr-facenet-v1 \
  --run-id run-exp-ver-001 \
  --threshold-output outputs/verification/threshold.json \
  --report-output outputs/verification/clean-report.json
```

The command rejects an unsupported target FAR. For example, FAR `0.001` requires at least 1,000 impostor calibration pairs under the repository's minimum empirical support rule.

## 6. Face-auth baseline workflow

Create a local enrollment template from a recorded video:

```bash
python -m src.face_auth.cli enroll \
  --video path/to/enrollment.mp4 \
  --frames 30 \
  --min-valid-frames 5 \
  --output local_templates/user-1.npz
```

Authenticate a separate probe recording:

```bash
python -m src.face_auth.cli authenticate \
  --video path/to/probe.mp4 \
  --template local_templates/user-1.npz \
  --threshold 0.60 \
  --threshold-version validation-only-example \
  --user-id user-1 \
  --context-hash demo-context-a \
  --profile BASELINE_ONLY
```

`0.60` is not a release threshold. Replace it with the frozen artifact value produced from approved calibration data. Enrollment and probe video must be separate captures.

Use `--camera 0` instead of `--video` for a webcam. The baseline profile does not include PAD, active liveness, or continuity and must not be presented as complete authentication security.

## 7. FULL profile prerequisites

The FULL profile requires all of the following before invocation:

- a validated TorchScript or supported ONNX PAD model;
- PAD preprocessing and live-class semantics matching the CLI arguments;
- identity, PAD, motion, continuity, and optional adversarial threshold versions;
- enough post-challenge frames to evaluate active liveness;
- target-device validation evidence.

If `--pad-model` is absent, FULL refuses to start. This is the intended fail-closed behavior. See `face_auth/README.md` for the command and gate map.

## 8. Passive PAD evaluation workflow

The PAD manifest uses opaque subject/session/device tokens and separates calibration from test rows. The example file illustrates format only:

```text
configs/pad_evaluation.example.csv
```

After acquiring an approved TorchScript or ONNX PAD model, select the threshold on calibration rows only. This TorchScript example uses the runtime default:

```bash
python -m src.face_auth.evaluation.pad_cli \
  --mode calibrate \
  --manifest data/manifests/pad-evaluation.csv \
  --manifest-id pad-evaluation-v1 \
  --artifact-root external_artifacts/pad \
  --pad-model local_models/pad-v1.ts \
  --pad-model-version pad-v1 \
  --threshold-version pad-validation-v1 \
  --run-id run-pad-calibration-v1 \
  --split calibration \
  --max-bpcer 0.05 \
  --output outputs/pad/pad-calibration-v1.json
```

Freeze the reported threshold, then evaluate the untouched test split with:

```bash
python -m src.face_auth.evaluation.pad_cli \
  --mode evaluate \
  --manifest data/manifests/pad-evaluation.csv \
  --manifest-id pad-evaluation-v1 \
  --artifact-root external_artifacts/pad \
  --pad-model local_models/pad-v1.ts \
  --pad-model-version pad-v1 \
  --live-threshold 0.80 \
  --threshold-version pad-validation-v1 \
  --run-id run-pad-test-v1 \
  --split test \
  --output outputs/pad/pad-test-v1.json
```

For the original Open Model Zoo `anti-spoof-mn3` ONNX artifact, add `--pad-runtime onnx`; the adapter then uses its documented 128x128 RGB input, class-zero bona-fide output, and mean/scale defaults. Override model-contract arguments only when the frozen checkpoint documentation requires it.

The evaluator records sample outcomes, valid-frame counts, model/threshold versions, latency, APCER, BPCER, ACER, worst-species and per-species APCER, presentation/exclusion counts, and Wilson 95% intervals. It also binds the run to the Git commit, manifest/model SHA-256, and each selected source video's SHA-256 and byte count. Inputs are rechecked after evaluation so mid-run changes abort the report.

Formal runs require a clean Git worktree and refuse an existing output path. `--allow-dirty` and `--overwrite` exist for explicit local debugging; do not use them for reportable experiments. Multi-face, insufficient-quality, load, and model failures are reported separately instead of being counted as PAD success.

## 9. Scenario and gate-threshold workflows

Build a deterministic attack video:

```bash
python -m src.attack_scenarios.cli \
  --manifest configs/scenarios/mid_frame_insertion.example.json
```

Calibrate prototype gate thresholds on validation rows only:

```bash
python -m src.face_auth.evaluation.calibrate_cli \
  --input configs/thresholds.validation.example.csv \
  --output local_thresholds/validation-v1.json \
  --version validation-v1
```

The example CSV is a format illustration, not experimental evidence.

## 10. Common failures

| Symptom | Meaning and response |
|---|---|
| `ModuleNotFoundError: numpy` | The system Python is being used without face-auth dependencies. Activate the intended environment. |
| FaceNet/MTCNN download or load failure | External pretrained weights are unavailable or uncached. Verify network, cache, version, license, and checksum. |
| `FULL profile requires --pad-model` | Expected configuration block; provide a validated model or use the explicitly limited baseline profile. |
| Requested FAR is unsupported | Increase impostor calibration pairs or select a statistically supported operating point. |
| Output already exists | Tools refuse silent overwrite. Use a new run/artifact ID; use `--overwrite` only for disposable local trials. |
| Many quality retries | Do not immediately lower thresholds. Measure target-device clean validation data first. |
| Replay detector misses re-encoded frames | Include codec transfer in validation and calibrate the content-distance threshold. |

## 11. Before committing or reporting

- confirm `git status -sb` contains only intended files;
- run the relevant research and full tests;
- record dataset, model, preprocessing, threshold, policy, and Git versions;
- preserve numerator, denominator, confidence interval, failures, and latency;
- exclude faces, embeddings, templates, checkpoints, and local absolute paths;
- distinguish implemented code, smoke evidence, final experiment evidence, and production extensions.
