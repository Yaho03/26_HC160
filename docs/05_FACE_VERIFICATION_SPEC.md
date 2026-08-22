# Face Verification Specification

## 1. Decision model

For enrollment embedding `e` and probe embedding `p`:

```text
score = cosine_similarity(e, p)
accept = score >= threshold
```

The embedding model, face preprocessing, score function, and threshold artifact form one versioned verification protocol. A threshold from a different model or preprocessing version is invalid.

## 2. Supported tracks

| Protocol | Role | Image size | Current model |
|---|---|---:|---|
| `resnet50-lfw10-bridge-v1` | Legacy bridge | 224 | LFW-10 classifier backbone |
| `facenet-vggface2-v1` | Primary candidate | 160 | InceptionResnetV1, VGGFace2 weights |
| `camera-facenet-prototype-v1` | Camera exploration | 160 | MTCNN crop + FaceNet |

Scores and thresholds from these protocols must not be merged.

## 3. Calibration

1. Build a frozen calibration-pair manifest.
2. Compute embeddings with a pinned model and preprocessing version.
3. Select the operating threshold using only calibration rows.
4. Store threshold, selection method, supported FAR range, source hashes, and software versions.
5. Freeze the threshold before attack, defense, or final test evaluation.

EER may be reported for comparison, but a financial scenario should also report TAR at supported low-FAR operating points. The repository must not claim a FAR level that the available negative-pair count cannot statistically support.

## 4. Clean evaluation

Required outputs:

- pair-level score and accept/reject decision;
- TP, TN, FP, FN;
- FAR/FMR, FRR/FNMR, TAR, TNR;
- EER and ROC-AUC;
- TAR at supported FAR targets;
- sample counts and confidence intervals;
- model, preprocessing, threshold, dataset, and protocol IDs.

## 5. Attack evaluation

Targeted impersonation attempts must begin from different-identity pairs rejected before attack. Primary success is:

```text
success_from_reject = (accepted_before is False) and (accepted_after is True)
```

`accepted_after` alone is not a valid targeted attack-success definition because it includes pairs already falsely accepted before attack.

## 6. Defense evaluation

Every defense is applied to:

- eligible successful adversarial probes;
- the corresponding clean probes;
- an independently sampled clean verification test set.

Reports must show attack ASR after defense and clean TAR/FRR change together. A defense that rejects everything is invalid even if its attack ASR is zero.

## 7. Threshold lifecycle

Threshold artifacts use an ID such as:

```text
thr_<protocol>_<calibration-manifest-hash>_<method-version>
```

Any change to weights, detector/alignment, resize, normalization, score function, or calibration data creates a new threshold artifact.

## 8. Executable calibration contract

`src/evaluation/verification_calibration.py` implements the model-independent part of EXP-VER-001. It accepts pair-level similarity scores only after embeddings have been generated and enforces:

- calibration rows only during threshold selection;
- test rows only during clean baseline evaluation;
- identical protocol, model, and preprocessing IDs;
- exact calibration pair-ID provenance;
- zero calibration/test pair overlap;
- both genuine and impostor rows in each evaluated split;
- finite numeric scores and unique pair IDs.

For a requested target FAR, at least `1 / target_far` impostor calibration pairs are required. A smaller set may report observed zero false accepts but cannot claim that FAR operating point. EER selection remains available for comparison and never re-selects the frozen test threshold.

Each score JSONL row follows `schemas/verification-score.schema.json`. Once model inference has produced disjoint calibration and test score files, generate both outputs with:

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

The command refuses to overwrite outputs by default. The report includes fixed-threshold confusion counts, explicit rate numerators and denominators, Wilson 95% intervals, ROC-AUC, and a clearly labeled descriptive test EER.
