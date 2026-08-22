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
