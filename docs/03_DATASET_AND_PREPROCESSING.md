# Dataset and Preprocessing

## 1. Current datasets

The repository currently assumes LFW deep-funneled images. The legacy classification pipeline chooses the ten identities with the most qualifying images, shuffles each identity, and creates image-level train/validation/test directories. This protocol is retained for historical classification reproduction only.

The primary verification protocol must use a frozen pair manifest with explicit train, calibration, development, and test roles. Threshold calibration and final test evaluation must never reuse the same pair rows.

## 2. Required dataset metadata

Every dataset snapshot records:

- dataset ID and schema version;
- source page and retrieval date;
- license and redistribution restrictions;
- archive SHA-256 and expected file count;
- preprocessing implementation and version;
- sample-level relative path and media hash;
- pseudonymous identity token;
- split and protocol IDs;
- known demographic and collection limitations.

Raw names may be needed to reconstruct LFW locally, but committed manifests and reports should use pseudonymous identity tokens.

## 3. Split policy

| Role | Permitted use |
|---|---|
| Train | Model and trained-defense fitting. |
| Calibration | Threshold and detector cutoff selection. |
| Development | Attack/defense parameter exploration. |
| Test | One final evaluation with frozen configuration. |

Where the experiment claims open-set generalization, subject identities must be disjoint. At minimum, pair IDs and media hashes must be disjoint across roles. Enrollment and probe images from the same capture must not cross roles.

## 4. Preprocessing identity

Preprocessing is part of the model artifact and must be versioned. Required fields include:

- face detector and alignment model/version;
- crop margin and image size;
- interpolation mode;
- channel order and numeric range;
- mean/standard-deviation normalization;
- color profile and image decoder version;
- whether the input is original, PNG, JPEG, or a defense-derived artifact.

Current protocols:

| Track | Current preprocessing |
|---|---|
| ResNet classification/bridge | Resize to 224x224, RGB, ImageNet normalization. |
| FaceNet verification defense | Resize to 160x160, RGB, `(pixel - 127.5) / 128.0`. |
| Camera prototype | MTCNN crop to 160x160 followed by FaceNet normalization. |

These are separate protocols and must not share a threshold unless a calibration run explicitly proves equivalence.

## 5. Artifact format

- Canonical clean and adversarial image: lossless PNG or tensor artifact.
- JPEG: derived artifact whose encoder, quality, and parent hash are recorded.
- Paths in tables: artifact-root-relative POSIX paths.
- Every image artifact: SHA-256, byte count, MIME type, sensitivity, and parent ID.
- Raw faces and embeddings: excluded from Git.

## 6. Leakage checks

Automated validation must reject:

- the same media hash in train and test;
- the same pair ID in calibration and test;
- trained-defense evaluation on its own attack-training rows;
- threshold artifacts created from test rows;
- result rows whose checkpoint or preprocessing ID differs from the threshold artifact.

## 7. LFW limitations

LFW is suitable for a research prototype, not for validating production financial authentication. It is not a representative sample of banking customers, cannot substantiate fairness claims, and may overlap with identities seen by public face-model pretraining datasets. Reports must disclose these limits.
