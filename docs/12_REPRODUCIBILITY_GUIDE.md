# Reproducibility Guide

## 1. Reproduction levels

| Level | Requirement |
|---|---|
| Unit | Standard-library contract and metric tests pass. |
| Smoke | Small CPU dataset runs through manifest, scoring, and reporting. |
| Baseline | Frozen dataset, checkpoint, threshold, and configuration reproduce aggregate metrics within tolerance. |
| Full | GPU attack/defense run reproduces all artifacts and report tables. |

## 2. Required run metadata

- Git commit and dirty-worktree flag;
- direct dependency lock hash;
- Python, PyTorch, torchvision, CUDA/MPS, and OS versions;
- device name and deterministic-mode settings;
- dataset, pair, checkpoint, preprocessing, and threshold artifact IDs;
- complete configuration and hash;
- Python, NumPy, and PyTorch seeds;
- start/end time and reproduction command;
- output artifact hashes.

## 3. Environment policy

`environment.yml` is retained for the original environment. New setup files should list direct dependencies only and separate lightweight tests, research ML, FaceNet, and camera extras. Machine-specific Conda `prefix` values are not part of a portable lock.

The FaceNet code requires `facenet-pytorch`; the existing general environment does not currently declare it. Until environments are consolidated, use the dedicated FaceNet requirement file for that track.

## 4. Seed policy

Set the same integer for Python, NumPy, PyTorch CPU, and all CUDA devices. Record whether deterministic algorithms are enabled. Deterministic mode may reduce performance or reject unsupported operations; any exemption is recorded in the run manifest.

## 5. Artifact policy

- Git: code, configuration, schemas, metadata, summary tables, and approved figures.
- External storage: datasets, raw/derived face images, embeddings, and checkpoints.
- Each external artifact has a Git-tracked metadata row with hash and sensitivity.
- Completed run artifacts are append-only.

## 6. Colab policy

- Checkout a specific commit or tag, not an unbounded `git pull`.
- Install a pinned direct-dependency file.
- Restore artifacts only after verifying hashes.
- Write outputs to a new run directory.
- Do not commit or push from the experiment Notebook.

## 7. Current baseline caveat

Historical attack images and checkpoints are stored outside Git. Their hashes and exact package versions were not recorded in the original runs. Those results remain preserved as legacy evidence but cannot be classified as fully reproducible until provenance is recovered.
