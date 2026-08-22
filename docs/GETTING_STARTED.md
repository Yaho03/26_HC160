# Getting Started

## 1. Read first

1. `00_PROJECT_OVERVIEW.md`
2. `01_RESEARCH_REQUIREMENTS.md`
3. `04_DATA_AND_ARTIFACT_CONTRACT.md`
4. `09_EVALUATION_METRICS.md`
5. `11_SECURITY_ETHICS_AND_LIMITATIONS.md`

## 2. Lightweight validation

The research contract and metric tests use the standard library:

```bash
python -m unittest discover -s tests/research -v
```

This does not install or execute the GPU research pipeline or the separate face-auth application tests.

## 3. Choose a track

- Legacy classification: follow the existing README and Colab attack pipeline.
- Verification research: provide LFW data, a versioned embedding checkpoint, a pair manifest, and a threshold artifact.
- Face-auth application prototype: follow its separate documentation and dependency file.

Do not mix checkpoints or thresholds across tracks.

## 4. Before running an experiment

- confirm dataset/model license;
- validate input manifests;
- record a clean Git commit;
- choose a new run ID and seed;
- freeze the configuration;
- verify external artifact hashes;
- confirm that test data was not used for calibration or tuning.

## 5. Before publishing results

- run all tests;
- show numerators and denominators;
- include clean-performance preservation;
- remove local absolute paths and sensitive artifacts;
- cite run IDs, hashes, and limitations.
