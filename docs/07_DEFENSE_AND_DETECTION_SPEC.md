# Defense and Detection Specification

## 1. Component types

| Type | Output | Examples |
|---|---|---|
| `transform` | Defended image plus timing | JPEG, smoothing, bit-depth, ROI. |
| `trained` | Versioned defended checkpoint and scores | Adversarial training. |
| `ensemble` | Combined decision and component votes | ROI/smoothing/randomized vote. |
| `temporal` | Session-level decision from real frames | Continuity/static/replay checks. |
| `detector` | Suspicion score and evidence | Feature squeezing consistency. |

These types must not share an ambiguous `defense_success` definition.

## 2. Transform defense

A transform defense maps an input artifact to a derived artifact. It records parent hash, transform parameters, encoder/decoder version, processing time, and post-transform verification result.

It is successful on an attack input only when the attack was accepted before defense and rejected after defense. Its clean cost is measured independently.

## 3. Trained defense

- Training rows, model-selection rows, and final test rows are disjoint.
- Best checkpoint selection uses validation data only.
- Final evaluation includes clean pairs, held-out attacks, unseen attacks, and an adaptive attack against the defended checkpoint.
- Training-set ASR is diagnostic, not final evidence.

## 4. Randomized and ensemble methods

Randomized smoothing is described as a stochastic heuristic unless the implementation produces a formal certificate with assumptions and radius. Ensemble reports record every component vote, missing/error votes, tie policy, and total latency.

## 5. Temporal and camera methods

Synthetic augmentations of one still image are a simulation, not proof of real temporal robustness. A valid camera experiment requires:

- genuine multi-frame sessions;
- replay/print/screen sessions;
- subject and session separation;
- frame timestamps and drop counts;
- calibration on normal sessions only;
- false-positive rate and detection delay.

## 6. Detection semantics

Feature squeezing and similar methods emit:

- detector score;
- detector threshold/version;
- hit/no-hit;
- evidence per transformation;
- false-positive and false-negative metrics.

A detector may inform a risk score or veto policy, but detection rate is not defense success rate and is not verification accuracy.

## 7. Provisional acceptance gate

A defense may be called promising when it:

- reduces conditional attack ASR by at least 50% relative to no defense;
- decreases clean TAR by no more than 2 percentage points at the frozen operating threshold;
- reports 95% confidence intervals and all errors;
- satisfies its latency budget on named reference hardware.

These values are provisional research gates, not production guarantees. Any change must be recorded before observing final test results.
