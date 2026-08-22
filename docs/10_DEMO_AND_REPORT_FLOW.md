# Demo and Report Flow

## 1. Demo principle

The demo explains a validated experiment run. It does not generate unreviewed security claims during presentation.

## 2. Recommended scenario

```text
1. Select a completed run ID.
2. Show dataset/model/threshold provenance.
3. Show a clean genuine and clean impostor decision.
4. Show source, target enrollment, and lossless adversarial probe.
5. Show reject-to-accept transition and attack budget.
6. Apply one defense and show the defended decision.
7. Show clean-performance cost beside attack reduction.
8. Show detector evidence separately from authentication decision.
9. Finish with limitations and reproduction command.
```

## 3. Required report sections

- scope and threat model;
- dataset, split, model, preprocessing, and threshold versions;
- clean baseline;
- attacks and budgets;
- defenses and clean trade-off;
- adaptive and transfer evaluation where applicable;
- runtime and reproducibility;
- sample counts, errors, and confidence intervals;
- privacy, ethics, and limitations;
- run IDs and artifact hashes.

## 4. Failure scenarios

The demo must include at least one expected failure:

- missing artifact or schema mismatch;
- low-quality retryable capture;
- model error that fails closed;
- attack that does not cross threshold;
- defense that lowers ASR but damages clean TAR;
- token replay or context mismatch in the separate reference prototype.

## 5. Prohibited presentation

- “100% secure” or equivalent claims;
- presenting training-set adversarial performance as held-out robustness;
- using temporal still-image results as camera deployment evidence;
- claiming certified robustness without a certificate;
- displaying identifiable face artifacts without authorization.
