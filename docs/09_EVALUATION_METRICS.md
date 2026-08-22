# Evaluation Metrics

## 1. Verification confusion counts

For `same_identity` ground truth and `accepted` decision:

- TP: genuine pair accepted;
- FN: genuine pair rejected;
- FP: impostor pair accepted;
- TN: impostor pair rejected.

```text
FAR/FMR = FP / (FP + TN)
FRR/FNMR = FN / (TP + FN)
TAR      = TP / (TP + FN)
TNR      = TN / (TN + FP)
accuracy = (TP + TN) / all pairs
```

Zero denominators produce an explicit undefined value, never zero.

## 2. Attack metrics

Eligible targeted attempts are different-identity pairs rejected before attack.

```text
targeted ASR = reject-to-accept successes / eligible targeted attempts
```

Also report L0/L2/L-infinity, query count, elapsed time, success by budget, and failure/error counts.

## 3. Defense metrics

```text
conditional defense success rate
  = accepted attacks changed to reject / accepted attacks before defense

conditional ASR after defense
  = attacks still accepted / accepted attacks before defense

population ASR after defense
  = attacks accepted after defense / all eligible attack attempts
```

These rates have different denominators and must not share one label.

Clean preservation:

```text
clean TAR delta = TAR_after_defense - TAR_before_defense
clean FRR delta = FRR_after_defense - FRR_before_defense
```

## 4. Detector metrics

Report TPR, FPR, precision, recall, ROC-AUC, threshold, attack species, and sample counts. Detector failures and authentication decisions remain separate.

## 5. EER and operating points

EER is estimated on calibration scores by selecting the threshold minimizing `abs(FAR - FRR)`. The selected threshold is then frozen. Final test EER may be reported descriptively, but it must not replace evaluation at the preselected threshold.

TAR@FAR is reported only where the number of impostor pairs supports the requested FAR. Reports include a confidence interval and do not imply production-grade certainty from a small dataset.

## 6. Runtime

- Attack and defense times have clear start/end boundaries.
- Model load and warm-up are reported separately from steady-state inference.
- Report p50, p95, and sample count on named hardware.
- Randomized and temporal methods report forward-pass/frame count.

## 7. Legacy-result warning

The committed classification report and verification report use different task definitions. The current verification summary also uses all 212 rows as the denominator for `defense_success_rate`; future centralized evaluation must preserve that historical value while exposing the conditional denominator separately.

## 8. Empirical FAR support rule

With `N` impostor pairs, the smallest non-zero empirical FAR is `1/N`. Therefore a target FAR is treated as supported only when:

```text
N * target_far >= 1
```

This is a minimum reporting gate, not proof of production-level certainty. Threshold artifacts store the impostor denominator, achieved FAR numerator/denominator, calibration manifest hash, and pair-ID-set hash. Final test evaluation uses the frozen numeric threshold and reports its own counts without changing the artifact.
