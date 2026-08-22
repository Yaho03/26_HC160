# Glossary

| Term | Definition |
|---|---|
| Enrollment | Creating the reference biometric representation for a claimed identity. |
| Probe | Face evidence compared with an enrollment reference. |
| Verification | One-to-one comparison answering whether a probe matches the claimed identity. |
| Identification/classification | Selecting one identity/class among several; not the primary financial-authentication task. |
| Genuine pair | Enrollment and probe belong to the same identity. |
| Impostor pair | Enrollment and probe belong to different identities. |
| FAR/FMR | Fraction of impostor attempts incorrectly accepted. |
| FRR/FNMR | Fraction of genuine attempts incorrectly rejected. |
| TAR | Fraction of genuine attempts correctly accepted. |
| EER | Operating point where FAR and FRR are approximately equal. |
| TAR@FAR | Genuine acceptance measured at a specified false-acceptance target. |
| Targeted ASR | Fraction of eligible reject-before impersonation attempts changed to accept. |
| White-box attack | Attacker can use model details and gradients. |
| Black-box attack | Attacker uses model queries without internal gradients. |
| Adaptive attack | Attack explicitly targets the defended model or defense transformation. |
| Transfer attack | Attack generated on one model and tested on another. |
| Defense success | A successful pre-defense attack becomes rejected after defense. |
| Clean degradation | Loss in clean verification performance caused by a defense. |
| Detector | Component producing suspicion evidence; not identical to authentication decision. |
| Calibration split | Data used to select thresholds and operating parameters. |
| Test split | Untouched data used only after parameters are frozen. |
| Artifact | Versioned file or record with hash, producer, sensitivity, and lineage. |
| Run manifest | Immutable metadata connecting code, configuration, inputs, outputs, environment, and command. |
| Identity drift | Purification or transformation changes the biometric identity representation. |
| PAD | Presentation Attack Detection for print, screen, mask, or replay media. |
| Legacy result | Historical result preserved under its original method even if it does not meet the new contract. |
