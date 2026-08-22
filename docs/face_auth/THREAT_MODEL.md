# Threat Model

## Protected Assets

- biometric template and template version;
- authentication session, nonce, and challenge;
- ordered frame evidence and its digest;
- gate results and policy decision;
- short-lived verification token and its consumed state.

## In-Scope Threats

| ID | Threat | Required response |
|---|---|---|
| ATK-01 | Unregistered impostor | identity failure |
| ATK-02 | Printed face | PAD or liveness failure |
| ATK-03 | Static phone screen | PAD or liveness failure |
| ATK-04 | Replayed face video | randomized challenge failure |
| ATK-05 | Mid-session image insertion | continuity, PAD, or adversarial failure |
| ATK-06 | Mid-session person switch | multi-face or continuity failure |
| ATK-07 | Digital adversarial input | secondary adversarial inspection |
| ATK-08 | Screen/print-transferred adversarial input | PAD, continuity, and adversarial inspection |
| ATK-09 | Blur, darkness, camera shake | retry instead of false attack accusation |
| ATK-10 | Frame replay or reorder | frame-integrity failure |
| ATK-11 | Verification-token replay | reject second consume |
| ATK-12 | Changed transaction context | reject consume and require new authentication |

## Explicit Limitations

The Python prototype cannot prove that a compromised operating system, camera driver, or virtual camera supplied genuine sensor frames. A future mobile implementation must add app/device attestation and hardware-backed keys. This limitation must be disclosed in reports and demos.
