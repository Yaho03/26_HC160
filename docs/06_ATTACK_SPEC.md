# Attack Specification

## 1. Purpose

Attack modules evaluate the robustness of an authorized research model. They are not instructions for attacking third-party systems. Experiments are restricted to project-owned models, data, and consented demonstration environments.

## 2. Threat models

| Type | Knowledge | Current examples |
|---|---|---|
| White-box | Model, preprocessing, gradients, and threshold are available. | FGSM, PGD, multi-pixel JSMA variant. |
| Score-based black-box | Only output scores/probabilities and query access are available. | Square-style and ZOO-style attacks. |
| Transfer | Attack is generated against a source model and evaluated on a different target model. | Planned cross-model experiment. |
| Adaptive defense-aware | Attack includes the known defense or defended model. | Required for trained/differentiable defenses. |

## 3. Common interface

An attack receives:

- source/probe and target-enrollment artifact IDs;
- verification protocol and threshold artifact IDs;
- attack configuration and threat model;
- seed and query/time budget.

It returns a schema-valid attack result plus optional canonical PNG/tensor artifacts. Attack implementations do not compute aggregate report metrics.

## 4. Required measurements

- score and decision before/after attack;
- reject-to-accept success;
- L0, L2, and L-infinity where meaningful;
- perturbation scale and norm space;
- steps, queries used, elapsed time;
- random seed and early-stop reason;
- canonical artifact hash and serialization format;
- errors separately from failed attacks.

## 5. Fair comparison rules

- Use the same eligible pair manifest for compared attacks.
- Freeze the target-selection policy.
- Report each attack's native constraint; do not equate JSMA sparsity with an L-infinity budget.
- Report black-box ASR as a function of query budget.
- Include failures and timeouts in denominators unless the protocol pre-registers another rule.
- Never tune attack parameters on the final test set.

## 6. Legacy mapping

The five scripts under `src/attacks/` are classification baselines. `src/verification/targeted_pgd_verification.py` is a ResNet-feature verification bridge. Committed FaceNet defense artifacts require a FaceNet batch attack producer before they become end-to-end reproducible.

Legacy result files remain unchanged and are labeled with their original success definition.
