# ADR-002: Lossless Canonical Adversarial Artifacts

- Status: Accepted for new runs
- Date: 2026-08-22

## Context

Existing FaceNet attack images were serialized as JPEG. JPEG error exceeded small perturbation budgets and changed attack success after reload.

## Decision

New canonical adversarial images use lossless PNG or tensor artifacts. JPEG outputs are derived artifacts with explicit parent hash, quality, and encoder metadata.

## Consequences

- Existing JPEG results remain unchanged as legacy evidence.
- Attack success is measured again from the canonical serialized artifact.
- Serialization is part of the experiment contract.
