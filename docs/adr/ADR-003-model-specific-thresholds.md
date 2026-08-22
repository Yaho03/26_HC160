# ADR-003: Model-Specific Threshold Artifacts

- Status: Accepted for redesign
- Date: 2026-08-22

## Context

The same numeric threshold appears in ResNet bridge, FaceNet, temporal, camera, and DB code despite different preprocessing paths.

## Decision

A threshold is a versioned artifact bound to model, checkpoint, preprocessing, score function, calibration manifest, and selection method.

## Consequences

- Hardcoded threshold constants are legacy-only.
- Model or preprocessing changes require recalibration.
- Test data cannot create or update a threshold artifact.
