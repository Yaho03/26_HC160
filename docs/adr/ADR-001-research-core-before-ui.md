# ADR-001: Research Core Before UI

- Status: Accepted for redesign
- Date: 2026-08-22

## Context

The repository contains meaningful experiments and a platform-oriented DB design, but metric definitions and artifact provenance are not yet unified.

## Decision

The reproducible research pipeline is authoritative. A UI is optional and may initially display only completed, validated run artifacts.

## Consequences

- Metric and contract work precedes API/DB/dashboard work.
- UI code cannot mutate completed experiment configuration or thresholds.
- No microservice or broker infrastructure is introduced for the research pipeline.
