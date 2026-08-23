# ADR-004: Files-First Experiment Registry

- Status: Accepted for redesign
- Date: 2026-08-22

## Context

The project is primarily a Colab/Python research workflow. A database does not solve missing provenance and adds deployment complexity.

## Decision

Use immutable run directories containing JSON/CSV/Parquet metadata and artifact hashes. PostgreSQL remains an optional future consumer for a demo platform.

## Consequences

- Experiments run without a database service.
- Run artifacts are portable between local and Colab environments.
- The existing `DB_SCHEMA.md` is treated as a platform proposal, not current implementation truth.
