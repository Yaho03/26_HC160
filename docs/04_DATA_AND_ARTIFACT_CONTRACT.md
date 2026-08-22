# Data and Artifact Contract

## 1. General rules

- All records include `schema_version`.
- `!` means required and non-null; `?` means optional or nullable.
- IDs are stable opaque strings; display names are not identifiers.
- Times are ISO-8601 UTC.
- Durations use milliseconds and field names end in `_ms`.
- Image-space perturbation norms use the `[0, 1]` pixel scale unless `norm_space` says otherwise.
- Completed records are immutable. Corrections create new IDs.
- Machine-absolute paths are invalid.

JSON Schema files in `/schemas` provide machine-readable minimum validation. This document defines semantics that JSON Schema alone cannot express.

## 2. Dataset manifest row

| Field | Type | Required | Null | Validation |
|---|---|---:|---:|---|
| `schema_version` | string | yes | no | Supported major version. |
| `dataset_id` | string | yes | no | Immutable snapshot ID. |
| `sample_id` | string | yes | no | Unique within dataset. |
| `identity_token` | string | yes | no | Pseudonymous; not a display name. |
| `relative_uri` | string | yes | no | Relative path only. |
| `media_sha256` | string | yes | no | 64 lowercase hex characters. |
| `split` | enum | yes | no | `train`, `calibration`, `development`, `test`. |
| `width_px`, `height_px` | integer | yes | no | Positive. |
| `license_id` | string | yes | no | Dataset license/policy reference. |
| `source_uri` | string | no | yes | Public source page if available. |

Producer: dataset builder. Consumers: pair builder, trainer, evaluator.

## 3. Verification pair row

Required fields: `schema_version:string!`, `pair_id:string!`, `protocol_id:string!`, `left_sample_id:string!`, `right_sample_id:string!`, `same_identity:boolean!`, `split:enum!`. Optional: `fold:integer?`, `pair_group_id:string?`.

Validation: left and right must differ; sample IDs must resolve to the same dataset; unordered duplicate pairs are forbidden; pair split must agree with the protocol manifest.

Producer: pair builder. Consumers: clean evaluator and attack runner.

## 4. Attack result row

Required fields:

- `attack_result_id:string!`, `run_id:string!`, `pair_id:string!`;
- `attack_id:string!`, `threat_model:enum!`, `attack_params:object!`;
- `model_artifact_id:string!`, `threshold_artifact_id:string!`;
- `similarity_before:number!`, `similarity_after:number!`;
- `accepted_before:boolean!`, `accepted_after:boolean!`;
- `success_from_reject:boolean!`, `elapsed_ms:number!`, `status:enum!`.

Optional fields: `epsilon:number?`, `alpha:number?`, `steps:integer?`, `queries_used:integer?`, `l0:number?`, `l2:number?`, `linf:number?`, `norm_space:string?`, `source_artifact_id:string?`, `target_enroll_artifact_id:string?`, `adversarial_artifact_id:string?`, `perturbation_artifact_id:string?`, `error_code:string?`.

Validation: `success_from_reject` equals `not accepted_before and accepted_after`; norms and elapsed time are non-negative; queries do not exceed configured budget; successful rows reference a canonical lossless adversarial artifact.

Producer: attack runner. Consumers: defense runner, evaluator, report.

## 5. Defense result row

Required fields: `defense_result_id:string!`, `run_id:string!`, `input_kind:enum!`, `defense_id:string!`, `defense_kind:enum!`, `defense_params:object!`, `similarity_before:number!`, `similarity_after:number!`, `accepted_before:boolean!`, `accepted_after:boolean!`, `elapsed_ms:number!`, `status:enum!`.

Exactly one of `attack_result_id` and `clean_pair_id` is required. `defense_success:boolean?` is non-null only for an attack input and means `accepted_before and not accepted_after`. Clean inputs instead contribute to FAR/FRR/TAR degradation. Optional fields include `output_artifact_id`, `checkpoint_artifact_id`, and `error_code`.

Producer: defense runner. Consumers: evaluator and report.

## 6. Authentication session record

Required fields: `session_id:string!`, `subject_token:string!`, `purpose:string!`, `capture_group_id:string!`, `enroll_artifact_id:string!`, `probe_artifact_ids:array!`, `threshold_artifact_id:string!`, `decision:enum!`, `started_at:datetime!`, `latency_ms:number!`. Optional fields: `risk_score:number? [0,100]`, `ended_at:datetime?`, `transaction_context_hash:string?`.

Producer: face-auth application. Consumers: policy audit and optional demo.

## 7. Detection rule result

Required fields: `rule_result_id:string!`, `session_id:string!`, `rule_id:string!`, `rule_version:string!`, `hit:boolean!`, `score_delta:number!`, `evidence:object!`, `evaluated_at:datetime!`.

Detection results are evidence, not verification ground truth. Producer: detector. Consumers: risk aggregator and report.

## 8. Checkpoint metadata

Required fields: `checkpoint_id`, `model_id`, `architecture`, `weights_source`, `weights_license`, `dataset_id`, `preprocessing_id`, `config_sha256`, `git_commit`, `seed`, `framework_versions`, `file_sha256`, and `created_at`. Optional: `metrics`, `parent_checkpoint_id`.

Producer: trainer or model importer. Consumers: model loader, threshold calibration, run validator.

## 9. Experiment run manifest

Required fields: `schema_version`, `run_id`, `experiment_id`, `requirement_ids`, `status`, `config_sha256`, `git_commit`, `environment_sha256`, `seed`, `device`, `started_at`, `input_artifact_ids`, `output_artifact_ids`, and `reproduce_command`. Completed runs also require `ended_at`.

Producer: experiment runner. Consumers: registry, report, optional UI.

## 10. Image/artifact reference

Required fields: `artifact_id`, `kind`, `relative_uri`, `sha256`, `bytes`, `mime_type`, `sensitivity`, `encryption`, and `created_at`. Optional: `parent_artifact_id`, `retention_until`, `producer_run_id`.

Sensitive biometric artifacts must not use `sensitivity=public`. Producer: all pipeline stages. Consumers: artifact loader and release checker.

## 11. Aggregated result row

Required fields: `aggregate_id`, `run_id`, `metric_id`, `value`, `unit`, `denominator`, `group`, `n_seeds`, and `source_rows_sha256`. Recommended fields: `numerator`, `ci_low`, `ci_high`.

The denominator must be explicit. For example, conditional defense success uses successful pre-defense attacks as its denominator; population attack rate uses all eligible attack attempts.
