# DATA AND ARTIFACT CONTRACT — 데이터·산출물 계약

| 항목 | 내용 |
|---|---|
| 문서명 | 데이터 및 산출물 계약서 |
| 버전 | v1.0 |
| 상태 | 확정 |
| 최종 수정일 | 2026-08-23 |

---

## 1. 공통 규칙

- 모든 record는 `schema_version`을 포함한다.
- `!`는 필수·non-null, `?`는 optional 또는 nullable을 뜻한다.
- ID는 안정적인 opaque string이며 display name을 식별자로 사용하지 않는다.
- 시간은 ISO-8601 UTC를 사용한다.
- Duration은 millisecond 단위이며 필드명은 `_ms`로 끝난다.
- `norm_space`에서 다르게 정하지 않으면 image perturbation norm은 `[0, 1]` pixel scale을 사용한다.
- 완료된 record는 immutable이다. 수정할 때는 새 ID를 만든다.
- 장비 절대 경로는 유효하지 않다.

`/schemas`의 JSON Schema가 기계 판독 가능한 최소 검증 기준이며, 이 문서는 JSON
Schema만으로 표현하기 어려운 의미 규칙을 정의한다.

## 2. Dataset manifest row

| 필드 | 타입 | 필수 | Null | 검증 |
|---|---|---:|---:|---|
| `schema_version` | string | yes | no | 지원하는 major version |
| `dataset_id` | string | yes | no | Immutable snapshot ID |
| `sample_id` | string | yes | no | Dataset 안에서 unique |
| `identity_token` | string | yes | no | 가명 값이며 display name이 아님 |
| `relative_uri` | string | yes | no | 상대 경로만 허용 |
| `media_sha256` | string | yes | no | 소문자 16진수 64자 |
| `split` | enum | yes | no | `train`, `calibration`, `development`, `test` |
| `width_px`, `height_px` | integer | yes | no | 양수 |
| `license_id` | string | yes | no | Dataset license/policy 참조 |
| `source_uri` | string | no | yes | 사용할 수 있으면 공개 출처 페이지 |

Producer는 dataset builder이며 consumer는 pair builder, trainer와 evaluator다.

## 3. Verification pair row

필수 필드는 `schema_version:string!`, `pair_id:string!`, `protocol_id:string!`,
`left_sample_id:string!`, `right_sample_id:string!`, `same_identity:boolean!`,
`split:enum!`이다. Optional 필드는 `fold:integer?`, `pair_group_id:string?`다.

Left와 right는 달라야 하고 두 sample ID는 같은 dataset에서 해석되어야 한다. 순서를
바꾼 중복 pair를 허용하지 않으며 pair split은 protocol manifest와 일치해야 한다.

Producer는 pair builder이며 consumer는 clean evaluator와 attack runner다.

## 4. Attack result row

필수 필드:

- `attack_result_id:string!`, `run_id:string!`, `pair_id:string!`
- `attack_id:string!`, `threat_model:enum!`, `attack_params:object!`
- `model_artifact_id:string!`, `threshold_artifact_id:string!`
- `similarity_before:number!`, `similarity_after:number!`
- `accepted_before:boolean!`, `accepted_after:boolean!`
- `success_from_reject:boolean!`, `elapsed_ms:number!`, `status:enum!`

Optional 필드는 `epsilon`, `alpha`, `steps`, `queries_used`, `l0`, `l2`,
`linf`, `norm_space`, source/target/adversarial/perturbation artifact ID와
`error_code`다.

`success_from_reject`는 `not accepted_before and accepted_after`와 같아야 한다.
Norm과 elapsed time은 음수가 될 수 없고 query는 budget을 넘을 수 없다. 성공 row는 정본
lossless adversarial artifact를 참조해야 한다.

Producer는 attack runner이며 consumer는 defense runner, evaluator와 report다.

## 5. Defense result row

필수 필드는 `defense_result_id`, `run_id`, `input_kind`, `defense_id`,
`defense_kind`, `defense_params`, 전후 similarity·accepted, `elapsed_ms`와
`status`다.

`attack_result_id`와 `clean_pair_id` 중 정확히 하나만 존재해야 한다.
`defense_success:boolean?`는 attack input에서만 non-null이며
`accepted_before and not accepted_after`를 뜻한다. Clean input은 FAR/FRR/TAR 저하
계산에 사용한다. Optional 필드는 output artifact, checkpoint artifact와 error code다.

Producer는 defense runner이며 consumer는 evaluator와 report다.

## 6. Authentication session record

필수 필드는 `session_id`, `subject_token`, `purpose`, `capture_group_id`,
`enroll_artifact_id`, `probe_artifact_ids`, `threshold_artifact_id`, `decision`,
`started_at`, `latency_ms`다. Optional 필드는 `risk_score [0,100]`,
`ended_at`, `transaction_context_hash`다.

Producer는 face-auth application이며 consumer는 policy audit와 optional demo다.

## 7. Detection rule result

필수 필드는 `rule_result_id`, `session_id`, `rule_id`, `rule_version`, `hit`,
`score_delta`, `evidence`, `evaluated_at`이다. Detection result는 verification
ground truth가 아니라 evidence다.

Producer는 detector이며 consumer는 risk aggregator와 report다.

## 8. Checkpoint metadata

필수 필드는 `checkpoint_id`, `model_id`, `architecture`, `weights_source`,
`weights_license`, `dataset_id`, `preprocessing_id`, `config_sha256`,
`git_commit`, `seed`, `framework_versions`, `file_sha256`, `created_at`이다.
Optional 필드는 `metrics`, `parent_checkpoint_id`다.

## 9. Experiment run manifest

필수 필드는 `schema_version`, `run_id`, `experiment_id`, `requirement_ids`,
`status`, `config_sha256`, `git_commit`, `environment_sha256`, `seed`,
`device`, `started_at`, input/output artifact ID와 `reproduce_command`다.
완료된 run에는 `ended_at`도 필요하다.

## 10. Image/artifact reference

필수 필드는 `artifact_id`, `kind`, `relative_uri`, `sha256`, `bytes`,
`mime_type`, `sensitivity`, `encryption`, `created_at`이다. Optional 필드는
`parent_artifact_id`, `retention_until`, `producer_run_id`다.

민감 biometric artifact는 `sensitivity=public`을 사용할 수 없다.

## 11. Aggregated result row

필수 필드는 `aggregate_id`, `run_id`, `metric_id`, `value`, `unit`,
`denominator`, `group`, `n_seeds`, `source_rows_sha256`다. `numerator`,
`ci_low`, `ci_high` 사용을 권장한다.

분모는 반드시 명시한다. 예를 들어 conditional defense success의 분모는 방어 전 성공한
attack이고, population attack rate의 분모는 모든 eligible attack attempt다.

## 12. 계약 변경 규칙

- 필드 추가는 호환 가능한 minor schema version 변경으로 처리한다.
- 필드 제거, 의미 또는 타입 변경은 major version을 올린다.
- JSON Schema 변경 시 이 문서와 producer·consumer test를 함께 수정한다.
- PR에는 이전 version 호환 또는 migration 정책을 적는다.
- 세부 PR 확인 항목은 `15_ISSUE_AND_PR_WORKFLOW.md`를 따른다.
