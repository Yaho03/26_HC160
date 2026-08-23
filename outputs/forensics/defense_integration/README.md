# 방어 결과 연동 감사

## 확정 집계

전처리 방어는 `evaluation_basis=artifact_reload`이며, 각 기법에서
`attack_success_before_defense=true`인 171건을 분모로 사용한다.

| 방어 | 우회 | 우회율 | 방어 성공 | 방어 성공률 |
|---|---:|---:|---:|---:|
| JPEG | 171/171 | 100.00% | 0/171 | 0.00% |
| Gaussian smoothing | 4/171 | 2.34% | 167/171 | 97.66% |
| Bit-depth | 131/171 | 76.61% | 40/171 | 23.39% |

적대적 학습은 `evaluation_basis=legacy_record`이며, 원 평가에서 공격 성공으로
기록된 212건을 분모로 사용한다.

| 방어 | 우회 | 우회율 | 방어 성공 | 방어 성공률 |
|---|---:|---:|---:|---:|
| Adversarial training | 4/212 | 1.89% | 208/212 | 98.11% |

두 표는 평가 기준과 분모가 달라 합산하거나 직접 순위를 매기지 않는다.

공개 산출물의 source/target identity와 계정 ID는 안정적인 pseudonym으로 바꾸고,
이미지 경로는 `artifact_<hash>` 참조로 치환했다. 원본 인물명과 로컬·Kaggle 경로는
대시보드 전달 파일에 포함하지 않는다.

## 산출물

- `defense_results_by_sample_id.csv`: 공격 원장+방어 결과 long form, 848행
- `defense_evaluation_sessions.csv`: 212개 방어평가 세션 wide form
- `preprocessing_defense_summary.csv`: 전처리 3종 분리 집계
- `adversarial_training_defense_summary.csv`: 적대적 학습 분리 집계
- `defense_integration_overview.json`: 대시보드용 평가 그룹 JSON
- `defense_join_audit.json`: 기존 2,000건 코호트와의 ID 계보 감사

상위 `outputs/forensics/`에는 비식별 세션 데이터와 함께
`attack_similarity_panel.png`, `attack_family_overview.png` 시각화를 생성한다.

## 조인 상태

- 재수령한 `defense_results_by_sample_id.csv`는 공격 원장 필드와 방어 결과가 이미
  `sample_id` 기준으로 합쳐진 848행의 self-contained handoff다.
- 방어평가 코호트: 212개 `sample_id`, 기법별 212행
- 포렌식 CSV: 2,000행, 2,000개 `attempt_id`
- 정확한 `sample_id == attempt_id` 일치: **0/212**
- 설정 기반 후보도 87개는 후보가 없고 125개는 후보가 2개라 자동 대체 조인이 불가능하다.

따라서 방어기법별 집계는 재수령 파일을 정식 방어평가 조인 결과로 확정한다. 기존
2,000행 포렌식 로그와는 실험 코호트와 ID 계보가 다르므로 억지로 합치지 않고 별도
산출물로 유지한다.

## 입력 해시

- `defense_results_by_sample_id.csv`: `965ed2277e21892461213f8446f387333d28984ee602f98b77b482dc5bcddb24`
- `defense_handoff.validation.json`: `77ff984ff96b08d7177560dfaf1d501a75b4351affeb94e663bfea9e296aa721`
- `attack_sessions.csv`: `62f27a2b0ab80087cbbb3d64473089be9fdfb40844dd67452dbf7002939852ae`
