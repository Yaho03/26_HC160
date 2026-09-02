# EXPERIMENT PLAN — 실험 계획

| 항목 | 내용 |
|---|---|
| 문서명 | 실험 계획서 |
| 버전 | v1.0 |
| 상태 | 확정 |
| 최종 수정일 | 2026-08-23 |

---

## 1. 공통 run 기록

모든 실험은 목적, 가설, input manifest, split 역할, model/checkpoint, threshold,
attack/defense configuration, 통제 변수, metric, 비교 기준, output, reproduction command와
알려진 한계를 기록한다.

## 2. 실험 목록

| ID | 목적과 가설 | Input과 통제 조건 | 주요 output과 비교 기준 |
|---|---|---|---|
| EXP-DATA-001 | Dataset provenance를 고정하고 split 분리를 증명 | Dataset archive, manifest builder; model 없음 | 유효한 manifest, 금지 overlap 0건 |
| EXP-VER-001 | Clean verification baseline 확립 | Calibration과 untouched test pair; 고정 model/preprocessing | FAR, FRR, EER, ROC-AUC, TAR@FAR, threshold artifact |
| EXP-VER-002 | Classification과 verification 동작 차이 설명 | 동일 legacy LFW-10 source; 별도 metric | Accuracy를 verification으로 취급하지 않는 비교 report |
| EXP-ATK-001 | White-box targeted 취약성 측정 | Reject-before test pair; FGSM/PGD budget | Reject-to-accept ASR, norm, latency, CI |
| EXP-ATK-002 | Black-box 효율 측정 | 동일 eligible pair; 고정 query budget | ASR-query curve, elapsed time, budget 준수 |
| EXP-DEF-001 | 전처리 방어 비교 | Attack result와 clean pair | ASR 감소와 clean TAR/FRR delta |
| EXP-DEF-002 | 학습형 방어 일반화 평가 | 분리된 attack train/validation/test row | Held-out/adaptive ASR, clean metric |
| EXP-DEF-003 | Ensemble 가치 평가 | 고정 component output | 최강 단일 component 대비 개선과 latency cost |
| EXP-DET-001 | Squeezing detector 평가 | 균형 clean/attack validation·test row | Detector ROC-AUC, TPR/FPR, verification 영향 |
| EXP-TEMP-001 | 실제 temporal/replay 동작 평가 | Genuine 및 attack video session | Attack 종류별 detection, false positive, delay |
| EXP-GEN-001 | 방어 목적 generative purification 평가 | Held-out clean/attack pair; 고정 denoiser | ASR, clean TAR, identity drift, p95 latency |
| EXP-TRN-001 | Attack transferability 측정 | Source/target embedding model | Cross-model ASR matrix |
| EXP-PERF-001 | Runtime budget 확립 | 명시한 CPU/GPU와 batch size | p50/p95 latency, throughput, memory |
| EXP-REP-001 | 반복 가능성 검증 | Seed 3개 이상, clean checkout | Metric variance와 artifact/run-manifest 완전성 |

## 3. 필수 통제 조건

- Test 전에 threshold를 고정한다.
- 비교하는 attack은 동일한 eligible pair ID를 사용한다.
- Defense 비교는 동일한 attack artifact를 사용한다.
- Clean/adversarial preprocessing version이 일치한다.
- Load failure와 model error를 별도로 보고한다.
- Hardware, batch size, warm-up과 timing boundary를 기록한다.
- 최종 report에 numerator, denominator와 confidence interval을 포함한다.

## 4. 요구사항 추적 예시

| 요구사항 | 구현 대상 | Test 대상 | 실험 | Artifact/report |
|---|---|---|---|---|
| DATA-001 | `src/contracts`, dataset builder | schema/hash test | EXP-DATA-001 | dataset manifest |
| VER-001 | `src/evaluation/verification_calibration.py` | `tests/research/test_verification_calibration.py` | EXP-VER-001 | threshold artifact + clean report |
| ATK-001 | 중앙 evaluation | transition test | EXP-ATK-001 | attack result |
| DEF-001 | defense evaluator | clean/attack denominator test | EXP-DEF-001 | trade-off table |
| DEF-002 | experiment split validator | leakage regression | EXP-DEF-002 | held-out report |
| DET-001 | detection evaluator | detector/auth 분리 | EXP-DET-001 | detector ROC |
| EXP-001 | run-manifest builder | manifest validation | EXP-REP-001 | run manifest |
| SEC-001 | artifact policy | sensitive-file scan | EXP-DATA-001 | release checklist |

각 미완료 행은 `15_ISSUE_AND_PR_WORKFLOW.md`의 규칙에 따라 구현 또는 실험 이슈로
전환한다.

## 실험별 상세 설계

번호 문서로 다루기에 세부가 많은 실험은 `experiments/`에 별도 설계를 둔다.

- `EXP-DET-001`의 웹캠 계측 도구 설계: `experiments/EXP-DET-001-camera-squeeze-probe.md`
