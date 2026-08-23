# RESEARCH REQUIREMENTS — 연구 요구사항

| 항목 | 내용 |
|---|---|
| 문서명 | 연구 요구사항 정의서 |
| 버전 | v1.0 |
| 상태 | 확정 |
| 최종 수정일 | 2026-08-23 |

---

## 1. 요구사항 강도

- **MUST**: 결과를 유효한 것으로 인정하기 위한 필수 조건
- **SHOULD**: 승인된 ADR에서 예외를 명시하지 않는 한 충족해야 하는 조건
- **MAY**: 선택적 확장

## 2. 요구사항

### 2.1 데이터와 전처리

| ID | 강도 | 요구사항 | 완료 증거 |
|---|---|---|---|
| DATA-001 | MUST | 모든 dataset snapshot은 변경 불가능한 manifest와 checksum을 가진다. | Schema-valid manifest이며 참조 파일의 hash가 모두 일치한다. |
| DATA-002 | MUST | Train, calibration, development와 test 역할을 명시한다. | 자동 split-overlap test가 통과한다. |
| DATA-003 | MUST | 이미지 참조는 artifact 기준 상대 경로이며 장비 절대 경로를 사용하지 않는다. | Contract validation이 `/content`, home 및 drive 종속 경로를 거부한다. |
| DATA-004 | MUST | Dataset license, 출처와 재배포 제한을 기록한다. | Dataset metadata의 license 필드가 비어 있지 않다. |

### 2.2 모델과 verification

| ID | 강도 | 요구사항 | 완료 증거 |
|---|---|---|---|
| MODEL-001 | MUST | 모든 checkpoint는 architecture, 출처, license, dataset, configuration, Git commit, seed와 SHA-256을 기록한다. | Checkpoint sidecar validation이 통과한다. |
| VER-001 | MUST | Threshold는 calibration data로만 선택하고 test 평가 전에 고정한다. | Threshold artifact가 서로 겹치지 않는 calibration manifest를 가리킨다. |
| VER-002 | MUST | Clean verification은 FAR/FMR, FRR/FNMR, EER, ROC-AUC와 지원 가능한 FAR에서 TAR을 보고한다. | Metric test와 clean test report가 존재한다. |
| VER-003 | MUST | 모델별 preprocessing과 threshold version을 분리할 수 없도록 결합한다. | Run validation이 model/preprocessing/threshold ID 불일치를 거부한다. |

### 2.3 공격

| ID | 강도 | 요구사항 | 완료 증거 |
|---|---|---|---|
| ATK-001 | MUST | Targeted impersonation 성공은 reject-to-accept 전이로 정의한다. | `success_from_reject == !accepted_before && accepted_after`가 성립한다. |
| ATK-002 | MUST | 결과마다 threat model과 perturbation/query budget을 저장한다. | Attack contract validation이 통과한다. |
| ATK-003 | MUST | 정본 adversarial image는 lossless 형식을 사용한다. | PNG/tensor artifact hash가 있고 JPEG는 파생 변환으로 표시된다. |
| ATK-004 | SHOULD | 자체 구현을 smoke subset에서 reference implementation과 교차 검증한다. | 비교 run과 tolerance report가 존재한다. |

### 2.4 방어와 탐지

| ID | 강도 | 요구사항 | 완료 증거 |
|---|---|---|---|
| DEF-001 | MUST | 모든 방어를 attack input과 clean input 모두에서 평가한다. | 같은 report에 attack ASR과 clean TAR/FRR delta가 표시된다. |
| DEF-002 | MUST | 학습형 방어는 서로 겹치지 않는 train, validation 및 test attack sample을 사용한다. | Leakage test가 통과하고 split ID가 기록된다. |
| DEF-003 | MUST | 학습형 또는 미분 가능한 방어는 adaptive attack으로 평가한다. | Adaptive-attack run이 방어된 checkpoint를 참조한다. |
| DEF-004 | MUST | Transform, trained, ensemble, temporal 및 detector component의 의미를 구분한다. | Defense metadata에 interface type이 기록된다. |
| DET-001 | MUST | Detector alert를 인증 정확도로 취급하지 않는다. | Detection metric과 authentication decision을 별도로 보고한다. |

### 2.5 실험과 보고서

| ID | 강도 | 요구사항 | 완료 증거 |
|---|---|---|---|
| EXP-001 | MUST | 완료된 모든 run은 configuration, code, environment, seed, device, input, output과 reproduction command를 기록한다. | Run manifest validation이 통과한다. |
| EXP-002 | MUST | 모든 metric은 numerator, denominator, unit과 grouping을 포함한다. | Aggregate-result validation이 통과한다. |
| EXP-003 | MUST | 명시적 예외가 없으면 randomized experiment는 seed를 3개 이상 사용한다. | Report에 seed 수가 있거나 승인된 limitation이 연결된다. |
| RPT-001 | MUST | Report는 측정 사실, 한계와 추론을 구분한다. | Report checklist가 통과한다. |
| RPT-002 | MUST | 기존 공개 결과를 보존하고 조용히 재생성하지 않는다. | Legacy artifact hash가 변경되지 않는다. |

### 2.6 보안·윤리와 선택적 UI

| ID | 강도 | 요구사항 | 완료 증거 |
|---|---|---|---|
| SEC-001 | MUST | Raw face, embedding과 checkpoint를 민감 artifact로 취급한다. | Git policy check와 artifact sensitivity label이 통과한다. |
| SEC-002 | MUST | 이 시스템이 운영 금융 인증 시스템이 아님을 report에 명시한다. | 필수 면책 문구가 존재한다. |
| SEC-003 | MUST | 별도 승인 전 생성형 AI 작업을 방어 목적 purification으로 제한한다. | Experiment registry scope check가 통과한다. |
| UI-001 | MAY | UI는 완료되고 검증된 run을 표시할 수 있다. | 표시값이 artifact hash와 run ID에 일치한다. |
| UI-002 | UI가 있으면 MUST | UI와 연구 pipeline을 분리한다. | UI가 threshold를 변경하거나 완료된 run을 덮어쓸 수 없다. |

## 3. 추적 규칙

구현된 모든 요구사항은 다음 연결을 가져야 한다.

```text
요구사항 ID → 구현 → 설정 → 자동 테스트
            → 실험 ID → run/artifact → 보고서 절
```

초기 추적표는 `08_EXPERIMENT_PLAN.md`에서 관리한다. 이후에는 run manifest에서
자동 생성하는 것을 목표로 한다. 이슈와 PR 연결 방식은
`15_ISSUE_AND_PR_WORKFLOW.md`를 따른다.
