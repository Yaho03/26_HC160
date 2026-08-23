# ISSUE AND PR WORKFLOW — 이슈·PR 운영 규칙

| 항목 | 내용 |
|---|---|
| 문서명 | 이슈·PR 운영 규칙 |
| 버전 | v1.0 |
| 상태 | 확정 |
| 최종 수정일 | 2026-08-23 |
| 목적 | 저장소 문서만 읽고도 일관된 이슈와 PR을 만들 수 있도록 작업 단위와 검증 규칙을 정의 |

---

## 1. 핵심 원칙

이 저장소의 작업은 별도 프롬프트나 구두 설명이 아니라 **저장소 문서**에서 시작한다.
사람과 AI 작업자는 동일하게 다음 순서를 따른다.

```text
README
→ docs/README.md
→ CONTRIBUTING.md
→ 해당 영역의 규범 문서
→ docs/13_IMPLEMENTATION_STATUS.md
→ 이 문서
→ GitHub 이슈 템플릿
→ PR 템플릿
```

이슈는 막연한 아이디어를 적는 곳이 아니다. 규범 문서의 요구사항, 구현 상태의 결손,
재현 가능한 버그 또는 승인된 실험 가설을 **검증 가능한 작업 단위**로 변환한 기록이다.

## 2. 문서 우선순위

문서가 충돌하면 다음 순서로 판단한다.

1. `docs/01_RESEARCH_REQUIREMENTS.md` — 요구사항과 유효성 기준
2. `docs/04_DATA_AND_ARTIFACT_CONTRACT.md` — 데이터·산출물 계약
3. `docs/05_FACE_VERIFICATION_SPEC.md` — 얼굴 검증 프로토콜
4. `docs/06_ATTACK_SPEC.md` — 공격 정의와 성공 판정
5. `docs/07_DEFENSE_AND_DETECTION_SPEC.md` — 방어·탐지 의미
6. `docs/08_EXPERIMENT_PLAN.md`, `docs/09_EVALUATION_METRICS.md` — 실험과 지표
7. `docs/11_SECURITY_ETHICS_AND_LIMITATIONS.md` — 보안·윤리·공개 제한
8. `docs/12_REPRODUCIBILITY_GUIDE.md`, `docs/14_LOCAL_RUNBOOK.md` — 재현·운영 절차
9. `docs/13_IMPLEMENTATION_STATUS.md` — 현재 구현 및 검증 상태
10. 날짜가 붙은 진행 보고서와 인수인계 문서 — 역사적 증거

역사 문서의 과거 결과는 보존하지만 새 규범 문서보다 우선하지 않는다.

## 3. 작업을 시작하기 전 읽을 문서

| 작업 종류 | 반드시 읽을 문서 |
|---|---|
| 데이터셋·split | `03_DATASET_AND_PREPROCESSING.md`, `04_DATA_AND_ARTIFACT_CONTRACT.md` |
| 모델·checkpoint | `04_DATA_AND_ARTIFACT_CONTRACT.md`, `05_FACE_VERIFICATION_SPEC.md` |
| 얼굴 verification | `05_FACE_VERIFICATION_SPEC.md`, `09_EVALUATION_METRICS.md` |
| 공격 | `06_ATTACK_SPEC.md`, `04_DATA_AND_ARTIFACT_CONTRACT.md` |
| 방어·탐지 | `07_DEFENSE_AND_DETECTION_SPEC.md`, `09_EVALUATION_METRICS.md` |
| 실험 | `08_EXPERIMENT_PLAN.md`, `12_REPRODUCIBILITY_GUIDE.md` |
| 실시간 인증 프로토타입 | `docs/face_auth/README.md`, `docs/face_auth/THREAT_MODEL.md` |
| 문서·상태 갱신 | `docs/README.md`, `13_IMPLEMENTATION_STATUS.md` |

## 4. 이슈 후보를 찾는 방법

다음 순서로 이슈 후보를 찾는다.

1. `docs/13_IMPLEMENTATION_STATUS.md`에서 `미구현`, `부분 구현`, `검증 필요` 항목을 찾는다.
2. `docs/01_RESEARCH_REQUIREMENTS.md`에서 자동화된 acceptance evidence가 없는 요구사항을 찾는다.
3. `docs/08_EXPERIMENT_PLAN.md`에서 실행 결과나 재현 증거가 없는 실험을 찾는다.
4. 코드와 `docs/04_DATA_AND_ARTIFACT_CONTRACT.md` 또는 JSON Schema의 불일치를 찾는다.
5. CI·테스트 실패 또는 재현 가능한 사용자 보고를 확인한다.
6. 선행 이슈가 끝났는지 확인한 뒤 우선순위를 결정한다.

다음 항목만으로는 이슈를 만들지 않는다.

- 근거 문서가 없는 기능 아이디어
- 재현 절차가 없는 추측성 버그
- 이미 완료됐지만 구현 상태 문서만 갱신되지 않은 작업
- 다른 이슈의 완료 조건에 포함되는 사소한 하위 작업
- 실제 사용자 생체정보나 승인되지 않은 공격 확장을 요구하는 작업

## 5. 요구사항 ID 규칙

기존 ID를 이슈 제목과 본문에서 그대로 사용한다.

| 접두어 | 영역 | 기준 문서 |
|---|---|---|
| `DATA-XXX` | 데이터셋·전처리·split | `01_RESEARCH_REQUIREMENTS.md` |
| `MODEL-XXX` | 모델·checkpoint·전처리 결합 | `01_RESEARCH_REQUIREMENTS.md` |
| `VER-XXX` | 얼굴 verification | `01_RESEARCH_REQUIREMENTS.md` |
| `ATK-XXX` | 적대적 공격 | `01_RESEARCH_REQUIREMENTS.md`, `06_ATTACK_SPEC.md` |
| `DEF-XXX` | 방어 | `01_RESEARCH_REQUIREMENTS.md`, `07_DEFENSE_AND_DETECTION_SPEC.md` |
| `DET-XXX` | 탐지·위험 신호 | `01_RESEARCH_REQUIREMENTS.md`, `07_DEFENSE_AND_DETECTION_SPEC.md` |
| `EXP-XXX` | 실험·재현성 | `08_EXPERIMENT_PLAN.md` |
| `RPT-XXX` | 보고서·지표 | `09_EVALUATION_METRICS.md` |
| `SEC-XXX` | 보안·개인정보·윤리 | `11_SECURITY_ETHICS_AND_LIMITATIONS.md` |
| `UI-XXX` | 선택적 데모 UI | `10_DEMO_AND_REPORT_FLOW.md` |
| `FR-XXX`, `ARCH-XXX`, `DEP-XXX`, `PERF-XXX` | 세션 기반 얼굴 인증 prototype의 기존 backlog ID | `docs/face_auth/BACKLOG.md` |

`docs/face_auth/BACKLOG.md`에는 역사적으로 `DOC`, `EVAL`, `SEC`, `ATK`, `EXP` ID도
존재한다. 이 값은 그대로 보존하되 최상위 요구사항과 혼동하지 않도록 이슈 본문에 출처를
반드시 적는다. 새 ID 체계는 별도 ADR 없이 확장하지 않는다.

새 요구사항 ID가 필요하면 이슈에서 임의로 만들지 않는다. 먼저 규범 문서 변경안을
제시하고 승인된 ID를 추가한 뒤 구현 이슈를 만든다.

## 6. 이슈 유형

### 6.1 Requirement

규범 문서에 정의된 요구사항을 구현한다. 제목은 다음 형식을 사용한다.

```text
[VER-003] 모델·전처리·threshold 버전 결합 검증
```

하나의 이슈는 하나의 관찰 가능한 결과를 가져야 한다. 구현, 해당 테스트, 필요한 문서
갱신은 같은 이슈에 포함한다.

### 6.2 Experiment

승인된 가설을 정해진 데이터와 지표로 검증한다.

```text
[EXP-ATK-002] PGD 공격 seed별 성공률과 분산 검증
```

실험 이슈에는 코드 작성보다 먼저 가설, 분모, split, ground truth, 통과 기준과 재현
정보가 있어야 한다.

### 6.3 Bug

문서 계약 또는 기대 동작과 실제 동작의 차이를 수정한다.

```text
[BUG] test split 절대 경로가 run manifest에 저장되는 문제
```

재현 절차, 기대 동작, 실제 동작, 영향 범위와 회귀 테스트가 필수다.

### 6.4 Task

새 기능이 아닌 설정, CI, 리팩터링, 문서 동기화 같은 작업이다.

```text
[TASK] 계약 검증 테스트를 pull request CI에 연결
```

## 7. 이슈 분해 기준

다음 조건을 모두 만족하면 하나의 이슈로 본다.

- 한 문장으로 완료 결과를 설명할 수 있다.
- 하나의 PR에서 리뷰할 수 있다.
- 완료 조건을 자동 테스트 또는 재현 명령으로 검증할 수 있다.
- 선행 조건과 제외 범위가 명확하다.
- 관련 코드와 문서 변경을 함께 끝낼 수 있다.

다음 경우에는 이슈를 나눈다.

- 데이터 계약 변경과 여러 독립 소비자 구현이 동시에 필요한 경우
- CPU 검증과 장시간 GPU 실험의 완료 시점이 다른 경우
- 연구 metric 계산과 데모 UI 표시가 서로 독립적인 경우
- security-critical 정책 변경과 단순 화면 변경이 섞인 경우
- 한 PR에서 안전하게 리뷰하기 어려운 대규모 이동이 포함된 경우

이슈를 나눌 때 `blocked by`, `blocks` 관계를 명시한다.

## 8. 우선순위

| 우선순위 | 기준 | 예시 |
|---|---|---|
| `P0-critical` | 보안 결론을 뒤집거나 데이터 유출·split 누수·결과 훼손 가능 | test/calibration 혼용, 민감 산출물 커밋 |
| `P1-high` | 핵심 요구사항·재현성·데모 흐름을 막음 | threshold provenance 누락, 계약 불일치 |
| `P2-medium` | 핵심 흐름은 동작하지만 신뢰성·사용성이 부족 | 오류 메시지, 보조 지표, 문서 보완 |
| `P3-low` | 선택적 개선·연구 확장 | 추가 시각화, 비필수 최적화 |

우선순위가 같으면 다음 순서로 처리한다.

```text
보안·데이터 무결성
→ 실험 결론의 유효성
→ 재현성
→ 핵심 데모 차단 요소
→ 유지보수와 편의 기능
```

### 8.1 권장 label 체계

GitHub에 실제 label이 없으면 관리자가 먼저 생성한다. 존재하지 않는 label을 이슈 본문에
있는 것처럼 기록하지 않는다.

| 분류 | 권장 label |
|---|---|
| Type | `feature`, `bug`, `task`, `experiment`, `documentation`, `refactor` |
| Priority | `P0-critical`, `P1-high`, `P2-medium`, `P3-low` |
| Component | `data`, `model`, `verification`, `attack`, `defense`, `detection`, `face-auth`, `report`, `infra` |
| Risk | `security-critical`, `privacy`, `data-contract`, `reproducibility` |

하나의 이슈에는 원칙적으로 Type 1개, Priority 1개와 필요한 Component/Risk label을 붙인다.
Milestone은 일정 이름만 보고 고르지 않고 해당 milestone의 완료 산출물과 이슈 완료 조건이
일치할 때만 지정한다.

## 9. 이슈 본문 필수 항목

모든 이슈에는 다음 내용이 있어야 한다.

1. 배경과 문제
2. 요구사항 ID와 출처 문서
3. 구현 또는 실험 범위
4. 명시적인 제외 범위
5. 완료 조건
6. 검증 명령 또는 실험 방법
7. 변경이 예상되는 계약·스키마·문서
8. 선행·후속 이슈
9. 보안·윤리·민감 데이터 영향

완료 조건은 “구현한다”가 아니라 관찰 가능한 문장으로 쓴다.

나쁜 예:

```text
- 공격 결과 검증 기능을 구현한다.
```

좋은 예:

```text
- schema version이 지원 범위를 벗어나면 validator가 실패 코드와 필드 경로를 반환한다.
- 정상 fixture와 잘못된 fixture를 포함한 자동 테스트가 통과한다.
- 변경된 필드가 docs/04_DATA_AND_ARTIFACT_CONTRACT.md와 JSON Schema에 동일하게 정의된다.
```

## 10. 이슈 준비 완료 기준

다음 항목이 모두 충족되기 전에는 구현을 시작하지 않는다.

- [ ] 출처 요구사항 또는 재현 가능한 문제가 있다.
- [ ] 범위와 제외 범위가 구분되어 있다.
- [ ] 완료 조건이 측정 가능하다.
- [ ] 필요한 데이터·모델·threshold·policy 버전을 알고 있다.
- [ ] 선행 이슈가 해결되었거나 명시되어 있다.
- [ ] 민감 데이터와 라이선스 영향을 확인했다.
- [ ] 필요한 문서·스키마 갱신 범위를 확인했다.

## 11. 브랜치와 커밋

브랜치는 이슈에서 만든다.

```text
feat/VER-003-bind-threshold-version
fix/reject-absolute-artifact-path
exp/EXP-ATK-002-pgd-seed-evaluation
docs/issue-pr-workflow
chore/contract-ci
```

커밋 제목은 한국어로 작성하고 Conventional Commit 타입을 사용한다.

```text
feat(verification): VER-003 threshold provenance 검증 추가
fix(contract): 절대 artifact 경로 거부
test(attack): PGD seed 회귀 테스트 추가
docs: 이슈 추적 규칙 보완
```

`main`에 직접 push하지 않는다.

## 12. 변경별 문서 동기화

| 변경 영역 | 반드시 확인할 문서·계약 |
|---|---|
| dataset·split | `03_DATASET_AND_PREPROCESSING.md`, manifest schema |
| CSV·JSON 필드 | `04_DATA_AND_ARTIFACT_CONTRACT.md`, `schemas/` |
| model·preprocessing·threshold | `05_FACE_VERIFICATION_SPEC.md`, model config, run manifest |
| 공격 성공 판정·budget | `06_ATTACK_SPEC.md`, attack schema |
| 방어·탐지 의미 | `07_DEFENSE_AND_DETECTION_SPEC.md`, defense schema |
| 지표·분모 | `09_EVALUATION_METRICS.md`, metric 테스트 |
| 실험 절차·결과 | `08_EXPERIMENT_PLAN.md`, `13_IMPLEMENTATION_STATUS.md` |
| 보안·데이터 공개 | `11_SECURITY_ETHICS_AND_LIMITATIONS.md` |
| 실행 명령 | `GETTING_STARTED.md`, `12_REPRODUCIBILITY_GUIDE.md`, `14_LOCAL_RUNBOOK.md` |
| 세션 인증 프로토타입 | `docs/face_auth/` 관련 계약·위협 모델 |

문서를 갱신하지 않았다면 PR 본문에 그 이유를 적는다.

## 13. PR 작성과 머지 조건

PR은 이슈의 완료 증거다. 다음 조건을 만족해야 한다.

- 관련 이슈를 `Closes #NNN`으로 연결한다.
- 이슈의 범위와 PR 변경 범위가 일치한다.
- 요구사항 ID, 구현 파일, 테스트, 실험·결과 위치를 연결한다.
- 실행한 명령과 실제 결과를 적는다.
- 표본 수와 실패·오류·제외 건수를 분리한다.
- 데이터, 모델, config, seed, threshold, policy 버전을 기록한다.
- 계약 변경 시 producer와 consumer 및 이전 버전 호환성을 확인한다.
- 민감한 얼굴·임베딩·template·checkpoint가 포함되지 않았는지 확인한다.
- 관련 문서와 구현 상태 문서를 갱신한다.
- CI 또는 해당 로컬 검증이 통과한다.

결과가 기대와 다르더라도 숨기지 않는다. 실험 실패는 코드 실패와 구분하여 기록한다.

## 14. AI 작업자 규칙

AI가 이 저장소에서 이슈 초안을 만들 때도 동일한 규칙을 따른다.

1. 먼저 이 문서의 1~3절에 지정된 문서를 읽는다.
2. `13_IMPLEMENTATION_STATUS.md`와 테스트 결과를 근거로 후보를 찾는다.
3. 기존 GitHub 이슈와 중복 여부를 확인할 수 없다면 그 제한을 표시한다.
4. 존재하지 않는 요구사항 ID, 성능 수치 또는 실험 결과를 만들지 않는다.
5. 하나의 이슈가 너무 크면 의존 관계가 있는 여러 이슈로 제안한다.
6. 사용자 요청이 이슈 작성뿐이면 코드·문서·Git 상태를 변경하지 않는다.
7. 구현을 요청받으면 이슈의 완료 조건과 문서 동기화 범위까지 함께 처리한다.

이 규칙을 읽은 뒤에는 별도의 “이슈 작성 프롬프트”가 없어도 다음 요청을 수행할 수
있어야 한다.

```text
저장소 문서를 기준으로 다음 우선순위 이슈 초안을 작성해줘.
```

## 15. 최종 추적 형식

기능 또는 실험 완료 시 다음 연결이 남아야 한다.

```text
요구사항 ID
→ GitHub 이슈
→ 브랜치와 PR
→ 구현 코드
→ 설정과 데이터 계약
→ 자동 테스트
→ 실험 run ID와 산출물
→ 결과 보고서
→ docs/13_IMPLEMENTATION_STATUS.md
```

연결 중 하나라도 빠지면 구현은 존재하더라도 프로젝트 차원에서는 완료로 보지 않는다.
