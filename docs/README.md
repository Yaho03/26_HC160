# HC160 문서 안내

이 디렉터리는 연구 명세, 실행 계약, 의사결정 기록을 한곳에서 관리한다. 문서 번호 순서대로 읽으면 프로젝트 목적에서 재현 절차까지 이어진다.

## 권장 읽기 순서

1. `00_PROJECT_OVERVIEW.md` — 목표, 범위, 성공 조건
2. `01_RESEARCH_REQUIREMENTS.md` — 검증 가능한 요구사항
3. `02_SYSTEM_ARCHITECTURE.md` — 연구 파이프라인과 경계
4. `03_DATASET_AND_PREPROCESSING.md` — 데이터 분할과 전처리
5. `04_DATA_AND_ARTIFACT_CONTRACT.md` — 파일 및 레코드 계약
6. `05_FACE_VERIFICATION_SPEC.md` — 검증 프로토콜
7. `06_ATTACK_SPEC.md` — 공격 정의와 성공 판정
8. `07_DEFENSE_AND_DETECTION_SPEC.md` — 방어 및 탐지 규칙
9. `08_EXPERIMENT_PLAN.md` — 단계별 실험 계획
10. `09_EVALUATION_METRICS.md` — 지표와 분모 규칙
11. `10_DEMO_AND_REPORT_FLOW.md` — 연구 결과를 데모로 연결하는 방식
12. `11_SECURITY_ETHICS_AND_LIMITATIONS.md` — 안전, 윤리, 한계
13. `12_REPRODUCIBILITY_GUIDE.md` — 동일 결과 재현 절차
14. `13_IMPLEMENTATION_STATUS.md` — 실제 구현, 검증 증거, 미완료 항목
15. `14_LOCAL_RUNBOOK.md` — 로컬 실행, 검증, 문제 해결 절차
16. `15_ISSUE_AND_PR_WORKFLOW.md` — 요구사항을 이슈와 PR로 전환하는 협업 규칙

처음 실행하는 사람은 `GETTING_STARTED.md`, 용어가 필요한 경우 `GLOSSARY.md`를 함께 본다. JSON 레코드 형식은 `../schemas/`가 기계 판독 가능한 기준이다.

작업을 시작하거나 이슈·PR을 작성하려는 사람과 AI 작업자는 반드시
`../CONTRIBUTING.md`와 `15_ISSUE_AND_PR_WORKFLOW.md`를 읽는다. 별도의 이슈 작성
프롬프트보다 저장소의 규범 문서와 이 운영 규칙을 우선한다.

## 문서 상태

- 위 번호 문서와 `schemas/`는 앞으로의 규범적 명세다.
- `adr/`는 되돌리기 어려운 설계 결정과 근거를 보존한다.
- `history/`는 특정 시점의 상태와 해시를 보존한다.
- 날짜가 포함된 기존 진행 보고서와 기존 공격·방어 인수인계 문서는 과거 실험 증거다. 새 명세보다 우선하지 않으며 삭제하거나 소급 수정하지 않는다.
- `face_auth/`는 구현된 세션 기반 인증 프로토타입의 별도 계약과 실행 문서다. `face_auth/README.md`에서 시작한다.
- 구현 상태는 `13_IMPLEMENTATION_STATUS.md`가 기준이며, 코드 구현과 실험 검증 완료를 구분한다.
- 이슈 분해, 우선순위, 완료 조건과 PR 머지 조건은 `15_ISSUE_AND_PR_WORKFLOW.md`가 기준이다.
