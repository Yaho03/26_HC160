# FACE AUTHENTICATION BACKLOG — 얼굴 인증 백로그

이 문서의 항목은 GitHub 이슈 생성을 승인받기 전까지 검토 가능한 요구사항 사양으로
유지한다. 이슈로 전환할 때는 `../15_ISSUE_AND_PR_WORKFLOW.md`를 따른다.

## 1. 상태 정의

- **구현됨**: Code와 자동 contract test 존재
- **부분 구현**: Code path는 있지만 외부 artifact, target-device calibration 또는 필수 experiment 누락
- **대기**: 완전한 구현 또는 evidence가 없음

## 2. 구현 snapshot — commit `c4c1e15`

| ID | 상태 | 증거 또는 남은 차단 요소 |
|---|---|---|
| DOC-001 | 구현됨 | 범위, API contract, threat model, module map과 limitation 문서화 |
| EVAL-001 | 부분 구현 | Legacy limitation 문서화. 과거 주장을 수정 held-out data에서 재실행하지 않음 |
| DEP-001 | 부분 구현 | Python dependency version 고정. 외부 model checksum과 clean-checkout bootstrap 미완료 |
| EXP-001 | 부분 구현 | Dataset manifest와 leakage validator 존재. 최종 subject/session/device-disjoint dataset 미생성 |
| ARCH-001 | 구현됨 | State, gate, policy와 fail-closed integration test 통과 |
| SEC-001 | 부분 구현 | Enrollment 분리·token lifecycle 존재. Encrypted persistent template storage 없음 |
| ATK-001 | 구현됨 | Manifest 기반 replay/insertion runner와 test 존재 |
| FR-101 | 구현됨 | Nonce, challenge, state, expiry와 replay test |
| FR-102 | 구현됨 | Webcam/video 공통 `FrameSource`, bounded latest-frame buffer와 drop test |
| FR-103 | 구현됨 | Quality failure를 retryable로 처리하고 model error와 구분 |
| FR-104 | 구현됨 | MTCNN이 모든 detection을 유지하고 multi-face evidence를 fail closed |
| FR-105 | 부분 구현 | Multi-frame template/verification, FaceNet score export와 threshold calibration 존재. 최종 승인 identity threshold 대기 |
| FR-106 | 구현됨 | 필수 failure/error/NOT_EVALUATED 상태에서 verify 금지 |
| FR-107 | 구현됨 | Token context, expiry, replay와 one-time consume test |
| FR-108 | 구현됨 | Webcam preview, progress, cancel, headless override와 structured error test |
| EXP-002 | Synthetic test 구현 | Genuine/impostor/multi-face/quality/error 자동화. Target-device evidence 대기 |
| FR-201 | 부분 구현 | TorchScript/ONNX PAD adapter, report schema, immutable output, manifest validator, evaluator, APCER/BPCER·species metric 존재. 검증 checkpoint와 held-out physical data 없음 |
| FR-202 | 부분 구현 | Random challenge logic, 실시간 동작 지시와 표시 이후 프레임 결합 존재. Physical replay/accessibility 평가 대기 |
| FR-203 | 부분 구현 | Tracking·template continuity 존재. Held-out switch/occlusion 연구 대기 |
| EXP-201 | 대기 | Physical print/screen/replay dataset과 report 없음 |
| EXP-202 | 부분 구현 | Synthetic insertion scenario 통과. Physical/person-switch matrix와 delay report 없음 |
| FR-301 | 부분 구현 | Optional transform-consistency veto 존재. Clean/adaptive calibration 대기 |
| FR-302 | 대기 | 올바른 held-out adversarial-training 재실행 미구현 |
| EXP-301 | 대기 | Digital/screen/print transfer 비교 없음 |
| PERF-001 | 대기 | 명시 hardware의 FPS/drop/P95 report 없음 |
| SEC-301 | 대기 | Mobile attestation provider boundary만 문서화 |

## 3. Phase 0

| ID | 작업 | 완료 조건 |
|---|---|---|
| DOC-001 | Scope, trust boundary, threat model | P0/P1과 prototype claim 합의 |
| EVAL-001 | 기존 100%/98.1% 주장 재검증 | Leakage와 clean false positive 보고 |
| DEP-001 | 재현 가능한 환경 | Import, version, checksum/seed 지침 기록 |
| EXP-001 | Disjoint dataset manifest | Subject/session/device/attack split 기록 |
| ARCH-001 | State·gate contract | Transition과 fail-closed test 통과 |
| SEC-001 | Enrollment/template/token lifecycle | Registration 분리, replay/context test 통과 |
| ATK-001 | Scenario runner | 결정적인 video/insertion/protocol scenario |

## 4. Phase 1

| ID | 작업 | 완료 조건 |
|---|---|---|
| FR-101 | Session, nonce, challenge, expiry | Duplicate·expiry behavior test |
| FR-102 | Webcam/video source와 bounded queue | 하나의 pipeline과 drop accounting |
| FR-103 | Quality gate | Retryable quality failure와 error 분리 |
| FR-104 | Detection, alignment, multi-face gate | 여러 얼굴 중 하나를 조용히 선택하지 않음 |
| FR-105 | Multi-frame enrollment와 verification | Validation threshold와 version 기록 |
| FR-106 | Policy engine | 필수 failure/error 상태에서 verify 불가 |
| FR-107 | One-time verification token | Replay, context change와 expiry 거부 |
| FR-108 | Live camera capture UX | Preview/progress, cancel, headless와 camera failure test |
| EXP-002 | Baseline E2E scenario | Genuine/impostor/multi-face/blur 자동화 |

## 5. Phase 2

| ID | 작업 | 완료 조건 |
|---|---|---|
| FR-201 | Passive PAD와 calibration | Print/screen 종류별 APCER/BPCER |
| FR-202 | Random active liveness | Pre-challenge action과 replay 거부 |
| FR-203 | Tracking과 identity continuity | Occlusion reacquisition과 person switch test |
| EXP-201 | Print/screen/replay experiment | 종류별 결과 보고 |
| EXP-202 | Mid-frame/person-switch experiment | Insertion length별 detection과 delay 보고 |

## 6. Phase 3

| ID | 작업 | 완료 조건 |
|---|---|---|
| FR-301 | Transform-consistency secondary inspection | Embedding dispersion과 clean calibration |
| FR-302 | 올바른 adversarial-training split | Holdout ASR, FAR와 FRR 보고 |
| EXP-301 | Digital/screen/print 비교 | Transfer result 분리 보고 |
| PERF-001 | Runtime 최적화 | FPS, drop과 P95 latency 보고 |
| SEC-301 | Attestation provider contract | Mock adapter와 mobile extension boundary |

## 7. 이슈 전환 규칙

- 상태가 “대기” 또는 “부분 구현”이고 선행 조건이 준비된 항목만 후보로 삼는다.
- 한 이슈에서 구현과 evidence 확보 시점이 다르면 구현 이슈와 experiment 이슈로 나눈다.
- 제목에 기존 ID를 유지하고 완료 조건을 그대로 복사하지 말고 측정 가능한 checklist로 바꾼다.
- 해당 row의 status를 바꾸려면 구현·test·experiment evidence를 PR에 연결한다.
- GitHub의 기존 이슈와 중복 여부를 확인한 뒤 생성한다.
