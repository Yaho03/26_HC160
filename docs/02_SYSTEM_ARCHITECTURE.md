# SYSTEM ARCHITECTURE — 시스템 아키텍처

| 항목 | 내용 |
|---|---|
| 문서명 | 시스템 아키텍처 설계서 |
| 버전 | v1.0 |
| 상태 | 확정 |
| 최종 수정일 | 2026-08-23 |

---

## 1. 아키텍처 원칙

재현 가능한 연구 pipeline이 정본이다. 대화형 demo와 향후 persistence layer는 변경
불가능한 run artifact를 읽는 consumer다.

```text
Dataset + license metadata
        → immutable sample manifest
        → pair 및 split manifest
        → model/checkpoint + preprocessing version
        → calibration threshold artifact
        → clean baseline
        → attack run
        → defense/detection run
        → 중앙 evaluation
        → 표, figure와 report
        → 선택적 read-only demo
```

## 2. 연구 계층

| 계층 | 책임 | 하면 안 되는 일 |
|---|---|---|
| Dataset | Manifest와 고정 split을 생성한다. | Threshold를 선택하거나 test label을 바꾼다. |
| Model | 버전이 지정된 classifier 또는 embedding model을 불러온다. | 로컬 장비에서 path나 threshold를 추론한다. |
| Verification | Pair를 embedding하고 threshold를 calibration하여 accept/reject를 판정한다. | Test 결과로 parameter를 조정한다. |
| Attack | 제한된 adversarial probe와 attack metadata를 생성한다. | Verification metric을 재정의한다. |
| Defense | Input 또는 model을 변환하고 defense result를 생성한다. | Clean reject를 defense success로 센다. |
| Detection | 의심 신호와 evidence를 생성한다. | Detection rate를 authentication accuracy라고 주장한다. |
| Evaluation | 모든 metric의 numerator와 denominator를 계산한다. | Source artifact를 변경한다. |
| Reporting | 검증된 aggregate를 렌더링한다. | 다른 정의로 숨은 metric을 다시 계산한다. |
| Demo | 선택된 검증 완료 run을 표시한다. | 완료된 run의 threshold/configuration을 변경한다. |

## 3. 현재 코드 매핑

| 경로 | 아키텍처 역할 |
|---|---|
| `src/datasets/` | Legacy 준비와 immutable manifest 탐색·검증·snapshot metadata |
| `src/training/` | Legacy classification training |
| `src/attacks/` | Legacy classification attack |
| `src/verification/` | Verification bridge |
| `src/verification/defenses/` | FaceNet verification defense prototype |
| `src/defenses/` | Legacy classification defense |
| `src/reports/` | Legacy report builder 및 향후 중앙 metric consumer |
| `src/face_auth/` | Session/policy/token reference prototype |
| `src/attack_scenarios/` | Manifest 기반 replay 및 frame-insertion video 생성 |
| `src/contracts/` | 외부 의존성이 없는 semantic artifact validation |
| `src/evaluation/` | 중앙 verification, attack, defense metric 및 EXP-VER-001 calibration |
| `src/experiments/` | Immutable run-manifest 표현 |
| `schemas/` | Dataset, score, threshold, run, attack, defense와 report 계약 |
| `outputs/` | 과거 summary 및 향후 immutable run directory |

## 4. 연구와 서비스의 경계

`src/face_auth` reference prototype은 session state, challenge, fail-closed gate policy와
일회용 token 소비를 제어한다. 이 모듈이 legacy experiment CSV를 운영 증거로 바꾸는 것은
아니다. Model gate는 버전이 있는 `GateResult`를 반환해야 하며, 연구 evaluation은 고정
dataset에서 model behavior를 독립적으로 측정한다.

향후 선택적 service는 다음을 제공할 수 있다.

- Session 생성과 evidence 제출
- Read-only experiment browser
- Artifact metadata 조회
- 통제된 demo scenario 실행

이를 위해 microservice, message broker 또는 time-series database가 반드시 필요한 것은 아니다.

## 5. 신뢰 경계

- Immutable manifest 밖의 파일은 신뢰하지 않는 input이다.
- Model weight는 hash와 metadata가 일치할 때만 허용한다.
- Client가 제공한 timestamp, user identity 또는 decision을 정본으로 사용하지 않는다.
- Camera-only Python prototype은 OS, driver 또는 virtual camera를 보증할 수 없다.
- 완료된 run directory는 append-only이며 수정할 때는 새 run ID를 만든다.

## 6. 실패 동작

- Contract violation: model 실행 전에 실패
- 필수 artifact 누락: 0 metric이 아닌 `BLOCKED_INPUT`
- Model exception: authentication reject나 defense success가 아닌 `ERROR`
- Low-quality capture: policy가 허용하면 재시도 가능한 결과
- 필수 security gate 누락: fail closed하고 result token을 발급하지 않음
