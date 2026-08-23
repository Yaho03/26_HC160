# IMPLEMENTATION STATUS — 구현 상태

| 항목 | 내용 |
|---|---|
| 문서명 | 구현 및 검증 상태 |
| 버전 | v1.1 |
| 상태 | 진행 중 |
| 기준 snapshot | commit `a5fe5ad`, branch `codex/realtime-face-auth-v2` |
| 최종 수정일 | 2026-08-23 |

---

## 1. Snapshot 요약

- 연구 기반과 artifact contract: 구현
- 개인정보를 고려한 dataset manifest workflow: 구현
- Verification threshold calibration과 clean-report workflow: score-record 계층에서 구현
- Session 기반 face-authentication reference prototype: 구현
- Physical PAD validation, 운영 persistence, mobile attestation과 release claim: 미완료

2026-08-22에 기록된 검증:

- Python 3.9 전체 suite: 144 tests passed
- Python 3.9 research suite: 37 tests passed
- Python 3.13 research suite: 37 tests passed
- Recorded-video FaceNet smoke test: 완료. `face_auth/SMOKE_TEST_REPORT_2026-08-22.md` 참조
- PAD report wiring smoke: constant-output test model로 calibration/test report 생성. Print APCER `1.0`은 reporting 동작을 확인한 값이며 PAD accuracy가 아니다.
- Physical capture 준비: 시작 전. 동의한 subject, print media, display-replay device와 held-out PAD evidence가 필요하다.

위 test 수는 해당 snapshot을 식별하는 증거다. 현재 revision은 현재 test 명령으로 다시
검증해야 한다.

## 2. 상태 label 의미

| Label | 의미 |
|---|---|
| 구현됨 | Code path와 자동 contract test가 존재 |
| Smoke 검증 | 명시한 input으로 제한된 로컬 실행 완료 |
| 실험 대기 | Code는 있으나 필요한 disjoint data와 최종 metric이 없음 |
| 외부 artifact 차단 | 필요한 model, dataset 또는 device evidence가 저장소에 없음 |
| 운영 확장 | 대회 수준 로컬 prototype의 의도적 범위 밖 |

“구현됨”은 보안 주장이 검증됐다는 뜻이 아니다. Experiment protocol, data provenance,
threshold, sample 수, confidence interval과 limitation을 기록해야 보고 가능한 기능이 된다.

## 3. 현재 구현 matrix

| 영역 | 상태 | 구현 증거 | 남은 증거 또는 작업 |
|---|---|---|---|
| Dataset snapshot | 구현됨 | `src/datasets/manifest.py`, `manifest_cli.py`, dataset schema | Raw identity 없는 승인 LFW snapshot manifest 생성·공개 |
| Split leakage 검사 | 구현됨 | Media-hash 및 optional identity-disjoint 검사 | 최종 subject/session/device split에서 실행 |
| Verification metric | 구현됨 | `src/evaluation/verification_metrics.py` | 고정 dataset에서 승인 model score에 적용 |
| FaceNet score export | 구현됨·실험 대기 | 명시 checkpoint loader, preprocessing 계약, pair/image validation, schema와 provenance sidecar | 승인 identity-disjoint calibration/test manifest에서 실행 |
| Verification calibration | 구현됨 | `verification_calibration.py`, provenance를 확인하는 `verification_baseline_cli.py` | 승인 export로 최종 EXP-VER-001 threshold와 clean report 생성 |
| Threshold provenance | 구현됨 | Threshold/clean-report schema가 model, preprocessing, score-export SHA-256을 결합 | 최종 output을 run/artifact manifest에 등록 |
| Session state·policy | 구현됨 | `src/face_auth/domain/`, unit/integration test | Persistent adapter와 concurrency/transaction test |
| Enrollment 분리 | 구현됨 | `enrollment_service.py`, 별도 `enroll` CLI | Template 암호화, 승인된 등록·폐기 |
| Evidence binding | 로컬 구현 | Nonce-bound capture manifest와 digest | 별도 client trust boundary의 signed/attested evidence |
| Baseline video authentication | Smoke 검증 | MTCNN, FaceNet, quality와 identity pipeline | Target device의 identity·quality threshold calibration |
| Camera input | 구현됨 | 공통 OpenCV `FrameSource` 계약 | Device matrix, 장시간 capture, drop·latency 실험 |
| Live camera interaction | 구현됨 | Memory-only preview, progress overlay, cancel, headless override, structured error | macOS 권한 부여와 target-camera manual smoke test |
| Repeated-content detection | Smoke 검증 | Codec-tolerant batch gate, challenge 이후 streaming veto와 즉시 `SECURITY_DENIED` 전환 | Codec/camera별 genuine/attack false-positive 연구 |
| Camera-motion gate | 구현됨 | Background motion estimator | Target-camera calibration. 현재 retry용 evidence |
| Passive PAD | Capture/evaluation harness 구현·외부 artifact 차단 | TorchScript/ONNX adapter, fail-closed registry, capture CLI/protocol, manifest validator, evaluator와 APCER/BPCER | 승인 model·license/checksum, 승인 capture session, held-out physical 평가 |
| PAD report provenance | 구현됨 | Run ID, Git state, manifest/model/source-video hash·byte, immutable output와 schema | 완료 report를 저장소 전체 run/artifact manifest에 등록 |
| Active liveness | 구현됨 | Randomized challenge, 실시간 동작 지시, 표시 프레임 경계 결합과 head-turn/blink logic | Physical replay 연구, 접근성 대안, threshold calibration |
| Identity continuity | 구현됨 | Template-anchored temporal gate | Held-out person-switch·occlusion 평가 |
| Adversarial inspection | Optional veto로 구현 | Transform-consistency와 feature-squeeze module | Clean calibration, adaptive attack, latency 평가 |
| Scenario generation | 구현됨 | Manifest 기반 insertion/replay builder | Scenario 확장과 실제 physical attack capture |
| Legacy adversarial training | 역사적 결과만 존재 | 기존 defense output과 script | 올바른 disjoint train/validation/test 재실행과 adaptive attack |
| UI 또는 service API | 미구현 | Local CLI만 존재 | 검증 artifact 확보 후 read-only demo/API 검토 |
| Secure persistence | 운영 확장 | In-memory store와 local NPZ | DB transaction, encryption, KMS/HSM, retention과 audit |
| Mobile/device attestation | 운영 확장 | Trust boundary 문서화 | iOS/Android client, server verification, hardware-backed key |

## 4. 혼동하면 안 되는 calibration workflow

| 경로 | 목적 | Input | Output |
|---|---|---|---|
| `src/evaluation/verification_calibration.py` | 연구 verification operating threshold와 clean baseline | 분리된 calibration/test manifest의 pair-level cosine score | Versioned threshold artifact와 EXP-VER-001 clean report |
| `src/face_auth/evaluation/calibration.py` | PAD, motion, replay, adversarial check 등 prototype gate threshold | Clean/attack gate value가 있는 validation CSV | Face-auth CLI용 local threshold bundle |

두 번째 workflow는 첫 번째를 대체하지 않는다. Face-auth configuration은 연구로 검증된
identity threshold와 별도로 검증한 gate threshold를 함께 참조해야 한다. 두 workflow 모두
test row로 tuning하면 안 된다.

## 5. 현재 주장할 수 있는 범위

현재 시연 가능한 내용:

- Deterministic state, policy, token, evidence와 artifact contract
- Recorded-video baseline authentication vertical slice
- FULL-profile gate의 fail-closed composition
- 재현 가능한 synthetic replay/insertion scenario
- Test leakage 없는 model-independent verification calibration

현재 주장할 수 없는 내용:

- 운영 금융 인증 준비 완료
- 검증된 PAD accuracy 또는 보편적인 replay detection rate
- 손상된 camera driver, virtual camera 또는 rooted device에 대한 robustness
- Fairness 또는 demographic parity
- Certified adversarial robustness
- Legacy defense 백분율을 held-out verification security로 해석하는 주장

## 6. 다음 증거 권장 순서

1. EXP-DATA-001로 승인 subject/session/device-disjoint manifest를 만든다.
2. 고정 calibration/test pair manifest와 pinned checkpoint로 FaceNet score JSONL 및 provenance sidecar를 만든다.
3. EXP-VER-001을 실행하고 identity threshold artifact를 고정한다.
4. License-compatible PAD checkpoint를 확보하고 checksum·preprocessing 계약을 기록한다.
5. Held-out genuine, print, screen, replay, insertion과 person-switch session을 capture한다.
6. Quality, PAD, liveness, continuity, replay, motion과 adversarial threshold를 validation에서만 calibration한다.
7. FMR/FNMR, APCER/BPCER, attack 종류별 결과, clean cost, latency, error와 CI를 보고한다.

이 순서에서 미완료 항목은 `15_ISSUE_AND_PR_WORKFLOW.md`에 따라 이슈로 전환한다.

## 7. 문서 검토에서 확인된 integration gap

| 우선순위 | Gap | 필수 후속 작업 |
|---|---|---|
| P0 before FULL claim | 승인 PAD checkpoint 또는 held-out physical dataset이 없음 | FULL profile을 보고 가능 상태로 만들기 전에 model을 확보하고 license·hash·calibration·evaluation 완료 |
| P1 | GitHub CI가 dependency-free research test만 실행 | 고정 face-auth CI job 또는 heavyweight validation workflow 추가 |
| P1 | PAD report에는 schema와 run/source provenance가 있지만 face-auth decision과 전체 run/artifact 등록이 미완료 | Decision schema 정의 및 모든 report를 run manifest와 artifact reference에 등록 |
| P1 | `requirements-face-auth.txt`는 version pin만 있고 hash lock이 없음. Target Python 3.11과 전체 local run Python 3.9가 다름 | Clean Python 3.11 검증 및 locked environment artifact 공개 |
| P1 | Template, session과 token이 local NPZ/in-memory adapter 사용 | Multi-process 또는 remote service 전에 encrypted transactional persistence 추가 |

PAD report는 더 이상 local manifest/model path를 직렬화하지 않는다. Formal run은 기본적으로
dirty worktree와 기존 output path를 거부한다. `--allow-dirty`와 `--overwrite`는
일회성 local debugging에서만 사용하는 비기본 escape hatch다.
