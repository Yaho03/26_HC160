# FACE AUTHENTICATION — 얼굴 인증 문서

## 1. 목적

`src/face_auth/`는 얼굴 인증 security control을 평가하는 local session 기반 reference
prototype이다. Banking backend, mobile SDK 또는 운영 biometric product가 아니다.

## 2. Runtime 흐름

```text
별도 enrollment capture
  → local multi-frame template

session 생성
  → nonce + randomized challenge
  → webcam 또는 recorded-video capture
  → frame integrity + quality + all-face detection
  → identity verification
  → FULL 전용 motion/replay/PAD/liveness/continuity gate
  → fail-closed policy decision
  → context-bound one-time result token
  → consume 또는 expire
```

## 3. Module map

| 경로 | 책임 |
|---|---|
| `domain/types.py` | Session, gate, decision, frame, manifest와 token value type |
| `domain/state_machine.py` | 허용 session transition과 terminal-state 보호 |
| `domain/policy.py` | BASELINE_ONLY/FULL 필수 gate와 failure 우선순위 |
| `application/session_service.py` | Session·challenge lifecycle |
| `application/evaluation_service.py` | Gate aggregate, decision과 token 발급 |
| `application/token_service.py` | Purpose/context-bound, expiring, one-time token |
| `application/enrollment_service.py` | 별도 multi-frame template 생성과 local NPZ 저장 |
| `application/evidence_service.py` | Nonce-bound ordered-frame evidence digest |
| `application/decision_artifact.py` | 개인정보 최소화 decision 직렬화, atomic output과 artifact reference |
| `adapters/capture_base.py` | 공통 frame-source와 bounded latest-frame buffer 계약 |
| `adapters/opencv_capture.py` | Recorded-video와 webcam input |
| `adapters/opencv_preview.py` | Memory-only camera preview, progress와 사용자 취소 |
| `inference/pipeline.py` | Baseline integrity, quality, all-face, identity evidence |
| `inference/full_pipeline.py` | FULL-profile gate composition |
| `inference/pad_adapter.py` | 명시적 TorchScript/ONNX PAD adapter. Heuristic pass fallback 없음 |
| `inference/pad_model_registry.py` | 승인 PAD model metadata, license/checksum과 runtime contract validation |
| `inference/active_liveness.py` | Challenge 이후 head-turn 또는 blink evidence |
| `inference/continuity.py` | Template-anchored multi-frame identity consistency |
| `inference/content_replay.py` | Batch·incremental frozen/repeated-content signal |
| `inference/camera_motion.py` | Background global-motion quality signal |
| `inference/adversarial_detector.py` | Transform-consistency optional veto |
| `evaluation/calibration.py` | Validation CSV 기반 prototype gate-threshold calibration |
| `evaluation/pad_manifest.py` | Opaque relative-path PAD video manifest validation |
| `evaluation/pad_evaluator.py` | Source-video hash와 excluded outcome을 포함한 video별 평가 |
| `evaluation/pad_metrics.py` | APCER, BPCER, ACER, attack-species metric과 Wilson interval |
| `evaluation/pad_cli.py` | Immutable-by-default labeled-video PAD report command |
| `evaluation/pad_capture.py` | 승인 physical PAD capture session과 append-only receipt |
| `evaluation/pad_capture_cli.py` | Capture 준비·검증·manifest materialization command |
| `../../schemas/pad-evaluation-report.schema.json` | PAD report와 provenance 계약 |
| `../../schemas/authentication-decision.schema.json` | Machine-readable terminal authentication decision 계약 |
| `cli.py` | 별도 `enroll`, `authenticate` command |

Attack-video 생성은 `src/attack_scenarios/`에서 별도로 관리한다.

## 4. Security profile

| Profile | 필수 gate | 적절한 용도 |
|---|---|---|
| `BASELINE_ONLY` | Frame integrity, quality, single face, identity | 개발·통합·recorded-video baseline |
| `FULL` | Baseline + camera motion, content replay, passive PAD, active liveness, continuity | 모든 model과 threshold를 검증한 뒤 사용하는 reference security composition |

설정한 경우 optional adversarial gate가 두 profile을 veto할 수 있다. 필수 gate가 누락되거나
error이면 `VERIFIED`를 반환할 수 없다.

## 5. 빠른 명령

```bash
python -m src.face_auth.cli enroll --help
python -m src.face_auth.cli authenticate --help
python -m unittest discover -s tests/unit -v
python -m unittest discover -s tests/integration -v
```

전체 예시와 troubleshooting은 `../14_LOCAL_RUNBOOK.md`를 따른다.

Webcam command는 기본 preview를 표시하고 recorded-video command는 headless다. `q` 또는
`Esc`로 preview capture를 취소한다. 의도적 headless webcam run에서만
`--no-preview`를 사용한다.

FULL preview에는 무작위 라이브니스 동작도 표시된다. 동작을 처음 표시한 프레임을
챌린지 경계로 사용하며, 이후 프레임만 active liveness를 충족할 수 있다. Recorded 또는
headless FULL capture를 구동하는 외부 UI는 `--challenge-start-frame-id`로 경계를 전달해야
하며, 경계가 없거나 유효하지 않으면 fail closed한다.

해당 경계 이후 FULL capture는 프레임마다 replay monitor를 갱신한다. 실패하면
`REPLAY DETECTED`를 표시하고 수집을 중단한 뒤, 캡처된 prefix를 기록하고 session을 즉시
`SECURITY_DENIED`로 전환하며 token을 발급하지 않는다. 이 조기 veto는 passive PAD나
최종 batch replay gate를 대체하지 않는다.

`--decision-output outputs/face-auth/decision.json`을 전달하면 identity label이나 biometric
input을 저장하지 않고 terminal policy 결과 또는 live replay veto를 보존한다. Output은
기본적으로 immutable이며 `--overwrite-decision-output`은 폐기 가능한 local rerun에서만
사용한다. 결과를 experiment evidence로 사용할 때는 생성한 run manifest에 `decision_id`를
등록한다.

## 6. 문서 지도

- `API_CONTRACT.md` — State, gate, evidence와 token 계약
- `THREAT_MODEL.md` — 보호 자산, 범위 내 threat와 trust-boundary 한계
- `IMPLEMENTATION_PLAN.md` — 구현 구조와 남은 작업
- `BACKLOG.md` — 요구사항별 작업과 현재 상태
- `EXPERIMENT_PLAN.md` — Split 규칙, 평가 group과 metric
- `SMOKE_TEST_REPORT_2026-08-22.md` — 제한된 recorded-video 증거와 해석
- `PAD_CAPTURE_PROTOCOL.md` — 승인 physical bona-fide/attack capture 및 custody 절차
- `../13_IMPLEMENTATION_STATUS.md` — 저장소 전체 구현·증거 matrix
- `../15_ISSUE_AND_PR_WORKFLOW.md` — 미완료 backlog를 이슈로 전환하는 규칙

## 7. 양보할 수 없는 제한 사항

- Local NPZ template는 암호화되지 않는다.
- In-memory session/token store는 durable transactional database가 아니다.
- Python camera process는 OS, driver 또는 virtual camera를 보증할 수 없다.
- 저장소에는 검증된 PAD checkpoint가 없다.
- Example threshold는 release threshold가 아니다.
- Synthetic replay success는 physical print/screen/deepfake 성능을 증명하지 않는다.
- `VERIFIED`는 authentication result이며 금융 transaction authorization이 아니다.
