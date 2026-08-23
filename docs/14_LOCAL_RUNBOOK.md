# LOCAL RUNBOOK — 로컬 실행 가이드

| 항목 | 내용 |
|---|---|
| 문서명 | 로컬 실행·검증 Runbook |
| 버전 | v1.1 |
| 상태 | 진행 중 |
| 최종 수정일 | 2026-08-23 |

---

## 1. 작업 위치와 브랜치

모든 명령은 저장소 root에서 실행한다. 작업 전 현재 브랜치와 변경 파일을 확인한다.

```bash
git status -sb
```

이 문서에 적힌 과거 branch 이름을 현재 작업 branch로 가정하지 않는다. 새 작업은
`CONTRIBUTING.md`와 `15_ISSUE_AND_PR_WORKFLOW.md`의 branch 규칙을 따른다.

## 2. Python 환경

- 목표 runtime: Python 3.11
- 자동화된 face-auth CI runtime: Python 3.11 + `requirements-face-auth.lock`
- 전체 prototype을 로컬에서 확인한 runtime: Python 3.9 + `requirements-face-auth.txt`
- Research contract test: ML 의존성이 없어 system Python 3.13에서도 실행 기록이 있음

격리 환경을 만들고 필요한 dependency를 설치한다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-face-auth.txt
```

CI와 clean Linux 검증은 CPython 3.11 `linux/amd64`용 전체 dependency lock을 사용한다.

```bash
python scripts/verify_face_auth_lock.py
python -m pip install --require-hashes -r requirements-face-auth.lock
python -m pip check
```

Minimal Debian/Ubuntu image에서는 OpenCV import 전에 `libgl1`, `libglib2.0-0`도 설치한다.
`requirements-face-auth.txt`는 개발용 cross-platform direct dependency 목록이고,
`requirements-face-auth.lock`은 CI target platform 전용 transitive hash lock이다.

승인된 ONNX PAD checkpoint를 평가할 때만 optional runtime을 설치한다.

```bash
python -m pip install -r requirements-pad-onnx.txt
```

`.venv`, 다운로드한 weight, raw face, embedding, template와 PAD model을 커밋하지 않는다.

## 3. 검증 명령

외부 ML dependency가 필요 없는 빠른 research validation:

```bash
python -m unittest discover -s tests/research -v
```

Face-auth dependency 설치 후 전체 validation:

```bash
python -m unittest discover -s tests -v
```

문서화된 snapshot에서 두 번째 명령은 Python 3.9로 144개 test를 통과했다. Test-only
TorchScript trace의 warning은 실패가 아니지만 failed 또는 errored test는 실패다. 현재
revision은 반드시 명령을 다시 실행해 확인한다.

`.github/workflows/face-auth.yml`은 Python 3.11에서 direct pin과 SHA-256 항목을 검증하고,
`--require-hashes`로 lock을 설치한 뒤 `pip check`, unit test와 integration test를 실행한다.
이 workflow는 camera나 PAD checkpoint를 내려받지 않으며 physical attack 정확도를 검증하지 않는다.

## 4. Dataset manifest workflow

Manifest builder는 가명 identity directory 아래 artifact-ready file만 허용한다.

```text
<artifact-root>/<split>/id_<16-or-more-lowercase-hex>/<image>.png
```

전체 build 명령은 `03_DATASET_AND_PREPROCESSING.md`를 따른다. 기존 manifest와 media는
다음과 같이 검증한다.

```bash
python -m src.datasets.manifest_cli validate \
  --manifest data/manifests/DATASET.jsonl \
  --artifact-root external_artifacts/DATASET \
  --require-identity-disjoint
```

Open-set subject separation을 주장하지 않는 protocol에서만
`--require-identity-disjoint`를 생략할 수 있다. Media hash는 언제나 split 역할을
넘어 중복될 수 없다.

## 5. Verification baseline workflow

명시적으로 승인된 VGGFace2 checkpoint에서 calibration score를 export한다. Dataset
manifest는 calibration과 test row를 모두 포함하여 기본 identity-disjoint 검사가
cross-split leakage를 확인할 수 있어야 한다.

```bash
python -m src.evaluation.facenet_score_export_cli \
  --dataset-manifest data/manifests/verification-dataset.jsonl \
  --artifact-root external_artifacts/verification \
  --pair-manifest data/splits/verification-calibration-pairs.jsonl \
  --pair-manifest-id pairs-calibration-v1 \
  --protocol-id facenet-vggface2-v1 \
  --model-checkpoint local_models/20180402-114759-vggface2.pt \
  --model-artifact-id facenet-vggface2-weights-v1 \
  --preprocessing-config configs/models/facenet-vggface2-preprocessing.json \
  --run-id run-score-calibration-v1 \
  --scores-output outputs/verification/calibration-scores.jsonl \
  --metadata-output outputs/verification/calibration-scores.metadata.json
```

`verification-test-pairs.jsonl`에는 새 output name과
`--run-id run-score-test-v1`을 사용한다. 두 export 사이에서 dataset manifest,
checkpoint, preprocessing config, model ID 또는 protocol을 바꾸면 안 된다.

두 provenance sidecar가 생성된 뒤에만 calibration과 평가를 실행한다.

```bash
python -m src.evaluation.verification_baseline_cli \
  --calibration-scores outputs/verification/calibration-scores.jsonl \
  --calibration-score-metadata outputs/verification/calibration-scores.metadata.json \
  --test-scores outputs/verification/test-scores.jsonl \
  --test-score-metadata outputs/verification/test-scores.metadata.json \
  --selection-method target_far \
  --target-far 0.001 \
  --threshold-artifact-id thr-facenet-v1 \
  --run-id run-exp-ver-001 \
  --threshold-output outputs/verification/threshold.json \
  --report-output outputs/verification/clean-report.json
```

Exporter는 embedding을 저장하지 않는다. Checkpoint, preprocessing, dataset,
pair-manifest, score file과 Git hash를 기록하고 기본적으로 dirty worktree와 기존 output을
거부한다. Calibration command는 변조된 score 또는 provenance 불일치를 거부한다.
Repository 최소 경험 규칙에서 FAR `0.001`은 impostor calibration pair가 1,000개
이상이어야 한다.

## 6. Face-auth baseline workflow

Recorded video에서 local enrollment template를 만든다.

```bash
python -m src.face_auth.cli enroll \
  --video path/to/enrollment.mp4 \
  --frames 30 \
  --min-valid-frames 5 \
  --output local_templates/user-1.npz
```

별도 probe recording을 인증한다.

```bash
python -m src.face_auth.cli authenticate \
  --video path/to/probe.mp4 \
  --template local_templates/user-1.npz \
  --threshold 0.60 \
  --threshold-version validation-only-example \
  --user-id user-1 \
  --context-hash demo-context-a \
  --profile BASELINE_ONLY \
  --decision-output outputs/face-auth/decision.json \
  --registration-context configs/run-registration.json
```

`0.60`은 release threshold가 아니다. 승인 calibration data에서 생성한 고정 artifact
값으로 교체한다. Enrollment와 probe video는 서로 다른 capture여야 한다.

Webcam은 `--video` 대신 `--camera 0`을 사용한다. Baseline profile에는 PAD, active
liveness 또는 continuity가 없으므로 완전한 인증 보안으로 소개하면 안 된다.

선택적 decision output은 `schemas/authentication-decision.schema.json`을 따른다. Policy와
gate version, terminal state, frame 수, attempt ID와 evidence SHA-256은 저장하지만 user ID,
challenge nonce, frame pixel, embedding 또는 template은 저장하지 않는다. 기존 output은
기본적으로 덮어쓰지 않는다. 보고 가능한 run에서는 `kind: decision` artifact reference를
생성하고 `decision_id`를 run manifest의 `output_artifact_ids`에 추가한다.
`configs/run-registration.example.json`을 복사한 뒤 placeholder를 승인된 실제 run
metadata로 바꾸고 `--registration-context`로 전달한다. 성공하면
`<output>.artifact-reference.json`과 `<output>.run-manifest.json`을 함께 생성한다.
각 실행은 고유한 run/output artifact ID를 사용해야 한다.

Camera input은 기본적으로 local OpenCV preview를 연다. Guide 안에 한 얼굴을 유지하고
`q` 또는 `Esc`로 취소한다. 의도적인 headless 실행에서만 `--no-preview`를 사용한다.
Preview는 memory-only이며 raw frame을 저장하지 않는다.

macOS에서는 Python을 시작한 application에 camera 권한을 허용한다. `CAMERA_UNAVAILABLE`
이면 **시스템 설정 > 개인정보 보호 및 보안 > 카메라**에서 실행 중인 앱, Terminal 또는 해당
Python host의 권한을 허용한 뒤 명령을 다시 시작한다. 권한 거부나 장치 미사용 상태는
traceback 대신 structured `CAPTURE_ERROR` JSON을 반환한다.

## 7. FULL profile 선행 조건

FULL profile 실행 전 다음이 모두 필요하다.

- 검증된 TorchScript 또는 지원 ONNX PAD model
- CLI argument와 일치하는 PAD preprocessing 및 live-class 의미
- Identity, PAD, motion, continuity와 optional adversarial threshold version
- Active liveness 평가에 충분한 post-challenge frame
- Target-device validation evidence

기본 preview를 사용하는 camera capture에서는 무작위 FULL challenge를 창에 표시하고,
처음 표시한 프레임을 자동으로 결합한다. Recorded-video 또는 `--no-preview` FULL 실행은
외부 challenge presenter가 기록한 `--challenge-start-frame-id N`이 필요하다. 이 값은
캡처된 프레임을 가리키면서 뒤에 `--min-valid-frames`개 이상의 프레임을 남겨야 한다.
그렇지 않으면 `CHALLENGE_BINDING_ERROR`를 반환한다. Preview가 경계를 자동 기록할 때는
외부 경계를 함께 전달하지 않는다.

FULL capture는 challenge 경계 이후 content-replay gate도 증분 실행한다. 반복·정지 run이
`--content-replay-max-run`을 넘으면 즉시 capture를 중단하고 `LIVE_SECURITY_VETO`를
반환한다. 기본값은 `2`, threshold version은 `content-replay-v2`다.
`--content-replay-max-difference`는 codec 차이를 허용하는 fingerprint 거리를 제어한다.
이 기본값은 prototype 값이며 보안 비율을 주장하기 전에 target-camera 검증이 필요하다.

`--pad-model`이 없으면 FULL은 fail closed하며 시작하지 않는다. Command와 gate map은
`face_auth/README.md`를 따른다.

## 8. Passive PAD 평가 workflow

PAD manifest는 opaque subject/session/device token을 사용하고 calibration과 test row를
분리한다. `configs/pad_evaluation.example.csv`는 형식 예시일 뿐 실험 증거가 아니다.

승인 TorchScript 또는 ONNX PAD model을 확보한 뒤 calibration row만으로 threshold를
선택한다.

```bash
python -m src.face_auth.evaluation.pad_cli \
  --mode calibrate \
  --manifest data/manifests/pad-evaluation.csv \
  --manifest-id pad-evaluation-v1 \
  --artifact-root external_artifacts/pad \
  --pad-model local_models/pad-v1.ts \
  --pad-model-version pad-v1 \
  --threshold-version pad-validation-v1 \
  --run-id run-pad-calibration-v1 \
  --split calibration \
  --max-bpcer 0.05 \
  --registration-context configs/run-registration.json \
  --output outputs/pad/pad-calibration-v1.json
```

보고된 threshold를 고정하고 변경하지 않은 test split을 평가한다.

```bash
python -m src.face_auth.evaluation.pad_cli \
  --mode evaluate \
  --manifest data/manifests/pad-evaluation.csv \
  --manifest-id pad-evaluation-v1 \
  --artifact-root external_artifacts/pad \
  --pad-model local_models/pad-v1.ts \
  --pad-model-version pad-v1 \
  --live-threshold 0.80 \
  --threshold-version pad-validation-v1 \
  --run-id run-pad-test-v1 \
  --split test \
  --registration-context configs/run-registration.json \
  --output outputs/pad/pad-test-v1.json
```

Original Open Model Zoo `anti-spoof-mn3` ONNX artifact는 `--pad-runtime onnx`를
추가한다. Adapter는 문서화된 128×128 RGB input, class-zero bona-fide output과 기본
mean/scale을 사용한다. 고정 checkpoint 문서가 요구할 때만 model-contract argument를
override한다.

Evaluator는 sample outcome, valid-frame 수, model/threshold version, latency, APCER,
BPCER, ACER, worst/per-species APCER, presentation/exclusion count와 Wilson 95% interval을
기록한다. Git commit, manifest/model hash 및 선택한 source video의 hash와 byte 수도
결합한다. 평가 후 input을 다시 확인하므로 실행 중 변경되면 report 생성을 중단한다.

Formal run은 clean worktree를 요구하고 기존 output path를 거부한다. Registration context의
`run_id`는 PAD `--run-id`와 같아야 하며, 성공한 report에는 immutable artifact-reference와
run-manifest sidecar가 생성된다. `--allow-dirty`와 `--overwrite`는 명시적인 local
debugging에서만 사용하고 등록된 run에서는 overwrite를 거부한다. Multi-face, quality 부족,
load와 model failure는 PAD success에 포함하지 않고 별도로 보고한다.

## 9. Scenario·gate threshold workflow

결정적인 attack video 생성:

```bash
python -m src.attack_scenarios.cli \
  --manifest configs/scenarios/mid_frame_insertion.example.json
```

Validation row만 사용한 prototype gate threshold calibration:

```bash
python -m src.face_auth.evaluation.calibrate_cli \
  --input configs/thresholds.validation.example.csv \
  --output local_thresholds/validation-v1.json \
  --version validation-v1
```

Example CSV는 형식 예시이며 실험 증거가 아니다.

## 10. 자주 발생하는 실패

| 증상 | 의미와 대응 |
|---|---|
| `ModuleNotFoundError: numpy` | Face-auth dependency 없는 system Python 사용. 의도한 environment 활성화 |
| FaceNet/MTCNN download 또는 load 실패 | External pretrained weight 부재. Network, cache, version, license와 checksum 확인 |
| `FULL profile requires --pad-model` | 예상된 configuration 차단. 검증 model 제공 또는 제한된 baseline profile 사용 |
| 요청 FAR 미지원 | Impostor calibration pair를 늘리거나 통계적으로 지원되는 operating point 선택 |
| Output already exists | Silent overwrite 차단. 새 run/artifact ID 사용 |
| Decision artifact already exists | 새 output path 사용. `--overwrite-decision-output`은 폐기 가능한 명시적 local rerun에서만 사용 |
| Quality retry가 많음 | Threshold를 바로 낮추지 말고 target-device clean validation data부터 측정 |
| Re-encoded frame replay 탐지 실패 | Codec transfer를 validation에 포함하고 content-distance threshold calibration |

## 11. Commit·report 전 확인

- `git status -sb`에서 의도한 파일만 있는지 확인
- 관련 research 및 full test 실행
- Dataset, model, preprocessing, threshold, policy와 Git version 기록
- Numerator, denominator, confidence interval, failure와 latency 보존
- Face, embedding, template, checkpoint와 local absolute path 제외
- 구현 code, smoke evidence, 최종 experiment evidence와 운영 확장을 구분
- `docs/13_IMPLEMENTATION_STATUS.md` 갱신
- `15_ISSUE_AND_PR_WORKFLOW.md`의 PR 머지 조건 확인
