# FACE VERIFICATION SPEC — 얼굴 검증 사양

| 항목 | 내용 |
|---|---|
| 문서명 | 얼굴 Verification 사양서 |
| 버전 | v1.0 |
| 상태 | 확정 |
| 최종 수정일 | 2026-08-23 |

---

## 1. 판정 모델

Enrollment embedding `e`와 probe embedding `p`에 대해 다음을 사용한다.

```text
score = cosine_similarity(e, p)
accept = score >= threshold
```

Embedding model, face preprocessing, score function과 threshold artifact는 하나의 versioned
verification protocol을 구성한다. 다른 model 또는 preprocessing version에서 얻은
threshold는 유효하지 않다.

## 2. 지원 트랙

| Protocol | 역할 | Image size | 현재 model |
|---|---|---:|---|
| `resnet50-lfw10-bridge-v1` | Legacy bridge | 224 | LFW-10 classifier backbone |
| `facenet-vggface2-v1` | 주요 후보 | 160 | InceptionResnetV1, VGGFace2 weights |
| `camera-facenet-prototype-v1` | Camera 탐색 | 160 | MTCNN crop + FaceNet |

서로 다른 protocol의 score와 threshold를 합치지 않는다.

## 3. Calibration

1. 고정된 calibration-pair manifest를 만든다.
2. 고정 model과 preprocessing version으로 embedding을 계산한다.
3. Calibration row만 사용하여 운영 threshold를 선택한다.
4. Threshold, 선택 방법, 지원 FAR 범위, source hash와 software version을 저장한다.
5. Attack, defense 또는 최종 test 평가 전에 threshold를 고정한다.

EER는 비교 목적으로 보고할 수 있다. 금융 시나리오는 지원 가능한 low-FAR operating
point의 TAR도 보고해야 한다. Negative pair 수로 통계적으로 지원할 수 없는 FAR을
주장해서는 안 된다.

## 4. Clean 평가 출력

- Pair별 score와 accept/reject decision
- TP, TN, FP, FN
- FAR/FMR, FRR/FNMR, TAR, TNR
- EER와 ROC-AUC
- 지원 가능한 FAR target에서 TAR
- Sample 수와 confidence interval
- Model, preprocessing, threshold, dataset 및 protocol ID

## 5. 공격 평가

Targeted impersonation attempt는 공격 전 거부된 다른 identity pair에서 시작한다.

```text
success_from_reject = (accepted_before is False) and (accepted_after is True)
```

`accepted_after`만으로 공격 성공을 정의하면 공격 전 이미 false accept였던 pair가
포함되므로 유효하지 않다.

## 6. 방어 평가

모든 방어는 다음 대상에 적용한다.

- Eligible successful adversarial probe
- 대응하는 clean probe
- 독립적으로 추출한 clean verification test set

Report는 방어 후 attack ASR과 clean TAR/FRR 변화를 함께 보여야 한다. 모든 입력을
거부하는 방어는 attack ASR이 0이어도 유효하지 않다.

## 7. Threshold 생명주기

Threshold artifact ID는 다음과 같은 형식을 사용한다.

```text
thr_<protocol>_<calibration-manifest-hash>_<method-version>
```

Weight, detector/alignment, resize, normalization, score function 또는 calibration data가
변경되면 새 threshold artifact를 만든다.

## 8. 실행 가능한 calibration 계약

`src/evaluation/verification_calibration.py`는 EXP-VER-001의 model-independent 부분을
구현한다. Embedding 생성이 끝난 pair-level similarity score를 입력받아 다음을 검증한다.

- Threshold 선택에는 calibration row만 사용
- Clean baseline 평가에는 test row만 사용
- Protocol, model 및 preprocessing ID 일치
- Calibration pair-ID provenance 일치
- Calibration/test pair overlap 0건
- 평가 split마다 genuine과 impostor row 존재
- 유한한 numeric score와 unique pair ID

Target FAR을 요청하면 impostor calibration pair가 최소 `1 / target_far`개 있어야 한다.
더 적은 dataset은 observed zero false accept를 보고할 수 있지만 해당 FAR operating
point를 주장할 수 없다. EER 선택은 비교 목적으로만 사용하며 고정 test threshold를 다시
선택하지 않는다.

Score JSONL row는 `schemas/verification-score.schema.json`을 따른다.
`facenet_score_export_cli`는 암시적 download 없이 명시한 VGGFace2 checkpoint를
불러오며, dataset manifest의 image hash를 검증한 뒤
`schemas/verification-score-export.schema.json`에 맞는 sidecar를 작성한다.

Calibration score export 예시는 다음과 같다.

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

변경하지 않은 test pair manifest에 새 run ID와 출력 이름을 사용하여 같은 과정을 반복한다.
두 export는 동일한 identity-disjoint dataset manifest, checkpoint, preprocessing config와
protocol을 사용해야 한다. 이후 threshold와 clean report를 생성한다.

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

두 명령은 기본적으로 기존 출력을 덮어쓰지 않는다. 정식 score export는 clean worktree와
identity-disjoint data를 요구한다. Calibration 명령은 score file hash와
model/preprocessing/dataset provenance를 검증한 뒤 고정 threshold confusion count,
분자·분모가 명시된 rate, Wilson 95% interval, ROC-AUC와 설명용 test EER를 보고한다.
