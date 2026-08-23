# DATASET AND PREPROCESSING — 데이터셋·전처리

| 항목 | 내용 |
|---|---|
| 문서명 | 데이터셋 및 전처리 사양서 |
| 버전 | v1.0 |
| 상태 | 확정 |
| 최종 수정일 | 2026-08-23 |

---

## 1. 현재 데이터셋

현재 저장소는 LFW deep-funneled image를 전제로 한다. Legacy classification pipeline은
조건을 충족하는 이미지가 가장 많은 identity 10개를 선택하고, identity별 이미지를 섞어
image 단위 train/validation/test directory를 만든다. 이 protocol은 과거 classification
결과를 재현하는 용도로만 유지한다.

주요 verification protocol은 train, calibration, development와 test 역할을 명시한 고정
pair manifest를 사용해야 한다. Threshold calibration과 최종 test 평가는 같은 pair row를
재사용하면 안 된다.

## 2. 필수 dataset metadata

모든 dataset snapshot은 다음을 기록한다.

- Dataset ID와 schema version
- 출처 페이지와 수집 일자
- License와 재배포 제한
- Archive SHA-256과 예상 파일 수
- Preprocessing 구현과 version
- Sample별 상대 경로와 media hash
- 가명 처리한 identity token
- Split과 protocol ID
- 알려진 인구통계·수집 한계

LFW를 로컬에서 재구성할 때 실제 이름이 필요할 수 있지만, 커밋하는 manifest와 report에는
가명 identity token을 사용한다.

## 3. Split 정책

| 역할 | 허용 용도 |
|---|---|
| Train | Model 및 학습형 방어 fitting |
| Calibration | Threshold와 detector cutoff 선택 |
| Development | Attack/defense parameter 탐색 |
| Test | Configuration을 고정한 뒤 수행하는 최종 평가 |

Open-set generalization을 주장하는 실험은 subject identity까지 분리해야 한다. 최소한 역할
간 pair ID와 media hash가 겹치면 안 된다. 동일 capture의 enrollment와 probe image도 서로
다른 역할로 나누지 않는다.

## 4. Preprocessing identity

Preprocessing은 model artifact의 일부이며 반드시 version을 지정한다.

- Face detector와 alignment model/version
- Crop margin과 image size
- Interpolation mode
- Channel order와 numeric range
- Mean/standard-deviation normalization
- Color profile과 image decoder version
- Input이 original, PNG, JPEG 또는 defense-derived artifact인지 여부

| 트랙 | 현재 preprocessing |
|---|---|
| ResNet classification/bridge | 224×224 resize, RGB, ImageNet normalization |
| FaceNet verification defense | 160×160 resize, RGB, `(pixel - 127.5) / 128.0` |
| Camera prototype | MTCNN으로 160×160 crop 후 FaceNet normalization |

Calibration run이 동등성을 명시적으로 증명하지 않는 한 서로 다른 protocol은 threshold를
공유하지 않는다.

## 5. Artifact 형식

- 정본 clean/adversarial image: lossless PNG 또는 tensor artifact
- JPEG: encoder, quality와 parent hash를 기록한 파생 artifact
- Table의 path: artifact-root 기준 POSIX 상대 경로
- 모든 image artifact: SHA-256, byte 수, MIME type, sensitivity와 parent ID
- Raw face와 embedding: Git 제외

## 6. Leakage 검사

자동 validation은 다음 상태를 거부해야 한다.

- 동일 media hash가 train과 test에 존재
- 동일 pair ID가 calibration과 test에 존재
- 학습형 방어가 자신의 attack-training row에서 평가됨
- Test row로 threshold artifact를 생성
- Result row의 checkpoint/preprocessing ID와 threshold artifact가 불일치

## 7. LFW 한계

LFW는 연구 prototype에는 사용할 수 있지만 운영 금융 인증을 검증하는 데이터셋이 아니다.
금융 고객을 대표하지 않으며 공정성 주장의 근거가 될 수 없다. 공개 face-model pretraining
dataset과 identity가 겹칠 수도 있다. 모든 report에 이 한계를 명시한다.

## 8. EXP-DATA-001 manifest 흐름

외부 의존성이 없는 manifest tool은 다음 구조의 artifact directory를 입력으로 받는다.

```text
<artifact-root>/
  train/id_<pseudonymous-hex>/image.png
  calibration/id_<pseudonymous-hex>/image.jpg
  development/id_<pseudonymous-hex>/image.png
  test/id_<pseudonymous-hex>/image.jpg
```

실제 identity 이름을 사용한 directory는 거부한다. Git 밖에서 가명 artifact tree를 만든 뒤
manifest와 snapshot record를 생성한다.

```bash
python -m src.datasets.manifest_cli build \
  --artifact-root /secure/artifacts/lfw-v1 \
  --manifest-output data/manifests/lfw-v1.jsonl \
  --metadata-output data/manifests/lfw-v1.snapshot.json \
  --manifest-uri data/manifests/lfw-v1.jsonl \
  --dataset-id lfw-v1 \
  --license-id LFW-terms \
  --source-uri https://vis-www.cs.umass.edu/lfw/ \
  --source-retrieved-at 2026-08-22 \
  --source-archive /secure/downloads/lfw-deepfunneled.tgz \
  --require-identity-disjoint
```

Artifact root를 제공하면 validation 과정에서 media hash와 dimension을 다시 계산한다.

```bash
python -m src.datasets.manifest_cli validate \
  --manifest data/manifests/lfw-v1.jsonl \
  --artifact-root /secure/artifacts/lfw-v1 \
  --require-identity-disjoint
```

Subject-disjoint generalization을 주장하는 protocol에만 `--require-identity-disjoint`를
사용한다. Media hash는 어떤 경우에도 split 역할을 넘어 중복될 수 없다.
