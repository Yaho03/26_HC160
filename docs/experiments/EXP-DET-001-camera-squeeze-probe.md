# EXP-DET-001 — 카메라 Squeeze Probe 설계

| 항목 | 내용 |
|---|---|
| 문서명 | 카메라 기반 squeezing detector 계측 도구 설계 |
| 요구사항 ID | `EXP-DET-001` (`08_EXPERIMENT_PLAN.md`), `DET-001` (`01_RESEARCH_REQUIREMENTS.md`) |
| 상태 | 설계 확정 |
| 작성일 | 2026-09-02 |
| 기준 문서 | `07_DEFENSE_AND_DETECTION_SPEC.md`, `09_EVALUATION_METRICS.md`, `adr/ADR-003` |

---

## 1. 문제

Squeezing detector가 두 갈래로 구현돼 있고 둘 다 임계값이 검증되지 않았다.

| 구현 | 측정량 | 임계값 상태 |
|---|---|---|
| `src/face_auth/inference/feature_squeeze.py` | `1 − cos(원본, 변환)` | CLI 인자, 기본값 없음, 버전명 `feature-squeeze-unvalidated` |
| `src/verification/defenses/verification_defense_temporal_camera.py` | `abs(cos(원본,등록) − cos(변환,등록))` | 소스에 하드코딩 (`ADR-003` 위반) |

두 측정량은 서로 다른 것을 잰다. 전자는 "입력에 조작 흔적이 있는가", 후자는
"그 조작이 등록자로 위장하는 방향인가"이다. 둘 중 하나를 버릴 이유가 없으므로
독립된 두 detector로 유지하고, 각각의 임계값을 실측으로 산출한다.

임계값을 산출하려면 웹캠에서 얻은 clean 표본과 adversarial 표본이 필요하다.
현재 카메라 스크립트는 키 입력당 한 프레임만 검사하고 결과를 남기지 않으므로
표본을 모을 수 없다. 이 문서는 그 계측 도구의 설계를 정의한다.

## 2. 범위

포함한다.

- 웹캠 세션에서 프레임별 squeeze 반응을 CSV로 기록하는 도구
- 동일 프레임에서 clean 표본과 adversarial 표본을 쌍으로 생성
- 세션 메타데이터 사이드카

포함하지 않는다.

- 임계값 산출과 ROC 분석 (별도 작업)
- PAD, print/screen/replay 촬영 (`EXP-PAD-001`, 촬영 데이터 대기)
- run manifest·artifact registry 연동 (계측 단계에서는 비용이 이득을 넘는다)
- detection delay (temporal gate의 지표이며 adversarial gate와 무관)

## 3. 핵심 설계 결정

### D1. 파생값이 아니라 원시 코사인값을 저장한다

기존 결과 CSV는 `sim_diff` 같은 파생값을 저장한다. 이 경우 측정 정의를 바꾸면
기록이 무효가 되고 재촬영해야 한다.

대신 변환마다 원시 코사인값 세 개를 저장한다.

```text
cos_orig_enroll         cos(원본 임베딩,  등록 임베딩)
cos_transformed_enroll  cos(변환 임베딩,  등록 임베딩)
cos_orig_transformed    cos(원본 임베딩,  변환 임베딩)
```

이 셋으로 두 detector의 측정량이 모두 재계산된다.

```text
self_consistency = 1 − cos_orig_transformed
template_shift   = abs(cos_orig_enroll − cos_transformed_enroll)
```

앞으로 다른 측정 정의를 시도하더라도 같은 기록을 다시 쓸 수 있다. 촬영은 한 번,
분석은 여러 번이라는 전제에서 이 결정이 가장 큰 이득을 준다.

### D2. 계측은 임계값을 모른 채 수행한다

`07_DEFENSE_AND_DETECTION_SPEC.md` 6절은 detector 출력에 threshold와 hit/no-hit를
요구한다. 이는 추론 시점의 detector 계약이다. 캘리브레이션 계측에는 적용하지
않는다. 임계값을 산출하려고 수집하는 데이터에 임계값 판정을 미리 새겨 넣으면
측정하려는 대상을 입력으로 되먹이게 된다.

따라서 계측 CSV에는 `hit`, `threshold`, `detected` 컬럼을 두지 않는다. 판정은
분석 단계에서 붙인다.

### D3. 변환 14종을 세 갈래로 기록한다

두 구현이 서로 다른 변환을 쓴다. 어느 것이 유용한지는 9절의 정지 이미지 스윕에서
1차로 좁혔으나, 그 결과가 웹캠에 그대로 옮겨간다는 근거는 없다. 따라서 승자만
남기지 않고 세 갈래를 함께 기록한다.

| 갈래 | 변환 | 목적 |
|---|---|---|
| core | `blur0.5`, `blur0.8`, `blur1.2`, `median3`, `median5`, `jpeg_q30` | 스윕 상위권. 주력 후보 |
| exploratory | `blur2.0`, `median7`, `jpeg_q50`, `jpeg_q10`, `lowres64`, `bit4_floor` | 웹캠에서 순위가 뒤집힐 경우의 대안 |
| baseline | `jpeg_q75`, `bit5_round` | 현재 배포된 face_auth 설정. 개선 폭을 같은 조건에서 재기 위함 |

촬영은 사람이 앉아 있어야 하는 유일한 단계이므로 되돌리기가 가장 비싸다. 변환을
좁게 잡아 재촬영하는 비용이 배치를 키우는 비용보다 크다.

원본 1장과 변환 14장을 하나의 배치로 묶어 임베딩을 1회 forward로 계산한다. 순차
호출 대비 실시간 기록이 가능해진다.

### D4. 임계값은 clean 세션만으로 산출한다

`07_DEFENSE_AND_DETECTION_SPEC.md` 5절을 따른다. adversarial 표본은 TPR 측정에만
쓰고 임계값 선택에는 넣지 않는다. 계측 도구는 두 표본을 함께 기록하되 `label`
컬럼으로 분리해 분석 단계에서 이 규칙을 강제할 수 있게 한다.

### D5. 산출한 임계값은 상수가 아니라 artifact다

`ADR-003`을 따른다. 임계값은 model, checkpoint, preprocessing, score function,
calibration manifest, selection method에 묶인 버전 artifact로 남긴다. 기존
하드코딩 상수는 legacy로 표시하고 artifact 경로를 참조하도록 바꾼다.

### D6. 얼굴 원본과 임베딩은 산출물에 넣지 않는다

CSV와 사이드카에는 수치와 불투명 ID만 남긴다. 얼굴 크롭과 임베딩 벡터는 프로세스
메모리에만 존재하고 디스크에 쓰지 않는다. `subject_id`는 사용자가 지정하는 불투명
라벨이며 실명·이메일·파일 경로를 담지 않는다.

## 4. 구조

기존 카메라 스크립트는 469줄에 UI, 공격 생성, 탐지 계산이 함께 있다. 계측까지
넣으면 유지할 수 없으므로 계산과 기록을 분리한다.

| 모듈 | 책임 | 의존 |
|---|---|---|
| `src/verification/defenses/squeeze_probe.py` | 순수 계산. 얼굴 크롭과 등록 임베딩을 받아 변환별 원시 코사인값을 반환. 카메라·파일 IO 없음 | 임베더 인터페이스 |
| `src/verification/defenses/probe_log.py` | CSV 행과 세션 사이드카 기록. 스키마 고정, 금지 필드 차단 | 없음 |
| `src/verification/defenses/probe_capture.py` | 계측 세션 CLI. 카메라 루프, 등록 확보, 공격 주기 제어 | 위 두 모듈, 데모의 `detect_and_crop`·`generate_adversarial` |
| `src/verification/defenses/facenet_embed.py` | `embed_batch`와 `FaceNetBatchEmbedder` 추가 | 기존 모듈 |

기록 루프를 기존 데모(`verification_defense_temporal_camera.py`)에 넣지 않고 별도
모듈로 둔다. 기록은 비대화식으로 정해진 표본 수까지 도는 반면 데모는 키 입력에
반응하고 판정을 화면에 그린다. 한 루프에 합치면 분기만 늘고 데모도 계측도 읽기
어려워진다. 데모의 얼굴 검출과 PGD 생성은 그대로 재사용하므로 로직 중복은 없다.

`squeeze_probe`가 카메라와 분리되므로 스텁 임베더로 단위 테스트가 가능하고,
face_auth 게이트에서도 같은 모듈을 재사용할 수 있다.

## 5. 기록 스키마

### 5.1 CSV — 행 단위 형식

표본 하나가 변환 6종에 대해 6행을 만든다. 변환을 추가하거나 제거해도 스키마가
바뀌지 않고, 변환별·측정량별 ROC를 pivot 한 번으로 뽑을 수 있다.

| 컬럼 | 설명 |
|---|---|
| `session_id` | 세션 불투명 ID |
| `subject_id` | 피험자 불투명 라벨 |
| `sample_id` | 표본 ID. 같은 프레임의 clean과 adversarial은 접미사로 구분 |
| `frame_idx` | 세션 내 프레임 순번 |
| `frame_ts_ms` | 캡처 시각. 단조 시계 기준 경과 밀리초 |
| `dropped_frames` | 해당 시점까지 누적 드롭 프레임 수 |
| `label` | `clean` 또는 `adversarial` |
| `transform` | 변환 이름 |
| `cos_orig_enroll` | D1 참조 |
| `cos_transformed_enroll` | D1 참조 |
| `cos_orig_transformed` | D1 참조 |
| `embed_ms` | 해당 표본의 임베딩 배치 소요 시간 |

`frame_ts_ms`와 `dropped_frames`는 `07_DEFENSE_AND_DETECTION_SPEC.md` 5절이
카메라 실험에 요구하는 항목이다. `embed_ms`는 7절의 latency budget 판정에 쓴다.

### 5.2 세션 사이드카 JSON

```text
session_id, subject_id, created_at, git_commit
camera            : index, width, height, fps_nominal, fourcc
model             : name, pretrained, weights_file, weights_sha256, preprocess
transforms        : 이름과 파라미터
attack            : kind, epsilon, steps, step_size, every
counters          : frames_read, read_failures, frames_without_face,
                    samples_clean, samples_adversarial, rows
target_frames     : 목표 표본 수
completed         : 목표에 도달했는지
interrupted_by    : 중단 사유. null이면 정상 종료
elapsed_sec, effective_fps
jpeg_headroom_q75 : 아래 참조
```

임계값 artifact가 참조할 provenance가 여기에 모인다.

사이드카는 `finally`에서 기록한다. 중단된 세션도 provenance를 남겨야 한다. CSV만 남고
사이드카가 없으면 어떤 모델과 파라미터로 얻은 값인지 알 수 없어 분석에 쓸 수 없다.
중단 여부는 `completed`와 `interrupted_by`로 구분한다.

`jpeg_headroom_q75`는 JPEG 재압축이 실제로 픽셀을 바꾸는 정도다. 0~255 척도의 평균
절대 변화량이며, 값이 0에 가까우면 입력이 이미 JPEG 압축돼 있어 JPEG 계열 변환이
탐지 신호를 만들지 못한다는 뜻이다. macOS AVFoundation은 `CAP_PROP_FOURCC`를 보고하지
않으므로 코덱 대신 이 값으로 판단한다. 기준값은 LFW 정지 JPEG 0.076, 웹캠 160x160
크롭 2.6~3.2다.

## 6. 표본 수집 방식

```bash
python -m src.verification.defenses.probe_capture --subject p01 --frames 300
```

등록 얼굴을 확보한 뒤 매 프레임에서 얼굴을 크롭해 clean 표본을 기록하고,
`--attack-every N` 프레임마다 그 프레임에 PGD를 적용해 adversarial 표본을 추가로
기록한다. `0`이면 공격 생성을 끈다. `Esc` 또는 `q`로 중단해도 그 시점까지의
CSV와 사이드카가 남는다.

같은 프레임에서 나온 clean과 adversarial이 쌍을 이루므로 조명과 자세가 교란요인으로
작용하지 않는다. PGD는 느리지만 N프레임에 한 번만 수행하므로 기록이 끊기지 않는다.

기존 데모의 `C`와 `A` 단발 검사는 시연용으로 그대로 둔다.

`dropped_frames`는 `read()` 실패 횟수이며 드라이버 수준의 드롭 카운트가 아니다.
OpenCV의 `VideoCapture`는 내부 버퍼 드롭을 보고하지 않으므로, 실제 처리 속도는
사이드카의 `effective_fps`로 판단한다. 이 한계를 사이드카 `notes`에도 남긴다.

## 7. 완료 조건

- `squeeze_probe`가 스텁 임베더로 검증된다. 변환 개수, 배치 순서, 원시 코사인값 세 개의 계산식.
- `probe_log`가 스키마를 고정하고 금지 필드를 거부한다.
- 기록 세션이 `표본수 × 변환수`행을 만들고 `label`에 두 값이 모두 나타난다.
- 같은 `frame_idx`로 clean 행과 adversarial 행을 조인할 수 있다.
- CSV와 사이드카에 얼굴 이미지, 임베딩 벡터, 절대 경로가 없다.
- 세션 사이드카가 model, 변환 파라미터, 공격 파라미터, git commit을 남긴다.

## 7.1 분석 도구

계측 CSV는 `src/verification/defenses/probe_analyze.py`가 읽는다.

```bash
python -m src.verification.defenses.probe_analyze \
    --probe outputs/probe/<session_id>/probe.csv --target-fpr 0.01
```

특징별 ROC-AUC, clean 전용 임계값, 그 임계값에서의 TPR/FPR/precision/recall과 표본
수를 낸다. 지표 집합은 `09_EVALUATION_METRICS.md` 4절을 따르고, 분모가 0이면 0이
아니라 undefined를 반환한다.

`threshold_at_fpr`는 목표 FPR을 넘지 않는 가장 낮은 관측값을 고른다. 분위수 선형
보간은 관측되지 않은 임계값을 만들고 동점 처리에서 목표를 초과할 수 있다. 표본이
부족해 `target_fpr * n < 1`이면 어떤 관측값으로도 목표를 만족할 수 없으므로 최댓값
위로 올려 FPR 0을 택한다. 이 경우 표본 수를 함께 보고해야 한다.

같은 함수는 라벨이 섞인 입력을 예외로 거부한다. D4의 규칙을 코드로 강제한 것이다.
결합 규칙도 clean 통계로만 정규화하며 공격 라벨로 가중치를 학습하지 않는다.

## 7.2 Threshold artifact 생성

```bash
python -m src.verification.defenses.probe_threshold \
    --probe outputs/probe/<session_id>/probe.csv \
    --session outputs/probe/<session_id>/session.json \
    --target-fpr 0.01 --out outputs/probe/<session_id>/detector_threshold.json
```

`schemas/detector-threshold-artifact.schema.json`을 따른다. `calibration`에는 clean
표본만, `evaluation`에는 adversarial 표본만 들어간다. 구조가 이 분리를 드러낸다.

clean 표본 수가 `target_fpr * n < 1`이면 임계값을 만들지 않고 예외를 낸다. 관측값으로
목표를 만족할 수 없는 상황에서 조용히 FPR 0으로 낮추면 근거 없는 임계값이 artifact로
굳는다. 목표 FPR 1%에는 clean 표본이 최소 100개 필요하다.

`limitations`는 사이드카에서 자동 도출한다. 피험자 수, 세션 수, 공격 종류 수를 세고
adaptive attack과 clean TAR delta 미측정을 항상 명시한다. 사람이 적기를 기다리지 않는다.

### D8. 한 세션에서 여러 공격을 함께 모은다

단일 공격 종류로 산출한 임계값은 그 공격의 지문을 외운 것과 구별되지 않는다. 촬영은
사람 시간이 들어 되돌리기가 가장 비싸므로, 공격 기회마다 종류를 번갈아 써서 한 세션이
여러 공격을 덮게 한다.

| 종류 | 파라미터 | 목적 |
|---|---|---|
| `pgd` | epsilon, steps, step_size | 표준 반복 공격 |
| `fgsm` | 1스텝, step_size = epsilon | perturbation 구조가 PGD와 달라 squeeze 반응도 다르다 |
| `pgd_low_eps` | epsilon x 0.25 | 더 작은 예산. 탐지가 어려운 쪽 경계 |

모든 종류는 같은 PGD 생성기를 파라미터만 바꿔 호출한다. 별도 구현을 두면 전처리와
정규화가 어긋날 수 있다. 공격 파라미터는 촬영 시작 전에 검증한다. 촬영을 다 하고
실패하면 사람 시간을 버린다.

CSV의 `attack_kind` 컬럼과 artifact의 `evaluation.tpr_by_attack_kind`가 종류별 결과를
분리한다. `07_DEFENSE_AND_DETECTION_SPEC.md` 7절은 공격 성공률을 단일 평균으로 숨기지
말 것을 요구한다. 종류별 표본이 적을 수 있으므로 분자와 분모를 함께 낸다.

`attack_kind` 컬럼이 없던 초기 세션은 `unspecified`로 표기한다. 소급 적용하지 않는다.

## 7.3 방어 전후 비교

`07_DEFENSE_AND_DETECTION_SPEC.md` 7절의 통과 기준은 두 가지다. conditional ASR을
50% 이상 줄이고, 고정 threshold에서 clean TAR 감소가 2%p 이하여야 한다.

```bash
python -m src.verification.defenses.probe_threshold \
    --probe outputs/probe/<session_id>/probe.csv \
    --session outputs/probe/<session_id>/session.json \
    --target-fpr 0.01 --window-frames 3 \
    --identity-threshold 0.47966246581077576 \
    --out outputs/probe/<session_id>/detector_threshold.json
```

`--identity-threshold`를 주면 `evaluation.defense_comparison`에 판정이 들어간다.
`09_EVALUATION_METRICS.md` 3절의 세 지표는 분모가 다르므로 이름과 분모를 함께 기록한다.

Eligible attempt는 방어 전 accept된 공격이다. 방어 전에 이미 거부된 공격을 분모에
넣으면 방어 성능이 부풀려진다. 분모가 0이면 예외를 낸다.

세션 `7b94fe4d1971` 결과다.

| 기준 | 목표 | 실측 | 판정 |
|---|---|---|---|
| conditional ASR 감소 | 50% 이상 | 67.2% (1.000 → 0.328) | 충족 |
| clean TAR 감소 | 2%p 이하 | 0.00%p (1.0000 → 1.0000) | 충족 |

방어 전 공격 58건이 모두 성공했다. 신원 유사도 중앙값이 0.9964로 clean의 0.8909보다
높다. 공격이 등록자보다 더 등록자에 가까운 임베딩을 만든다는 뜻이며, 방어 없이는
이 공격을 하나도 막지 못한다.

이 판정은 `07` 7절이 "잠정"이라고 명시한 연구용 기준이며 운영 보장을 뜻하지 않는다.
남은 한계는 아래 11절을 참조한다.

## 8. 이후 작업

1. 표본 확대. 현재 단일 피험자·단일 세션이므로 일반화 근거가 없다.
2. 여러 공격 종류로 촬영. 도구는 준비됐으나 아직 `pgd` 단일 세션만 있다.
3. Adaptive attack 평가. 공격자가 detector를 아는 경우의 내성.
4. 하드코딩 상수를 artifact 참조로 교체.

## 10. 집계 단위 (D7)

임계값은 적용 단위와 같은 단위로 산출해야 한다. `FeatureSqueezeInspector`는 최근
`max_frames`(기본 3)개 중 최악값을 게이트 점수로 쓴다. 프레임 단위로 정한 임계값을
이 게이트에 그대로 넣으면 실현 FPR이 윈도 크기만큼 배가된다.

세션 `7b94fe4d1971`에서 측정한 값이다. 프레임 단위 임계값 9.51을 고정하고 윈도만
바꿨다.

| 윈도 | 세션 FPR | 독립 가정 | 세션 TPR | clean TAR delta |
|---|---|---|---|---|
| 1 | 1.00% | 1.00% | 0.733 | -1.00%p |
| 2 | 2.01% | 1.99% | 0.881 | -2.01%p |
| 3 | 3.02% | 2.97% | 0.948 | -3.02%p |
| 5 | 5.07% | 4.90% | 1.000 | -5.07%p |

기본 설정인 윈도 3에서 clean TAR 감소가 3.02%p로 `07_DEFENSE_AND_DETECTION_SPEC.md`
7절 예산 2%p를 초과한다. 실현 FPR이 독립 가정과 거의 일치하는데, 연속 프레임인데도
특징값 상관이 낮다는 뜻이다. 촬영 중 얼굴이 계속 움직이므로 프레임 간 squeeze 반응이
독립에 가깝다.

따라서 artifact는 `aggregation` 블록에 적용 단위, 윈도 크기, 집계 규칙을 함께 기록하고,
임계값도 집계 후 표본으로 산출한다. 집계 방식 없이 임계값만 기록하면 같은 숫자가 프레임
단위로도 세션 단위로도 해석되어 실현 FPR이 세 배 달라진다. `aggregation.window_frames`는
`FeatureSqueezeConfig.max_frames`와 일치해야 한다.

## 11. 첫 artifact (세션 `7b94fe4d1971`)

| 항목 | 값 |
|---|---|
| 특징 | 상위 6개 결합, clean 통계로 z 정규화 후 합산 |
| 집계 | session, 윈도 3, max |
| 목표 FPR | 0.01 |
| 달성 FPR | 0.00 |
| TPR | 0.672 |
| ROC-AUC | 0.982 |
| clean 표본 / 윈도 | 300 / 298 |
| adversarial 윈도 | 58 |
| clean TAR delta | -0.00%p (예산 2%p 충족) |

이 값은 release threshold가 아니다. `limitations` 다섯 건 중 어느 하나도 해소되지
않았다. TPR 0.672는 적대적 입력 세 개 중 하나를 놓치는 수준이다. 파이프라인이 계측부터
artifact까지 이어진다는 것을 보이는 용도로만 쓴다.

`limitations`는 측정 결과에 따라 달라진다. clean TAR delta를 측정하면 그 항목이 목록에서
빠지고, 예산을 초과하면 초과 사실이 대신 들어간다. 측정한 것을 미측정이라고 적으면 한계
목록 전체의 신뢰가 떨어지므로 이 정합성을 테스트로 고정했다.

## 9. 정지 이미지 스윕 (2026-09-02)

웹캠 촬영 전에 변환 후보를 좁히려고 기존 공격 패키지로 1차 스윕을 돌렸다.

| 항목 | 값 |
|---|---|
| 데이터 | `facenet_verification_attack_package` 120쌍 |
| clean | `source_file` (LFW 원본) |
| adversarial | `adv_file` (targeted PGD, epsilon 0.005 고정) |
| 후보 | 변환 19종 x 측정량 2종 |
| 지표 | ROC-AUC (`09_EVALUATION_METRICS.md` 4절) |

### 9.1 결론 세 가지

`jpeg_q75`가 약한 이유는 원본이 이미 JPEG이라 quality 75 재압축이 거의 무손실이기
때문이다. quality를 낮추면 회복된다. quality 10까지 내리면 clean도 함께 손상돼 다시
떨어진다.

| 변환 | self_consistency | template_shift |
|---|---|---|
| `jpeg_q75` | 0.678 | 0.595 |
| `jpeg_q50` | 0.994 | 0.964 |
| `jpeg_q30` | 0.996 | 0.991 |
| `jpeg_q10` | 0.990 | 0.980 |

이 결론은 입력이 이미 JPEG일 때만 성립한다. 파일 확장자와는 무관하다. 변환은 디코딩된
픽셀 배열에 적용되므로 원본이 어떤 확장자로 저장돼 있었는지는 영향이 없다. 문제는 픽셀
값이 이미 JPEG 양자화 격자 위에 놓여 있다는 것이다.

LFW 원본 30장으로 측정한 평균 절대 픽셀 변화량은 다음과 같다. 척도는 0~255다.

| 조건 | 변화량 | 비고 |
|---|---|---|
| q75 1차 적용 | 0.076 | 사실상 변화 없음 |
| q75 2차 적용 | 0.031 | 1차의 41%. 재압축이 이미 고정점에 수렴 |
| q30 1차 적용 | 3.031 | q75의 40배 |
| 노이즈 추가 후 q75 적용 | 1.200 | 원본 q75의 15.8배 |

마지막 행이 중요하다. 센서 노이즈를 더해 픽셀을 JPEG 양자화 격자에서 벗어나게 하면
같은 q75가 15.8배 더 크게 작용한다. 즉 `jpeg_q75`가 약하다는 결론은 정지 JPEG 입력에
한정되며, 웹캠이 raw 프레임을 주면 성립하지 않을 수 있다.

반대로 카메라가 MJPEG 스트림을 주면 프레임이 이미 JPEG 압축된 상태이므로 LFW와 같은
상황이 된다. 어느 쪽인지는 카메라와 해상도·fps 설정에 따라 달라진다. 따라서 세션
사이드카에 `camera.fourcc`를 기록하고, JPEG 계열 결과를 이 값 없이 해석하지 않는다.
`jpeg_q75`를 baseline 갈래에 남겨 둔 이유이기도 하다.

`lowres`는 self_consistency와 구조적으로 맞지 않는다. 축소는 perturbation뿐 아니라
얼굴 구조 자체를 지우므로 clean에서도 임베딩이 크게 이동한다. 해상도를 올리면
개선되지만 끝까지 blur/median에 미치지 못한다.

| 변환 | self_consistency | template_shift |
|---|---|---|
| `lowres32` | 0.559 | 0.909 |
| `lowres64` | 0.755 | 0.985 |
| `lowres112` | 0.807 | 0.981 |

두 측정량을 결합하면 각각보다 나아진다. clean 통계로만 정규화하고 고정 규칙으로
합쳤다. 공격 라벨은 TPR 측정에만 사용했다.

| 규칙 | AUC | TPR at FPR 1% |
|---|---|---|
| `blur0.8` self_consistency 단독 | 0.9996 | 0.983 |
| `blur1.2` template_shift 단독 | 0.9990 | 0.983 |
| 두 측정량 합 | 1.0000 | 1.000 |
| 상위 6개 특징 합 | 1.0000 | 1.000 |
| 전체 38개 특징 합 | 0.9998 | 0.992 |

약한 특징을 모두 넣으면 오히려 떨어진다. 소수의 강한 특징만 결합해야 한다.

### 9.2 이 수치를 성능 주장으로 쓰지 않는다

AUC 1.0000은 좋은 결과가 아니라 설정이 지나치게 쉽다는 신호다.

- 공격이 한 종류다. 전부 동일 파라미터의 targeted PGD, epsilon 0.005다. 탐지기가 특정 공격의 지문을 외운 것과 구별되지 않는다.
- LFW 정지 이미지다. 웹캠 노이즈가 없어 clean 분산이 비현실적으로 작다.
- 적응형 공격을 평가하지 않았다. `07_DEFENSE_AND_DETECTION_SPEC.md` 7절이 요구하는 항목이다.
- 120쌍이므로 신뢰구간이 넓다. 점추정만으로 판단할 수 없다.

이 스윕의 유일한 용도는 웹캠 촬영에 넣을 변환 후보를 좁히는 것이다. 보고서에
detector 성능으로 인용하지 않는다.

## 12. Adaptive attack 평가 (2026-09-02)

지금까지의 결과는 공격자가 detector의 존재를 모른다는 가정에 기댄다. 그 가정이
깨질 때 무슨 일이 생기는지 측정했다.

```bash
python -m src.verification.defenses.adaptive_attack \
    --package <공격 패키지 루트> \
    --probe outputs/probe/7b94fe4d1971/probe.csv \
    --identity-threshold 0.47966246581077576 \
    --limit 25 --weights 0,1,5
```

EOT는 변환을 공격 루프에 넣어, 등록자 유사도를 높이면서 변환 전후 임베딩 차이를
작게 유지하도록 최적화한다. 두 번째 항이 탐지 회피다. `consistency_weight`가
회피에 두는 비중이다.

### 12.1 결과

표본 25쌍, 임계값 9.5102 (프레임 단위, 목표 FPR 1%).

| 공격 | 탐지율 | 인증 성공 | 둘 다 | 점수 중앙값 |
|---|---|---|---|---|
| 방어 무지 (표준 PGD) | 100% | 100% | 0% | 136.8 |
| EOT `w=1` | 100% | 100% | 0% | 64.2 |
| EOT `w=5` | 100% | 100% | 0% | 34.2 |

"둘 다"는 인증을 통과하면서 탐지도 회피한 비율이며 실제 공격 성공률이다.

### 12.2 숫자보다 추세가 중요하다

탐지율은 모두 100%지만 점수 중앙값이 136.8에서 34.2로 내려갔다. 회피 가중치를
올릴수록 계속 낮아진다. **EOT는 작동하고 있으며 다만 아직 임계값에 도달하지
못했다.** 34.2는 임계값 9.51의 3.6배이므로 여유가 있지만, 추세가 하락 중이므로
가중치를 더 올리거나 스텝을 늘리면 결과가 달라질 수 있다.

따라서 "방어가 adaptive attack에 강하다"가 아니라 "이 설정의 EOT는 임계값에 도달하지
못했다"가 정확한 표현이다.

### 12.3 강건성의 출처

detector가 쓰는 상위 6개 특징 중 5개가 미분 불가한 변환에서 나온다.

```text
jpeg_q75|sc, jpeg_q50|sc, blur2.0|ts, jpeg_q30|sc, median3|sc, median5|sc
```

EOT는 미분 가능한 blur만 최적화할 수 있고, 점수의 대부분을 차지하는 JPEG와 median
항은 직접 건드리지 못한다. 방어의 강건성이 상당 부분 이 비대칭에서 나온다는 뜻이다.

이는 설계 우위지만 BPDA 같은 근사 기법으로 우회할 수 있는 종류의 우위다. 미분 불가
연산의 backward pass를 항등함수로 근사하면 EOT가 그 항들도 최적화할 수 있다.

### 12.4 말할 수 있는 것과 없는 것

말할 수 있다.

- Gradient 기반 EOT로는 이 detector를 뚫지 못했다.
- EOT가 탐지 점수를 4분의 1로 낮추는 데는 성공했다.
- 강건성이 미분 불가 변환에 상당 부분 의존한다.

말할 수 없다.

- Adaptive attack 전반에 강건하다. BPDA를 평가하지 않았다.
- 더 강한 EOT 설정에도 견딘다. 추세가 하락 중이다.

`07_DEFENSE_AND_DETECTION_SPEC.md` 7.1절이 이 구분을 규범으로 고정한다.
