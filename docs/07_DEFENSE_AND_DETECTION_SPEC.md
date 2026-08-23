# DEFENSE AND DETECTION SPEC — 방어·탐지 사양

| 항목 | 내용 |
|---|---|
| 문서명 | 방어 및 탐지 사양서 |
| 버전 | v1.0 |
| 상태 | 확정 |
| 최종 수정일 | 2026-08-23 |

---

## 1. Component 유형

| 유형 | 출력 | 예시 |
|---|---|---|
| `transform` | 방어 적용 image와 처리 시간 | JPEG, smoothing, bit-depth, ROI |
| `trained` | 버전이 있는 defended checkpoint와 score | Adversarial training |
| `ensemble` | 결합 decision과 component vote | ROI/smoothing/randomized vote |
| `temporal` | 실제 frame에서 얻은 session-level decision | Continuity/static/replay check |
| `detector` | Suspicion score와 evidence | Feature squeezing consistency |

이 유형들은 모호한 하나의 `defense_success` 정의를 공유하면 안 된다.

## 2. Transform defense

Transform defense는 input artifact를 derived artifact로 변환한다. Parent hash, transform
parameter, encoder/decoder version, processing time과 변환 후 verification result를 기록한다.

Attack input에서는 방어 전 accept였던 공격이 방어 후 reject일 때만 성공이다. Clean cost는
별도로 측정한다.

## 3. Trained defense

- Training row, model-selection row와 최종 test row를 분리한다.
- Best checkpoint는 validation data만으로 선택한다.
- 최종 평가는 clean pair, held-out attack, unseen attack과 defended checkpoint 대상 adaptive attack을 포함한다.
- Training-set ASR은 진단값이며 최종 증거가 아니다.

## 4. Randomized·ensemble method

Randomized smoothing은 가정과 radius를 포함한 공식 certificate를 생성하지 않는 한
stochastic heuristic으로 표현한다. Ensemble report는 모든 component vote, missing/error
vote, tie policy와 total latency를 기록한다.

## 5. Temporal·camera method

하나의 still image를 합성 증강한 결과는 simulation이며 실제 temporal robustness의 증거가
아니다. 유효한 camera experiment에는 다음이 필요하다.

- Genuine multi-frame session
- Replay/print/screen session
- Subject와 session 분리
- Frame timestamp와 drop 수
- Normal session만 사용한 calibration
- False-positive rate와 detection delay

## 6. Detection 의미

Feature squeezing 등의 method는 다음을 출력한다.

- Detector score
- Detector threshold/version
- Hit/no-hit
- Transformation별 evidence
- False-positive 및 false-negative metric

Detector는 risk score 또는 veto policy에 사용될 수 있지만 detection rate는 defense
success rate나 verification accuracy가 아니다.

## 7. 잠정 통과 기준

다음 조건을 만족하면 방어를 유망하다고 표현할 수 있다.

- No-defense 대비 conditional attack ASR을 50% 이상 감소
- 고정 operating threshold에서 clean TAR 감소가 2 percentage point 이하
- 95% confidence interval과 모든 error 보고
- 명시한 reference hardware의 latency budget 충족

이 값은 연구용 잠정 기준이며 운영 보장을 뜻하지 않는다. 기준을 변경하려면 최종 test
결과를 보기 전에 기록한다.
