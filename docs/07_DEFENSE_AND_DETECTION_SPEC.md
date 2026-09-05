# DEFENSE AND DETECTION SPEC — 방어·탐지 사양

| 항목 | 내용 |
|---|---|
| 문서명 | 방어 및 탐지 사양서 |
| 버전 | v1.3 |
| 상태 | 확정 |
| 최종 수정일 | 2026-09-06 |

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

### 6.1 Squeezing detector의 두 갈래

Squeezing detector는 서로 다른 것을 재는 두 측정량으로 나뉜다. 하나의 게이트로
합치지 않는다.

| 게이트 | 측정량 | 보는 것 |
|---|---|---|
| `adversarial` | `1 - cos(원본, 변환)` | 입력에 조작 흔적이 있는가. 등록 템플릿과 무관 |
| `adversarial_template` | `abs(cos(원본,등록) - cos(변환,등록))` | 그 조작이 등록자로 위장하는 방향인가 |

두 측정량은 변환의 유용성 순위를 다르게 매긴다. 웹캠 실측에서 한쪽에 무력한 변환이
다른 쪽에서는 유효했다. 따라서 임계값을 각각 산출하고, 각각 독립된 optional veto로
정책에 참여한다. 둘 중 하나만 배선해도 동작한다.

변환과 임베딩은 프레임당 한 번만 계산해 두 게이트가 공유한다.

### 6.2 임계값은 적용 단위와 같은 단위로 산출한다

게이트가 여러 프레임을 묶어 판정하면, 프레임 단위로 정한 임계값의 실현 FPR이
윈도 크기만큼 커진다. 임계값 artifact에 적용 단위, 윈도 크기, 집계 규칙을 함께
기록하고 캘리브레이션도 같은 단위로 수행한다.

집계 방식 없이 임계값만 기록하면 같은 숫자가 프레임 단위로도 세션 단위로도 해석되어
실현 FPR이 달라진다. 이는 `adr/ADR-003`이 요구하는 threshold provenance의 일부다.

### 6.3 추론 계약과 계측 계약의 구분

위 출력 목록은 **추론 시점** detector의 계약이다. 임계값을 산출하기 위한
**캘리브레이션 계측**에는 적용하지 않는다.

계측 데이터에 threshold와 hit/no-hit를 미리 새겨 넣으면 측정하려는 대상을 입력으로
되먹이게 된다. 임계값이 아직 없는 단계에서 임의값으로 판정을 기록하면, 이후 분석은
그 임의값에 조건부인 결과만 산출한다.

따라서 계측 산출물은 다음을 따른다.

- 판정 없는 원시 관측값만 기록한다. `hit`, `threshold`, `detected` 컬럼을 두지 않는다.
- 파생 점수가 아니라 재계산 가능한 원시량을 기록한다. 측정 정의가 바뀌어도 재촬영하지 않기 위해서다.
- 판정과 지표 산출은 분석 단계에서 수행하고, 그때 threshold artifact를 참조한다.

`EXP-DET-001`의 계측 스키마가 이 규칙의 적용 예다.
`experiments/EXP-DET-001-camera-squeeze-probe.md`를 참조한다.

Detector threshold는 `schemas/detector-threshold-artifact.schema.json`을 따른다.
verification threshold와 분모가 다르므로 `threshold-artifact.schema.json`을 재사용하지
않는다. verification은 genuine/impostor 쌍에서 FAR/TAR을 재고 detector는
clean/adversarial 표본에서 FPR/TPR을 잰다.

## 7. 잠정 통과 기준

다음 조건을 만족하면 방어를 유망하다고 표현할 수 있다.

- No-defense 대비 conditional attack ASR을 50% 이상 감소
- 고정 operating threshold에서 clean TAR 감소가 2 percentage point 이하
- 95% confidence interval과 모든 error 보고
- 명시한 reference hardware의 latency budget 충족 (7.2절)

이 값은 연구용 잠정 기준이며 운영 보장을 뜻하지 않는다. 기준을 변경하려면 최종 test
결과를 보기 전에 기록한다.

### 7.1 판정에 필요한 분모

Conditional ASR의 분모는 방어 전 accept된 공격이다. 방어 전에 이미 거부된 공격을
분모에 넣으면 방어 성능이 부풀려진다. 분모가 0이면 0이 아니라 명시적 오류를 반환한다.

Clean cost의 분자에서는 방어 전에 이미 거부된 clean 표본을 뺀다. 방어 탓이 아니다.

두 기준을 모두 만족해도 다음이 확인되지 않으면 방어 성능을 주장하지 않는다.

- 여러 피험자와 세션
- 여러 공격 종류
- Adaptive attack. 공격자가 detector를 알고 있는 경우

특히 gradient 기반 adaptive attack만 평가한 결과로 adaptive robustness를 주장하지
않는다. 미분 불가한 변환은 gradient 공격이 직접 최적화할 수 없어 방어에 유리하게
보이지만, BPDA 같은 근사 기법으로 우회할 수 있다. 어떤 공격을 평가했고 어떤 공격을
평가하지 않았는지 함께 기록한다.

### 7.2 Reference hardware와 latency budget

7절의 네 기준 중 latency만 값이 정의되어 있지 않았다. 기준이 있는데 값이 없으면 영원히
판정할 수 없으므로 값을 정의한다.

#### 7.2.1 이 절은 사후에 기록됐다

**먼저 이 사실을 적는다.** 7절은 "기준을 변경하려면 최종 test 결과를 보기 전에
기록한다"고 요구한다. 이 절은 그 요구를 지키지 못했다. `PERF-001` 측정
(`docs/experiments/PERF-001-detector-latency.md`)이 2026-09-05에 먼저 있었고 이 절은
2026-09-06에 쓰였다.

따라서 **이 기준으로 얻는 "충족" 판정은 독립적 근거가 아니다.** 아래 값은 측정값을
이미 알고 있는 상태에서 정해졌으며, 측정값이 경계의 어느 쪽에 떨어지는지 알면서 경계를
그은 것이다. 이 절을 근거로 방어가 성능 요건을 만족한다고 주장하면 안 된다.

이 절의 목적은 판정이 아니라 **다음 측정을 위한 사전 등록**이다. 값이 한 번 고정되면
이후의 측정은 정직하게 판정될 수 있다. 값을 바꾸려면 다음 측정 전에 바꾼다.

#### 7.2.2 Budget 값

측정값이 아니라 외부 기준에서 위에서 아래로 배분한다. 우리 수치에서 역산하면 기준이
아니라 결과의 다른 표현이 된다.

출발점은 대화형 응답 시간의 통상적 한계다. 1초를 넘으면 사용자가 흐름이 끊겼다고
인지한다는 것이 HCI에서 오래 인용돼 온 값이며(Nielsen, *Usability Engineering*, 1993의
0.1초 / 1초 / 10초 한계), 이 프로젝트의 어떤 측정과도 무관하게 정해진 값이다.

| 대상 | Budget (p95) |
|---|---|
| 인증 판정 종단 (캡처 종료 → 수락·거부 표시) | 1,000 ms |
| 그중 detector 게이트 | **300 ms** |

배분 근거. detector 게이트는 정상 verification 위에 얹는 부차적 veto다. 주 경로(얼굴
검출, verification embedding, policy)에 700 ms를 남긴다. 게이트가 종단 예산의 3분의 1을
넘으면 부차적이라고 부를 수 없다.

**이 3 대 7 배분은 판단이며 외부 근거가 없다.** 그리고 측정값(배치화 전 616 ms, 후
224 ms)이 이 경계의 양쪽에 놓인다는 것을 알면서 정했다. 7.2.1절의 제약이 그대로
적용된다.

#### 7.2.3 Reference hardware

Budget 판정에 쓸 수 있는 측정은 다음을 모두 만족해야 한다.

- 실행 아키텍처가 호스트 CPU와 일치한다. **번역 계층(Rosetta 2, QEMU 등) 위의 실행은
  reference hardware로 인정하지 않는다.** 번역 실행은 네이티브 커널 경로(NNPACK 등)를
  비활성화시키므로 절대 시간이 하드웨어를 대표하지 않는다.
- 기기, OS, Python 빌드 아키텍처, 프레임워크 버전, 연산 장치를 모두 기록한다.
- p95는 nearest-rank로 정의하고, 워밍업 10회 이상을 제외한 표본 50개 이상에서 낸다.
- 입력은 단일 크롭 반복이 아니라 실제 세션의 프레임 분포를 쓴다.

#### 7.2.4 현재 판정 — 불가

`PERF-001` 측정은 Python이 x86_64 빌드로 Rosetta 2 번역 실행된 환경에서 나왔다.
7.2.3절에 따라 **reference hardware 요건을 만족하지 않는다.** 같은 환경 안의 상대
비교(배치화 8.30배 등)는 유효하지만 budget 충족 여부의 근거로 쓸 수 없다.

또한 입력이 크롭 1장 반복이므로 7.2.3절의 마지막 항목도 만족하지 않는다.

따라서 latency 기준의 상태는 다음과 같다.

| 항목 | 상태 |
|---|---|
| Budget 값 | 정의됨 (7.2.2절, 단 7.2.1절의 제약을 받음) |
| Reference hardware 정의 | 정의됨 (7.2.3절) |
| 적합한 측정 | **없음** |
| 충족 판정 | **불가** |

판정하려면 네이티브 arm64 빌드에서 실제 세션 프레임으로 다시 재야 한다. 그 측정은 이
절이 먼저 기록된 뒤에 이루어지므로 사전 등록 요건을 만족한다.
